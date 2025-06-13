from attrdict import AttrDict
import math
import random
import argparse
import os
import shutil
import time
import json
import functools
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.utils.data import ConcatDataset, random_split   
from itertools import combinations,islice
from math import comb
from collections import OrderedDict



# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.losses import cross_entropy, ova
from lib.experts import SyntheticExpertOverlap, PreComputedExpert
from lib.modules import ClassifierRejector, ClassifierRejectorWithContextEmbedder
from lib.datasets import load_cifar, load_gtsrb, ContextSampler , load_fashion_mnist
from lib.wideresnet import WideResNetBase



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _average_metrics(list_of_dicts):
    """
    Helper that averages numeric metrics across multiple experts.
    If a field is string (like "cov" 'xx/yy'), we store the first.
    Adjust as you see fit.
    """
    if not list_of_dicts:
        return {}

    n = len(list_of_dicts)
    # Use the first dict to get all keys and identify numeric vs. non-numeric
    if not list_of_dicts:
        return {}

    sum_dict = {}
    # Initialize with the first dictionary to handle non-numeric types correctly
    for k, v in list_of_dicts[0].items():
        if not isinstance(v, (int, float)):
            sum_dict[k] = v # Store the first non-numeric value

    for d in list_of_dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0.0) + v

    # Build the average dict
    avg_dict = {}
    for k, v in sum_dict.items():
        if isinstance(v, (int, float)):
            avg_dict[k] = v / n
        else:
            avg_dict[k] = v  # Keep the first non-numeric value
    return avg_dict


def _perform_evaluation_run(model,
                            experts_for_run,
                            loss_fn,
                            cntx_sampler,
                            n_classes,
                            data_loader,
                            config,
                            budget,
                            n_finetune_steps,
                            lr_finetune,
                            p_cntx_inclusion,
                            device):
    """
    Performs a single, full evaluation pass over the data_loader.
    This helper contains the core logic that was duplicated.

    Args:
        experts_for_run (list): A list of experts to use for this run.
            - If len > 1, a random expert is chosen per batch.
            - If len == 1, that single expert is used for all batches.
    """
    model.eval()
    if config["l2d"] == 'single_maml':
        model.train()

    is_finetune = (config["l2d"] in ['single', 'single_maml']) and (n_finetune_steps > 0)
    
    # Backup model state if we are finetuning during evaluation
    if is_finetune:
        model_state_dict = copy.deepcopy(model.state_dict())
        # The model object backup is needed if finetuning modifies more than just state_dict
        model_backup = copy.deepcopy(model) 

    # --- Initialization of accumulators ---
    losses, confidence_diff, is_rejection = [], [], []
    clf_predictions, exp_predictions = [], []
    
    # =================================================================
    # 1. First Pass: Get model predictions and expert behaviors
    # =================================================================
    for data in data_loader:
        if len(data) == 2:
            images, labels, labels_sparse = data[0].to(device), data[1].to(device), None
        else:
            images, labels, labels_sparse = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # Select expert for the batch
        if len(experts_for_run) > 1:
            expert = random.choice(experts_for_run)
        else:
            expert = experts_for_run[0]

        # --- Finetuning Logic (if applicable) ---
        if is_finetune:
            # Prepare context for finetuning
            expert_cntx = cntx_sampler.sample(n_experts=1)
            cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
            exp_preds_cntx = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_sparse), device=device)
            expert_cntx.mc = exp_preds_cntx.unsqueeze(0)
            
            model.train()
            images_cntx = expert_cntx.xc.squeeze(0)
            targets_cntx = expert_cntx.yc.squeeze(0)
            costs_cntx = (exp_preds_cntx == targets_cntx).int()

            for _ in range(n_finetune_steps):
                outputs_cntx = model(images_cntx, expert_cntx).squeeze(0)
                loss = loss_fn(outputs_cntx, costs_cntx, targets_cntx, n_classes)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.params.clf.parameters():
                        param.copy_(param - lr_finetune * param.grad)
            
            if config["l2d"] == 'single':
                model.eval()

        # --- Prediction Logic ---
        with torch.no_grad():
            expert_cntx = None
            if np.random.binomial(1, p_cntx_inclusion) == 1:
                expert_cntx = cntx_sampler.sample(n_experts=1)
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
                exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_sparse), device=device)
                expert_cntx.mc = exp_preds.unsqueeze(0)

            outputs = model(images, expert_cntx if config["l2d"] == 'pop' else None)
            outputs = outputs.squeeze(0) if outputs.dim() > 2 else outputs

            probs = F.sigmoid(outputs) if config["loss_type"] == "ova" else outputs
            clf_probs, clf_preds = probs[:, :n_classes].max(dim=-1)
            exp_probs = probs[:, n_classes]

            confidence_diff.append(clf_probs - exp_probs)
            clf_predictions.append(clf_preds)
            is_rejection.append((outputs.max(dim=-1)[1] == n_classes).int())
            
            exp_pred = torch.tensor(expert(images, labels, labels_sparse), device=device)
            m = (exp_pred == labels).int()
            exp_predictions.append(exp_pred)

            losses.append(loss_fn(outputs, m, labels, n_classes).item())

        # Restore model for the next batch if finetuning was done
        if is_finetune:
            model = model_backup
            model.load_state_dict(copy.deepcopy(model_state_dict))
            if config["l2d"] == 'single': model.eval()
            else: model.train()
            
    # Restore model state completely after the evaluation run
    if is_finetune:
        model = model_backup # a bit redundant but safe
        model.load_state_dict(model_state_dict)
    model.eval()

    # =================================================================
    # 2. Second Pass: Calculate metrics based on sorted predictions
    # =================================================================
    confidence_diff = torch.cat(confidence_diff)
    indices_order = confidence_diff.argsort()

    # Reorder all collected data
    is_rejection = torch.cat(is_rejection)[indices_order]
    clf_predictions = torch.cat(clf_predictions)[indices_order]
    exp_predictions = torch.cat(exp_predictions)[indices_order]

    kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}
    data_loader_new = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data_loader.dataset, indices=indices_order),
        batch_size=data_loader.batch_size, shuffle=False, **kwargs)

    max_defer = math.floor(budget * len(data_loader.dataset))
    
    # --- Metric calculation ---
    correct_sys, exp, exp_total, correct, total, real_total = 0, 0, 0, 0, 0, 0
    clf_alone_correct, exp_alone_correct = 0, 0

    for data in data_loader_new:
        labels = data[1].to(device)
        batch_size = len(labels)

        for i in range(batch_size):
            r = is_rejection[real_total].item()
            # Enforce budget constraint
            if is_rejection[:real_total].sum().item() >= max_defer:
                r = 0
            
            prediction = clf_predictions[real_total].item()
            exp_prediction = exp_predictions[real_total].item()
            
            clf_alone_correct += (prediction == labels[i]).item()
            exp_alone_correct += (exp_prediction == labels[i].item())

            if r == 0:  # Classifier makes the decision
                total += 1
                correct += (prediction == labels[i]).item()
                correct_sys += (prediction == labels[i]).item()
            else:  # Deferred to expert
                exp_total += 1
                exp += (exp_prediction == labels[i]).item()
                correct_sys += (exp_prediction == labels[i]).item()
            
            real_total += 1

    # --- Finalize and return metrics ---
    cov = f"{total}/{real_total}"
    metrics = {
        "cov": cov,
        "sys_acc": 100.0 * correct_sys / real_total if real_total > 0 else 0,
        "exp_acc": 100.0 * exp / (exp_total + 1e-4),
        "clf_acc": 100.0 * correct / (total + 1e-4),
        "exp_acc_alone": 100.0 * exp_alone_correct / real_total if real_total > 0 else 0,
        "clf_acc_alone": 100.0 * clf_alone_correct / real_total if real_total > 0 else 0,
        "val_loss": np.mean(losses),
    }
    return metrics


def evaluate(model,
             experts_test,
             loss_fn,
             cntx_sampler,
             n_classes,
             data_loader,
             config,
             logger=None,
             budget=1.0,
             n_finetune_steps=0,
             lr_finetune=1e-1,
             p_cntx_inclusion=1.0,
             mean_across_experts=True,
             experts_train=None, # Not used in this function, kept for signature consistency
             train_cntx_sampler=None): # Not used in this function, kept for signature consistency
    """
    Evaluates the model by dispatching to a helper function.
    - If mean_across_experts=True, it evaluates against each expert individually and averages the results.
    - If mean_across_experts=False, it evaluates once, picking a random expert for each batch.
    """
    device = next(model.parameters()).device

    if not mean_across_experts:
        # Case 1: Evaluate once, picking a random expert from the test set for each batch.
        metrics = _perform_evaluation_run(
            model, experts_test, loss_fn, cntx_sampler, n_classes, data_loader,
            config, budget, n_finetune_steps, lr_finetune, p_cntx_inclusion, device
        )
        msg = "[Single Run] "
    
    else:
        # Case 2: Evaluate N times (once for each expert) and average the metrics.
        print("Mean across experts")
        list_of_metrics = []
        for expert in experts_test:
            metrics_e = _perform_evaluation_run(
                model, [expert], loss_fn, cntx_sampler, n_classes, data_loader,
                config, budget, n_finetune_steps, lr_finetune, p_cntx_inclusion, device
            )
            list_of_metrics.append(metrics_e)
        
        metrics = _average_metrics(list_of_metrics)
        msg = "[Average across experts] "

    # --- Logging ---
    for k, v in metrics.items():
        msg += f"{k} {v:.6f} " if isinstance(v, float) else f"{k} {v} "
    
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
        
    return metrics

            


def train_epoch(iters,
                train_loader,
                model,
                optimizer_lst,
                scheduler_lst,
                epoch,
                experts_train,
                loss_fn,
                cntx_sampler,
                n_classes,
                config,
                logger,
                n_steps_maml=5,
                lr_maml=1e-1):
    """ Train for one epoch """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, data in enumerate(train_loader):
        if len(data) == 2: 
            input, target = data
            input, target = input.to(device), target.to(device)
            index = None
        else:
            input, target, index = data # ignore additional labels
            input, target, index = input.to(device), target.to(device), index.to(device)
        n_experts = len(experts_train)

        # For MAML: need to do backprop once at start to initialize grads
        if (i==0) and (config["l2d"] == 'single_maml'):
            outputs = model(input)
            loss = loss_fn(outputs, torch.zeros_like(target, device=target.device), target, n_classes) # loss per expert
            loss.backward()
            model.zero_grad()
        
        if (config["l2d"] == 'pop') or (config["l2d"] == 'single_maml'):
            expert_cntx = cntx_sampler.sample(n_experts=n_experts)
            # sample expert predictions for context
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_train):
                cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_index), device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)

        if config["l2d"] == 'pop':
            outputs = model(input,expert_cntx) # [E,B,K+1]
        elif config["l2d"] == 'single':
            outputs = model(input) # [B,K+1]
            outputs = outputs.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K+1]

        if config["l2d"] == 'single_maml':
            for optimizer in optimizer_lst:
                optimizer.zero_grad()

            loss_cum = 0
            for idx_exp, expert in enumerate(experts_train): 
                local_model = copy.deepcopy(model)
                local_model.train()

                # freeze base network and classifier in train-time finetuning
                for param in local_model.params.base.parameters():
                    param.requires_grad = False
                for param in local_model.fc_clf.parameters():
                    param.requires_grad = False

                local_optim = torch.optim.SGD(local_model.parameters(), lr=lr_maml)
                local_optim.zero_grad()

                images_cntx = expert_cntx.xc[idx_exp]
                targets_cntx = expert_cntx.yc[idx_exp]
                exp_preds_cntx = expert_cntx.mc[idx_exp]
                costs = (exp_preds_cntx==targets_cntx).int()
                for _ in range(n_steps_maml):
                    outputs = local_model(images_cntx)
                    loss = loss_fn(outputs, costs, targets_cntx, n_classes)
                    loss.backward()
                    local_optim.step()
                    local_optim.zero_grad()

                # unfreeze base network and classifier for global update
                for param in local_model.params.base.parameters():
                    param.requires_grad = True
                for param in local_model.fc_clf.parameters():
                    param.requires_grad = True

                m = torch.tensor(expert(input, target, index), device=device)
                costs = (m==target).int()

                outputs = local_model(input)
                loss = loss_fn(outputs, costs, target, n_classes) / len(experts_train)
                loss.backward()

                for p_global, p_local in zip(model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

                loss_cum += loss
            
            epoch_train_loss.append(loss_cum.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:,:n_classes], target, topk=(1,))[0] # just measures clf accuracy
            losses.update(loss_cum.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            for optimizer, scheduler in zip(optimizer_lst,scheduler_lst):
                optimizer.step()
                scheduler.step()

        else: # l2d=single,pop
            loss = 0
            for idx_exp, expert in enumerate(experts_train):
                m = torch.tensor(expert(input, target, index), device=device)
                costs = (m==target).int()
                loss += loss_fn(outputs[idx_exp], costs, target, n_classes) # loss per expert
            loss /= len(experts_train)
            epoch_train_loss.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[0,:,:n_classes], target, topk=(1,))[0] # just measures clf accuracy
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            for optimizer in optimizer_lst:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, scheduler in zip(optimizer_lst,scheduler_lst):
                optimizer.step()
                scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters+=1

        # if i % 10 == 0:
        if i % 50 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    return iters, np.average(epoch_train_loss)


def train(model,
          train_dataset,
          validation_dataset,
          loss_fn,
          experts_train,
          experts_test,
          cntx_sampler_train, 
          cntx_sampler_eval,
          config):
    writer = SummaryWriter(log_dir=config["ckp_dir"])

    logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
    logger.info(f"p_out={config['p_out']}  seed={config['seed']}")
    logger.info(config)
    logger.info('No. of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    n_classes = config["n_classes"]
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["train_batch_size"], shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["val_batch_size"], shuffle=False, **kwargs)
    model = model.to(device)
    cudnn.benchmark = True

    epochs = config["epochs"]
    lr_wrn = config["lr_wrn"]
    lr_clf_rej = config["lr_other"]

    # assuming epochs >= 50
    if epochs > 100:
        milestone_epoch = epochs - 50    
    else:
        milestone_epoch = 50
    optimizer_base = torch.optim.SGD(model.params.base.parameters(), 
                        lr=lr_wrn,
                        momentum=0.9, 
                        nesterov=True,
                        weight_decay=config["weight_decay"])
    scheduler_base_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, len(train_loader)*milestone_epoch, eta_min=lr_wrn/1000)
    scheduler_base_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_base, factor=1., total_iters=0)
    scheduler_base_constant.base_lrs = [lr_wrn/1000 for _ in optimizer_base.param_groups]
    scheduler_base = torch.optim.lr_scheduler.SequentialLR(optimizer_base, [scheduler_base_cosine,scheduler_base_constant], 
                                                           milestones=[len(train_loader)*milestone_epoch])

    parameter_group = [{'params': model.params.clf.parameters()}]
    if config["l2d"] == "pop":
        parameter_group += [{'params': model.params.rej.parameters()}]
    optimizer_new = torch.optim.Adam(parameter_group, lr=lr_clf_rej)    
    scheduler_new_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new, len(train_loader)*milestone_epoch, eta_min=lr_clf_rej/1000)
    scheduler_new_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_new, factor=1., total_iters=0)
    scheduler_new_constant.base_lrs = [lr_clf_rej/1000 for _ in optimizer_new.param_groups]
    scheduler_new = torch.optim.lr_scheduler.SequentialLR(optimizer_new, [scheduler_new_cosine,scheduler_new_constant], 
                                                          milestones=[len(train_loader)*milestone_epoch])

    optimizer_lst = [optimizer_base, optimizer_new]
    scheduler_lst = [scheduler_base, scheduler_new]

    scoring_rule = config['scoring_rule']
    best_validation_score = np.inf
    # patience = 0
    iters = 0

    n_finetune_steps_eval = config['n_steps_maml'] if config['l2d']=='single_maml' else 0
    for epoch in range(0, epochs):
        print("Training")
        iters, train_loss = train_epoch(iters, 
                                        train_loader, 
                                        model, 
                                        optimizer_lst, 
                                        scheduler_lst, 
                                        epoch,
                                        experts_train,
                                        loss_fn,
                                        cntx_sampler_train,
                                        n_classes,
                                        config,
                                        logger,
                                        config['n_steps_maml'],
                                        config['lr_maml'])
        #Keeping track of training loss
        writer.add_scalar('train/loss', train_loss, epoch)
        metrics = evaluate(model,
                           experts_test,
                           loss_fn,
                           cntx_sampler_eval,
                           n_classes,
                           valid_loader,
                           config,
                           logger,
                           n_finetune_steps=n_finetune_steps_eval,
                           lr_finetune=config['lr_maml'],
                           mean_across_experts=False)

        validation_score = metrics[scoring_rule] if scoring_rule=='val_loss' else -metrics[scoring_rule]

        #Keeping track of validation loss and accuracy
        val_loss = metrics['val_loss']
        val_acc = metrics['sys_acc']
        writer.add_scalar('Loss/Validation', val_loss, epoch)  
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        if validation_score < best_validation_score:
            best_validation_score = validation_score
            torch.save(model.state_dict(), os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"))
        # Additionally save the whole config dict
        with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
            json.dump(config, f)


def eval(model, val_data, test_data, loss_fn, experts_test, val_cntx_sampler, test_cntx_sampler, config,mean_across_experts=True,experts_train=None):
    '''val_data and val_cntx_sampler are only used for single-expert finetuning'''
    model_state_dict = torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device)

    model.load_state_dict(model_state_dict)
    model = model.to(device)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["val_batch_size"], shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)

    scoring_rule = 'val_loss'
    for budget in config["budget"]:
        if config["l2d"] != 'single_maml':
            test_cntx_sampler.reset()
            logger = get_logger(os.path.join(config["ckp_dir"], "eval{}.log".format(budget)))
            model.load_state_dict(copy.deepcopy(model_state_dict))
            evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, \
                     budget,mean_across_experts=mean_across_experts)
        
        if (config["l2d"] == 'single_maml') or ((config["l2d"] == 'single') and config["finetune_single"]):
            logger = get_logger(os.path.join(config["ckp_dir"], "eval{}_finetune.log".format(budget)))
            
            n_finetune_steps_lst = [n_steps for n_steps in config["n_finetune_steps"] if n_steps >= config["n_steps_maml"]] \
                                    if (config["l2d"] == 'single_maml') else config["n_finetune_steps"]
            lr_finetune_lst = [config["lr_maml"]] if (config["l2d"] == 'single_maml') else config["lr_finetune"]

            steps_lr_comb = list(itertools.product(n_finetune_steps_lst, lr_finetune_lst))
            val_scores = []
            for (n_steps, lr) in steps_lr_comb:
                print(f'no. finetune steps: {n_steps}  step size: {lr}')
                val_cntx_sampler.reset()
                model.load_state_dict(copy.deepcopy(model_state_dict))
                #NR - Change to expert train because that's where the data is for the generated expert labels scenario
                if config["dataset"] == "generated_expert_labels_cifar" or config["dataset"] == "generated_expert_labels_fashion":
                    metrics = evaluate(model, experts_train, loss_fn, val_cntx_sampler, config["n_classes"], val_loader, config, None, budget, \
                                n_steps, lr,mean_across_experts=False)
                elif config["dataset"] == "generated_expert_labels_gtsrb":
                    metrics = evaluate(model, experts_test, loss_fn, val_cntx_sampler, config["n_classes"], val_loader, config, None, budget, \
                                n_steps, lr,mean_across_experts=False)

                else:
                    metrics = evaluate(model, experts_test, loss_fn, val_cntx_sampler, config["n_classes"], val_loader, config, None, budget, \
                                n_steps, lr,mean_across_experts=False)
                score = metrics[scoring_rule] if scoring_rule=='val_loss' else -metrics[scoring_rule]
                val_scores.append(score)
            idx = np.nanargmin(np.array(val_scores))
            best_finetune_steps, best_lr = steps_lr_comb[idx]
            test_cntx_sampler.reset()
            model.load_state_dict(copy.deepcopy(model_state_dict))
            metrics = evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, budget, \
                                best_finetune_steps, best_lr)

    # # Rebuttal experiment (onyl for l2d=pop)
    # for budget in config["budget"]:
    #     for p_cntx_inclusion in config["p_cntx_inclusion"]:
    #         test_cntx_sampler.reset()
    #         logger = get_logger(os.path.join(config["ckp_dir"], "eval{}_pc{}.log".format(budget,p_cntx_inclusion)))
    #         model.load_state_dict(copy.deepcopy(model_state_dict))
    #         evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, budget, p_cntx_inclusion=p_cntx_inclusion)

def build_experts(dataset,n_classes,p_out,n_experts,expert_labels):
    """
    Returns
    -------
    experts_train : list
    experts_test  : list
    """
    k = int(p_out)                      # number of oracle classes
    experts_train, experts_test = [], []

    # ------------------------------------------------------------
    # 1) datasets that use SyntheticExpertOverlap
    # ------------------------------------------------------------
    if dataset in {"gtsrb", "cifar10", "fashion"}:
        TOTAL = math.comb(n_classes, k)
        STEP  = 17

        # ---- build training experts ----
        for i in range(n_experts):
            r = (i * STEP) % TOTAL
            if k == 1:
                oracle = i % n_classes
            else:
                oracle = next(islice(combinations(range(n_classes), k), r, None))
            experts_train.append(
                SyntheticExpertOverlap(
                    classes_oracle=oracle,
                    n_classes=n_classes,
                    p_in=1.0,
                    p_out=0.0,
                )
            )

        # ---- build test experts ----
        half = n_experts // 2
        experts_test.extend(experts_train[:half])   # 50 % from train

        for i in range(half):                       # 50 % new experts
            r = ((i + 10) * STEP) % TOTAL
            if k == 1:
                oracle = (i + 15) % n_classes
            else:
                oracle = next(islice(combinations(range(n_classes), k), r, None))
            experts_test.append(
                SyntheticExpertOverlap(
                    classes_oracle=oracle,
                    n_classes=n_classes,
                    p_in=1.0,
                    p_out=0.0,
                )
            )

        if dataset == "cifar10":                    # keep old quirk
            experts_test.extend(experts_train)

    # ------------------------------------------------------------
    # 2) datasets that load pre-computed label arrays
    # ------------------------------------------------------------
    elif dataset.startswith("generated_expert_labels"):
        if expert_labels is None:
            raise ValueError("expert_labels must be provided for pre-computed datasets")

        root = f"data/{dataset}"
        tag  = f"e_{expert_labels}_p{p_out}"
        train_path = f"{root}/{tag}/train_array.npy"
        test_path  = f"{root}/{tag}/test_array.npy"

        train_labels = np.load(train_path)
        test_labels  = np.load(test_path)

        experts_train = [PreComputedExpert(train_labels[i]) for i in range(n_experts)]
        experts_test  = [PreComputedExpert(test_labels[i])  for i in range(n_experts)]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return experts_train, experts_test

def main(config):
    set_seed(config["seed"])
    config["ckp_dir"] = f"./runs/{config['dataset']}/{config['loss_type']}/l2d_{config['l2d']}/{config['train_type']}/e_{str(config['expert_labels'])}_p{str(config['p_out'])}_seed{str(config['seed'])}"
    # config["ckp_dir"] = f"./runs/{config['dataset']}/{config['loss_type']}/l2d_{config['l2d']}_lr{config['lr_maml']}_s{config['n_steps_maml']}/p{str(config['p_out'])}_seed{str(config['seed'])}" # tuning MAML
    os.makedirs(config["ckp_dir"], exist_ok=True)
   
    #Cifar10 - augmented labels
    if config["dataset"] == "generated_expert_labels_cifar":
        config["n_classes"] = 10
        train_data, val_data, test_data = load_cifar(variety='10', data_aug=False, seed=config["seed"],expert_type="generated_experts")
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
    #GTSRB - augmented labels 
    elif config["dataset"] == "generated_expert_labels_gtsrb":
        config["n_classes"] = 43
        train_data, val_data, test_data = load_gtsrb(expert_type="generated_experts")
        # resnet_base = resnet20(norm_type=config["norm_type"])
        # n_features = resnet_base.n_features
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
    #Fashion - augmented labels
    elif config ["dataset"] == "generated_expert_labels_fashion":
        config["n_classes"] = 10
        train_data, val_data, test_data = load_fashion_mnist(expert_type="generated_experts")
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
 
    #Cifar10 - sparse labels
    elif config["dataset"] == 'cifar10':
        config["n_classes"] = 10
        train_data, val_data, test_data = load_cifar(variety='10', data_aug=False, seed=config["seed"],expert_type="limited_demo")
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels
    
    #GTSRB - sparse labels
    elif config["dataset"] == 'gtsrb':
        config["n_classes"] = 43
        train_data, val_data, test_data = load_gtsrb(expert_type="limited_demo")    
        # resnet_base = resnet20(norm_type=config["norm_type"])
        # n_features = resnet_base.n_features
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels 

    #Fashionmnist - sparse labels
    elif config["dataset"] == "fashion":
        config["n_classes"] = 10
        train_data, val_data, test_data = load_fashion_mnist()
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type=config["norm_type"])
        n_features = resnet_base.nChannels

    else:
        raise ValueError('dataset unrecognised')

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = cross_entropy
        with_softmax = True
    else: # ova
        loss_fn = ova

    with_attn=False
    config_tokens = config["l2d"].split("_")
    if (len(config_tokens) > 1) and (config_tokens[0] == 'pop'):
        if config_tokens[1] == 'attn':
            with_attn = True
        config["l2d"] = "pop"

    if config["warmstart"]: 
        fn_aug = '' if config['norm_type']=='batchnorm' else '_frn'
        if config["dataset"] == "generated_expert_labels_cifar" or config['dataset'] == "cifar10":
            pre_trained_path = "cifar10"
        elif config["dataset"] == "generated_expert_labels_gtsrb" or config['dataset'] == "gtsrb":
            pre_trained_path = "gtsrb"
        elif config["dataset"] == "generated_expert_labels_fashion" or config['dataset'] == "fashion":
            pre_trained_path = "fashion"

        warmstart_path = f"pretrained/{pre_trained_path}/resnet/checkpoint.best"   
        checkpoint = torch.load(warmstart_path, map_location=device)
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        # resnet_base.load_state_dict(torch.load(warmstart_path, map_location=device))
       

    if config["l2d"] == "pop":
        
        #abalation studies 
        if config["train_type"] == 'lf':
            resnet_base.load_state_dict(checkpoint['model_state_dict'],strict=False)    
            resnet_base = resnet_base.to(device)
            model = ClassifierRejectorWithContextEmbedder(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, \
                                                    with_attn=with_attn, with_softmax=with_softmax, decouple=config["decouple"], \
                                                    depth_embed=config["depth_embed"], depth_rej=config["depth_reject"], train_type=config["train_type"])
            checkpoint_attn = torch.load(f"pretrained/{pre_trained_path}/attention/{int(config['p_out'])}/ckp.latest",map_location=device,weights_only=False)
            old_state = checkpoint_attn['model']
            new_state   = model.state_dict()
            load_state  = OrderedDict()

            for k, v in old_state.items():
                if k in new_state and v.shape == new_state[k].shape:
                    load_state[k] = v

            # # Optional — see what didn’t match
            # missing = [k for k in new_state.keys() if k not in load_state]
            # unexpected = [k for k in old_state.keys() if k not in new_state]
            # print("Will load   :", list(load_state.keys()), "...")
            # print("Missing     :", missing)
            # print("Unexpected  :", unexpected)

            new_state.update(load_state)
            model.load_state_dict(new_state,strict=False)

            print("Loading warmstart model")

        elif config["train_type"] == 'w':
            resnet_base.load_state_dict(checkpoint['model_state_dict'],strict=False)    
            resnet_base = resnet_base.to(device)
            model = ClassifierRejectorWithContextEmbedder(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, \
                                                    with_attn=with_attn, with_softmax=with_softmax, decouple=config["decouple"], \
                                                    depth_embed=config["depth_embed"], depth_rej=config["depth_reject"], train_type=config["train_type"])
            print("Loading warmstart model")


          

  
            


    #for single l2d
    else:
        resnet_base.load_state_dict(checkpoint['model_state_dict'],strict=False)
        resnet_base = resnet_base.to(device)
        model = ClassifierRejector(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, with_softmax=with_softmax, \
                                   decouple=config["decouple"])
    
    config["n_experts"] = 10 # assume exactly divisible by 2 
    experts_train = []
    experts_test = []

    experts_train, experts_test = build_experts(
        dataset=config["dataset"],
        n_classes=config["n_classes"],
        p_out=int(config["p_out"]),
        n_experts=config["n_experts"],
        expert_labels=config.get("expert_labels", None)
    )       

    # Context sampler train-time: just take from full train set (potentially with data augmentation)
    kwargs = {'num_workers': 0, 'pin_memory': True}

    cntx_sampler_train = ContextSampler(train_data.data, train_data.targets, train_data.transform, \
                                            n_cntx_pts=config["n_cntx_pts"], original_indices=train_data.original_indices, device=device, **kwargs)
    
        

    # Context sampler val/test-time: partition val/test sets
    prop_cntx = 0.2
    val_cntx_size = int(prop_cntx * len(val_data))


    val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data, [val_cntx_size, len(val_data)-val_cntx_size], \
                                                                 generator=torch.Generator().manual_seed(config["seed"]))
    print("Just val data cntx size:",len(val_data_cntx))
    print("Val size:", len(val_data_trgt))
    test_cntx_size = int(prop_cntx * len(test_data))

    test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_data, [test_cntx_size, len(test_data)-test_cntx_size], \
                                                                 generator=torch.Generator().manual_seed(config["seed"]))
    print("context_size:",test_cntx_size)
    print("test size:", len(test_data_trgt))
    #NR - changes to val_data_cntx and test_data_cntx

  
    cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.data[val_data_cntx.indices], 
                                    labels=val_data_cntx.dataset.targets[val_data_cntx.indices], 
                                    transform=val_data.transform, 
                                    n_cntx_pts=config["n_cntx_pts"], 
                                    device=device, 
                                    original_indices=val_data_cntx.dataset.original_indices[val_data_cntx.indices] if config["dataset"] == "generated_expert_labels_cifar" or config["dataset"] == "generated_expert_labels_gtsrb" or config["dataset"] == "generated_expert_labels_fashion" else None,
                                    **kwargs)
    cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.data[test_data_cntx.indices], 
                                      labels=np.array(test_data_cntx.dataset.targets)[test_data_cntx.indices], 
                                      transform=test_data.transform, 
                                      n_cntx_pts=config["n_cntx_pts"], 
                                      device=device, 
                                      original_indices=test_data_cntx.dataset.original_indices[test_data_cntx.indices] if config["dataset"] == "generated_expert_labels_cifar" or config["dataset"] == "generated_expert_labels_gtsrb" or config["dataset"]  == "generated_expert_labels_fashion" else None,
                                      **kwargs)
    
    if config["mode"] == 'train':

        if config["dataset"] == "generated_expert_labels_gtsrb":
            print("Training on generated expert labels for gtsrb")
            train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_test, cntx_sampler_train, cntx_sampler_val, config)
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True,experts_train=experts_train)

        elif config["dataset"] == "generated_expert_labels_cifar":
            train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_train, cntx_sampler_train, cntx_sampler_val, config)
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True,experts_train=experts_train)
        elif config["dataset"] == "generated_expert_labels_fashion":
            train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_train, cntx_sampler_train, cntx_sampler_val, config)
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True,experts_train=experts_train)

        else:
            train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_test, cntx_sampler_train, cntx_sampler_val, config)
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True)

        # eval(model, val_data_trgt, val_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_val, config) # tuning MAML
    else: # evaluation on test data
        if config["dataset"] == "generated_expert_labels_cifar" or config["dataset"] == "generated_expert_labels_gtsrb" or config["dataset"] == "generated_expert_labels_fashion":
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True,experts_train=experts_train)
        else:
            eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config,mean_across_experts=True)

        # eval(model, val_data_trgt, val_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_val, config) # tuning MAML


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1071)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_wrn", type=float, default=1e-1, help="learning rate for wrn.")
    parser.add_argument("--lr_other", type=float, default=1e-2, help="learning rate for non-wrn model components.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="default",
                            help="specify the experiment name. Checkpoints will be saved with this name.")
    
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument("--p_out", type=float, default=0.1) # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    parser.add_argument('--l2d', choices=['single', 'single_maml', 'pop', 'pop_attn'], default='single')
    parser.add_argument('--loss_type', choices=['softmax', 'ova'], default='softmax')
    parser.add_argument("--n_cntx_pts", type=int, default=50)
    parser.add_argument('--scoring_rule', choices=['val_loss', 'sys_acc'], default='val_loss')
    parser.add_argument('--norm_type', choices=['batchnorm', 'frn'], default='batchnorm')

    parser.add_argument("--dataset", choices=["cifar10",  "gtsrb","generated_expert_labels_cifar","generated_expert_labels_gtsrb","generated_expert_labels_fashion","fashion"], default="cifar10") 
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--warmstart', action='store_true')
    # parser.set_defaults(warmstart=True)
    parser.add_argument("--depth_embed", type=int, default=6)
    parser.add_argument("--depth_reject", type=int, default=4)
    
    #NR - expert stuff

    ## MAML
    parser.add_argument('--n_steps_maml', type=int, default=5)
    parser.add_argument('--lr_maml', type=float, default=1e-1)
    parser.add_argument('--decouple', action='store_true')
    parser.set_defaults(decouple=False)

    ## EVAL
    parser.add_argument('--budget', nargs='+', type=float, default=[1.0]) #[0.01,0.02,0.05,0.1,0.2,0.5,1.0]
    # parser.add_argument('--p_cntx_inclusion', nargs='+', type=float, default=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # rebuttal experiment
    parser.add_argument('--finetune_single', action='store_true')
    parser.set_defaults(finetune_single=True)
    parser.add_argument('--n_finetune_steps', nargs='+', type=int, default=[1,2,5,10,20])
    parser.add_argument('--lr_finetune', nargs='+', type=float, default=[1e-1,1e-2])

    #For logging
    parser.add_argument('--expert_labels',type=int,required=True)

    #Important when loading pretrained models for either generated labels or limited demonstrations
    parser.add_argument('--train_type',type=str,required=True)
    
    

    config = parser.parse_args().__dict__
    main(config)
