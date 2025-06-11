from __future__ import print_function
import random
import time
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from LinearModel import LinearNN, DeepNN , ClassifierRejectorWithContextEmbedder, ClassifierRejector
import datasets.cifar as cifar
import datasets.nih as nih
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import fbeta_score
from torch.amp import GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from focal_loss.focal_loss import FocalLoss
from itertools import combinations, islice
from math import comb
import copy 
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler
from utils import load_from_checkpoint
from Expert import CIFAR100Expert, NIHExpert, CIFAR10Expert , SyntheticExpertOverlap, GTSRBExpert , HAM1000Expert , FashionExpert
from feature_extractor.embedding_model import EmbeddingModel
from itertools import product
import pandas as pd

from predictions.predict import *
from evaluate.evaluate import *


def plot_confusion_matrix(cm, class_names, title='Confusion matrix'):
    """Plot confusion matrix using seaborn and matplotlib"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt.gcf()

def set_model(args,class_weights=None):
    """Initialize models

    :param args: training arguments
    :return: tupl
        - model: Initialized model
        - criteria_x: Supervised loss function
        - ema_model: Initialized ema model
    """
    if args.dataset.lower() == 'cifar100' or args.dataset.lower() == 'cifar10':
        feature_dim = 128
        depth_embed = 6 
        actual_classes = 100 if args.dataset.lower() == 'cifar100' else 10  
    elif args.dataset.lower() == 'nih':
        feature_dim = 512
    elif args.dataset == 'gtsrb':
        feature_dim = 128
        depth_embed = 5
        actual_classes = 43
    elif args.dataset == 'ham10000':
        feature_dim = 128
    elif args.dataset == 'fashion':
        depth_embed = 6 
        actual_classes = 10
        feature_dim = 128
    else:
        print(f'Dataset {args.dataset} not defined')
        sys.exit()
    
    with_attn = args.with_attn #can be attn or mlp or single 
    if with_attn == 'mlp' or with_attn == 'attn':
        model = ClassifierRejectorWithContextEmbedder(n_features=feature_dim,num_classes=2,with_softmax=False,
                                                  depth_embed=depth_embed,actual_classes=actual_classes,dim_hid=128,
                                                  with_attn=with_attn)
    else:
        model = ClassifierRejector(num_classes=2,n_features=feature_dim,with_softmax=False)
    
    

            
    model.train()
    model.cuda()

    if args.focal:
        print("Using Focal Loss")
        if class_weights is not None:
            print("Weighted")
            if args.match:
                print("X is weighted cross entropy loss")
                criteria_x = nn.CrossEntropyLoss(weight=class_weights).cuda()
            else:
                print("X is weighted FL")
                criteria_x = FocalLoss(gamma=args.gamma, weights=class_weights, reduction='mean').cuda()
            criteria_u = FocalLoss(gamma=args.gamma, reduction='none').cuda()
        else:
            print("Non-Weighted")
            if args.match:
                print("X is cross-entropy")
                criteria_x = nn.CrossEntropyLoss().cuda()
            else:
                print("X is focal loss")
                criteria_x = FocalLoss(gamma=args.gamma, reduction='mean').cuda()
            # criteria_x = nn.CrossEntropyLoss().cuda()
            criteria_u = FocalLoss(gamma=args.gamma, reduction='none').cuda()
    else:
        print("Using CE Loss")
        if class_weights is not None:
            print("Weighted")
            criteria_x = nn.CrossEntropyLoss(weight=class_weights).cuda()
            criteria_u = nn.CrossEntropyLoss(reduction='none',weight=class_weights).cuda()
        else:
            print("Non-Weighted")
            criteria_x = nn.CrossEntropyLoss().cuda()
            criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    if args.eval_ema:

        if with_attn == 'mlp' or with_attn == 'attn':   
            # Use the model with context
            ema_model = ClassifierRejectorWithContextEmbedder(num_classes=2,n_features=feature_dim,with_softmax=False,
                                                  depth_embed=depth_embed,actual_classes=actual_classes,dim_hid=128,
                                                  with_attn=with_attn)
        else:
            # Use the model without context
            ema_model = ClassifierRejector(num_classes=2,n_features=feature_dim,with_softmax=False)
        
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval()
    else:
        ema_model = None    
              
    return model, criteria_x, criteria_u, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """Momentum update of evaluation model (exponential moving average)

    :param model: Model
    :param ema_model: EMA-Model
    :param ema_m: Ema parameter
    :return:
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)    
        
        
def train_one_epoch(
    epoch,
    model,
    ema_model,
    emb_model,
    criteria_x,
    criteria_u,
    optim,
    lr_schdlr,
    dltrain_x,
    dltrain_u,
    args,
    n_iters,
    logger,
    prob_list,
    scaler,
    experts_train,
    cntx,
    experts_train_bin,  
):
    """
    Train one epoch on the train set, but further break down the forward pass
    into Embedding vs. Main Model forward for more detailed timing.
    """
    n_experts = len(experts_train)  
    model.train()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    n_correct_u_lbs_meter = AverageMeter()
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    # Timing Meters
    data_time_meter = AverageMeter()
    embed_time_meter = AverageMeter()    # For emb_model.get_embedding()
    model_time_meter = AverageMeter()    # For model(embedding)
    loss_time_meter = AverageMeter()
    backward_time_meter = AverageMeter()
    step_time_meter = AverageMeter()
    iteration_time_meter = AverageMeter()

    if args.lam_u > 0:
        dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    else:
        dl_x, dl_u = iter(dltrain_x), None

    for it in range(n_iters):
        iter_start = time.time()

        # ---------------------------
        # 1) Data Loading
        # ---------------------------
        data_start = time.time()
        ims_x_weak, lbs_x, im_id = next(dl_x)

        if dl_u is not None:
            (ims_u_weak, ims_u_strong), lbs_u_real, im_id = next(dl_u)

        data_end = time.time()
        data_time_meter.update(data_end - data_start)

        lbs_x = lbs_x.cuda()

        if dl_u is not None:
            lbs_u_real = lbs_u_real.cuda()

        if dl_u is not None:
            imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).cuda()
        else:
            imgs = ims_x_weak.cuda()
        bt = ims_x_weak.size(0)

        if dl_u is not None:
            mu = int(ims_u_weak.size(0) // bt)



        # ---------------------------
        # 2) Forward pass: break it down
        # ---------------------------

        with torch.autocast(device_type="cuda", dtype=torch.float16):

            # 2a) Embedding extraction
            embed_start = time.time()
            # Force data load to complete on GPU
            torch.cuda.synchronize()
            # print("Batch size: ", imgs.shape)
            embedding = emb_model.get_embedding(batch=imgs)
            torch.cuda.synchronize()
            embed_end = time.time()
            embed_time_meter.update(embed_end - embed_start)

            ##NR - Context sampler with expert labelling
            expert_cntx = cntx.sample(n_experts = n_experts)
            exp_preds_cntx = [] 
            for idx_exp, expert in enumerate(experts_train):
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_sparse)).cuda()
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx) # [E,NC,1]
        
            #NR Getting cntx.em features
            E , Nc = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            expert_cntx.em    = em_flat.view(E, Nc, -1) # [E,Nc,1280]

            # 2b) Main model forward
            model_start = time.time()

            #inference with context and without context 

            if args.with_attn == "attn" or args.with_attn == "mlp":
                # Use the model with context
                logits = model(embedding, expert_cntx)  
            elif args.with_attn == "single":
                logits = model(embedding)  # Use the model without context
            # logits = model(embedding,expert_cntx)

            torch.cuda.synchronize()
            model_end = time.time()
            model_time_meter.update(model_end - model_start)

            loss = 0

            lbs_x_orig = lbs_x.clone()

            if dl_u is not None:
                lbs_u_real_orig = lbs_u_real.clone()

            for idx_exp , expert in enumerate(experts_train_bin):
                logits_x = logits[idx_exp][:bt]

                if dl_u is not None:
                    logits_u_w, logits_u_s = torch.split(logits[idx_exp][bt:], bt * mu)
                    lbs_u_real = torch.tensor(expert(labels=lbs_u_real_orig), dtype=torch.long).cuda()

                lbs_x = torch.tensor(expert(labels=lbs_x_orig), dtype=torch.long).cuda() 
                
                # ---------------------------
                # 3 a) Supervised Loss Computation
                # ---------------------------
                loss_start = time.time()
                # Convert logits to probabilities for Focal Loss
                if args.focal:
                    if args.match:
                        loss_x = criteria_x(logits_x,lbs_x)
                    else:
                        m = torch.nn.Softmax(dim=1)
                        loss_x = criteria_x(m(logits_x), lbs_x)
                    # loss_x = criteria_x(logits_x,lbs_x)
                else:
                    loss_x = criteria_x(logits_x, lbs_x)  # CrossEntropyLoss


                # ---------------------------
                # 3 b) Unsupervised Loss Computation
                # ---------------------------
                if dl_u is not None:
                    with torch.no_grad():
                        probs = torch.softmax(logits_u_w, dim=1)
                        if args.DA:
                            prob_list.append(probs.mean(0))
                            if len(prob_list) > 32:
                                prob_list.pop(0)
                            prob_avg = torch.stack(prob_list, dim=0).mean(0)
                            probs = probs / prob_avg
                            probs = probs / probs.sum(dim=1, keepdim=True)

                        scores, lbs_u_guess = torch.max(probs, dim=1)
                        mask = scores.ge(args.thr).float()

                    # For unlabeled loss (logits_u_s)
                    if args.focal:
                        m = torch.nn.Softmax(dim=1)
                        loss_u = (criteria_u(m(logits_u_s), lbs_u_guess) * mask).mean()
                    else:
                        loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()

                curr_loss = loss_x + args.lam_u * loss_u if dl_u is not None else loss_x
                loss += curr_loss
                torch.cuda.synchronize()
                loss_end = time.time()
                loss_time_meter.update(loss_end - loss_start)
            loss /= n_experts

        # ---------------------------
        # 4) Backward Pass
        # ---------------------------
        backward_start = time.time()
        optim.zero_grad()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        backward_end = time.time()
        backward_time_meter.update(backward_end - backward_start)

        # ---------------------------
        # 5) Optimizer Step + LR + EMA
        # ---------------------------
        step_start = time.time()
        scaler.step(optim)
        scaler.update()

        lr_schdlr.step()
        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        torch.cuda.synchronize()
        step_end = time.time()
        step_time_meter.update(step_end - step_start)

        # ---------------------------
        # Bookkeeping
        # ---------------------------
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item()) if dl_u is not None else 0
        mask_meter.update(mask.mean().item()) if dl_u is not None else 0

        if dl_u is not None:
            corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        else:
            corr_u_lb = torch.zeros_like(lbs_x).float()
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item()) if dl_u is not None else 0
        n_strong_aug_meter.update(mask.sum().item()) if dl_u is not None else 0.02

        iter_end = time.time()
        iteration_time_meter.update(iter_end - iter_start)

        # Optionally log every 64 iterations (tweak to your preference)
        if (it + 1) % 64 == 0:
            logger.info(
                f"{args.dataset}-x{args.n_labeled}-s{args.seed}, {args.exp_dir} | "
                f"Epoch:{epoch}, Iter: {it + 1}. "
                f"loss_u: {loss_u_meter.avg:.3f}, loss_x: {loss_x_meter.avg:.3f}. "
                f"n_correct_u: {n_correct_u_lbs_meter.avg:.2f}/{n_strong_aug_meter.avg:.2f}, "
                f"Mask:{mask_meter.avg:.3f}, LR: {optim.param_groups[0]['lr']:.6f}, "
                f"Time(64 iters): {iteration_time_meter.avg * 64:.2f}s"
            )
            logger.info(
                f"    Avg DataTime: {data_time_meter.avg:.4f}s | "
                f"Embed: {embed_time_meter.avg:.4f}s | "
                f"ModelFwd: {model_time_meter.avg:.4f}s | "
                f"Loss: {loss_time_meter.avg:.4f}s | "
                f"Backward: {backward_time_meter.avg:.4f}s | "
                f"Step: {step_time_meter.avg:.4f}s | "
                f"Total (per iter): {iteration_time_meter.avg:.4f}s"
            )

            # If you only want the last 64-iteration average each time:
            data_time_meter.reset()
            embed_time_meter.reset()
            model_time_meter.reset()
            loss_time_meter.reset()
            backward_time_meter.reset()
            step_time_meter.reset()
            iteration_time_meter.reset()

    # Return the relevant metrics
    return (
        loss_x_meter.avg,
        loss_u_meter.avg,
        mask_meter.avg,
        n_correct_u_lbs_meter.avg / (n_strong_aug_meter.avg + 1e-8),
        prob_list,
    )


#NR - Function to calculate class weights
def get_class_weights(dltrain_x, n_classes):
    all_labels = []
    for batch in dltrain_x:
        img, lbs, _ = batch
        all_labels.extend(lbs.numpy())
    all_labels = np.array(all_labels)

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(n_classes), 
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    return class_weights

def main():
    parser = argparse.ArgumentParser(description='FixMatch Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')    
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=2,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=400,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=25,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)    

    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--DA', default=False, help='use distribution alignment')

    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')   
    
    parser.add_argument('--exp-dir', default='EmbeddingFM_bin', type=str, help='experiment directory')
    parser.add_argument('--ex_strength', default=60, help='Strength of the expert')
    parser.add_argument('--ex_seed', default=0, help='Seed for the expert')
    
    #NR-For p_out 
    parser.add_argument('--p-out',default=0.4,type=float,help='overlap probability')
    
    #NR - For weighted loss 
    parser.add_argument('--gamma', type=float, default=2, help='Gamma for Focal Loss')
    parser.add_argument('--focal', default=False, action='store_true', help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--weighted',default=False,type=bool,help='use weighted loss?')
    parser.add_argument('--match',default=False,action='store_true',help="use focal match or not")
    
    #NR - Deeper net
    parser.add_argument('--deeper',default=False,type=bool,help='Deeper net maybe')
    parser.add_argument('--with-attn',type=str,default="attn")

    args = parser.parse_args()

    #NR - model architecture 
    
    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))
    
    tb_logger = SummaryWriter(output_dir)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 200

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    
   
    if 'cifar100' in args.dataset.lower():
        #NR - Generate n datasets with different seeds, generate on the fly or pre-determined - big question 
        expert = CIFAR100Expert(20, int(args.ex_strength), 1, 0, 123)
        dltrain_x, dltrain_u = cifar.get_train_loader(
            args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='fixmatch',weighted=False)
        dlval = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)


    elif  'cifar10' in args.dataset.lower():
        print("CIFAR10")
        from itertools import combinations, islice
        from math import comb
        k = int(args.p_out)
        n = 10
        TOTAL = comb(n,k)
        STEP = 17
        experts_train = []
        experts_test = [] 
        experts_train_bin = []
        experts_test_bin = []
        config = {
            "n_experts": 10,
            "p_out" : int(args.p_out),
            "n_classes": 10
        }

        for i in range(config["n_experts"]): # train
            r = (i * STEP) % TOTAL
            if args.p_out == 1:
                class_oracle = i % config['n_classes']
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_train_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_train.append(expert) 
        
        
        experts_test += experts_train[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)
        experts_test_bin += experts_train_bin[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)

        
        for i in range(config["n_experts"]//2): # then sample 50% new experts
            r = (i + 10 * STEP) % TOTAL
            if config["p_out"] == 1:
                class_oracle = (i + 15 % config["n_classes"])
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_test_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_test.append(expert)
  

        dltrain_x, dltrain_u , train_cntx_sampler , dl_x_eval , dl_u_eval = cifar.get_train_loader(
            args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='fixmatch',weighted=False)
        dlval , val_cntx_sampler = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)
        
        print("dltrain_x",len(dltrain_x.dataset))
        print("dltrain_u",len(dltrain_u.dataset))
        print("dlval",len(dlval.dataset))

    elif 'fashion' in args.dataset.lower():
        
        from itertools import combinations, islice
        from math import comb
        k = int(args.p_out)
        n = 10
        TOTAL = comb(n,k)
        STEP = 17
        experts_train = []
        experts_test = [] 
        experts_train_bin = []
        experts_test_bin = []
        config = {
            "n_experts": 10,
            "p_out" : int(args.p_out),
            "n_classes": 10
        }

        for i in range(config["n_experts"]): # train
            r = (i * STEP) % TOTAL
            if args.p_out == 1:
                class_oracle = i % config['n_classes']
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_train_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_train.append(expert) 
        
        
        experts_test += experts_train[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)
        experts_test_bin += experts_train_bin[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)

        for i in range(config["n_experts"]//2): # then sample 50% new experts
            r = (i + 10 * STEP) % TOTAL
            if config["p_out"] == 1:
                class_oracle = (i + 15 % config["n_classes"])
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_test_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_test.append(expert)

        dltrain_x, dltrain_u , train_cntx_sampler , dl_x_eval , dl_u_eval = cifar.get_train_loader(
            args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='fixmatch',weighted=False)
        dlval , val_cntx_sampler = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)
        

    elif 'gtsrb' in args.dataset.lower():
        from itertools import combinations, islice
        from math import comb
        k = int(args.p_out)
        n = 43
        TOTAL = comb(n,k)
        STEP = 17
        experts_train = []
        experts_test = [] 
        experts_train_bin = []
        experts_test_bin = []
        config = {
            "n_experts": 10,
            "p_out" : int(args.p_out),
            "n_classes": 43
        }

        for i in range(config["n_experts"]): # train
            r = (i * STEP) % TOTAL
            if args.p_out == 1:
                class_oracle = i % config['n_classes']
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_train_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_train.append(expert) 
        
        
        experts_test += experts_train[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)
        experts_test_bin += experts_train_bin[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)

        for i in range(config["n_experts"]//2): # then sample 50% new experts
            r = (i + 10 * STEP) % TOTAL
            if config["p_out"] == 1:
                class_oracle = (i + 15 % config["n_classes"])
            else:
                class_oracle = next(islice(combinations(range(n), k), r, None))
            print("Classes oracle:",class_oracle)
            expert_bin = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0)
            experts_test_bin.append(expert_bin)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=0,binary=False)
            experts_test.append(expert)

        dltrain_x, dltrain_u , train_cntx_sampler , dl_x_eval , dl_u_eval = cifar.get_train_loader(
            args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='fixmatch',weighted=False)
        dlval , val_cntx_sampler = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)


    elif 'nih' in args.dataset.lower():
        expert = NIHExpert(int(args.ex_strength), 2)
        dltrain_x, dltrain_u = nih.get_train_loader(
            expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, method='fixmatch')
        dlval = nih.get_val_loader(expert, batch_size=64, num_workers=2)
        
        print("dltrain_x",len(dltrain_x))
        print("dltrain_u",len(dltrain_u))
        print("dlval",len(dlval))

    elif 'ham10000' in args.dataset.lower():
      
        k = int(args.p_out)  # number of oracle classes
        n = 7               
        TOTAL = comb(n, k)   # total combinations
        STEP = 17
        r = (args.seed * STEP) % TOTAL
        classes_oracle = next(islice(combinations(range(n), k), r, None))

        logger.info(f"  Classes oracle = {classes_oracle}")
        expert = HAM1000Expert(classes_oracle, n_classes=7, p_in=1.0, p_out=0.0)
        dltrain_x, dltrain_u = cifar.get_train_loader(args.dataset, expert, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root,
            method='fixmatch',weighted=False)
        dlval = cifar.get_val_loader(args.dataset, expert, batch_size=64, num_workers=2)
        print("dltrain_x",len(dltrain_x.dataset))
        print("dltrain_u",len(dltrain_u.dataset))
        print("dlval",len(dlval.dataset))


    print("args weighted is:",args.weighted)
    class_weights = get_class_weights(dltrain_x, args.n_classes) if args.weighted else None
    print("class weights is:",type(class_weights))
    model, criteria_x, criteria_u, ema_model = set_model(args,class_weights=class_weights)
    emb_model = EmbeddingModel(os.getcwd(), args.dataset)
    print(model)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))


    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    
    # optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
    #                         momentum=args.momentum, nesterov=False)

    # lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    base_lr   = 1e-3
    head_lr   = 1e-2
    wd        = 1e-5

    param_groups = [
        { 
        'params': model.fc.parameters(), 
        'lr': head_lr, 
        'weight_decay': wd 
        },
        {
        'params': [p for n,p in model.named_parameters() if not n.startswith('fc.')],
        'lr': base_lr,
        'weight_decay': wd
        }
    ]

    optim = torch.optim.Adam(param_groups)

    # ——— 2) scheduler ————————————————————————————————————————
    # a cosine annealing from base_lr → base_lr/1000 over N steps:
    epochs = 50
    T_max = len(dltrain_x) * epochs
    lr_schdlr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=T_max,
        eta_min=base_lr/1000
    )


    model, ema_model, optim, lr_schdlr, start_epoch, metrics, prob_list = \
        load_from_checkpoint(output_dir, model, ema_model, optim, lr_schdlr, mode='fixmatch')

    # lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    scaler = GradScaler()

    train_args = dict(
        model=model,
        ema_model=ema_model,
        emb_model=emb_model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger,
        prob_list=prob_list,
        scaler=scaler,
        experts_train = experts_train,
        cntx = train_cntx_sampler,
        experts_train_bin = experts_train_bin
    )

    best_acc = -1
    best_epoch = 0
    best_f05 = -1.0  # [EARLY STOP CHANGE] track best F0.5
    best_acc = -1.0


    if metrics is not None:
        best_acc = metrics['best_acc']
        best_epoch = metrics['best_epoch']


    logger.info('-----------start training--------------')
    class_names = [str(i) for i in range(args.n_classes)]

    steps_grid = [0]                
    lr_grid    = [0]

    for epoch in range(start_epoch, args.n_epoches):
        loss_x, loss_u, mask_mean, guess_label_acc, prob_list = train_one_epoch(epoch, **train_args)

        # # top1, ema_top1 = evaluate(model, ema_model, emb_model, dlval, criteria_x)
        # # f_score = evaluate_f0_5_sklearn(model,ema_model,emb_model,dlval)

        tb_logger.add_scalar('loss_x', loss_x, epoch)
        tb_logger.add_scalar('loss_u', loss_u, epoch)
        tb_logger.add_scalar('guess_label_acc', guess_label_acc, epoch)
        tb_logger.add_scalar('mask', mask_mean, epoch)

        if epoch % 5 == 0:
            records = []
            for steps, lr_finetune in product(steps_grid, lr_grid):
                top1, ema_top1, f05_model, f05_ema, cm_model = evaluate_merged(model,ema_model,emb_model,dlval,criteria_x,beta=0.5,
                                                                               experts_test=experts_test,cntx=val_cntx_sampler,
                                                                               experts_test_bin=experts_test_bin,steps=steps,lr_finetune=lr_finetune,
                                                                               with_attn=args.with_attn)
                records.append({"steps": steps, "lr": lr_finetune, "top1": top1,
                    "ema_top1": ema_top1, "f05_model": f05_model, "f05_ema": f05_ema}
                )

            results_df = pd.DataFrame(records)
            # print(results_df)
            # ---- find best combination ------------------------------------------------
            best_idx  = results_df["top1"].idxmax()       # index of highest accuracy
            best_row  = results_df.loc[best_idx]

            print(f"Best top-1 accuracy: {best_row.top1:.4f} "
                f"achieved with steps = {best_row.steps} and lr = {best_row.lr:.6f}")
            
            tb_logger.add_scalar('test_acc', top1, epoch)
            tb_logger.add_scalar('test_ema_acc', ema_top1, epoch)
            tb_logger.add_scalar('f_score',f05_model,epoch)
            fig = plot_confusion_matrix(cm_model, class_names, 
                                   f'Model Confusion Matrix (Epoch {epoch})')
            tb_logger.add_figure('Confusion_Matrix/Model', fig, epoch)
            
            # if best_acc < top1:
            #     best_acc = top1
            #     best_epoch = epoch

      
            
            if best_row.top1 >= best_acc + 0.02:
                # It's an improvement of >= 2%, so update best_acc
                best_acc = best_row.top1
                best_epoch = epoch
                save_obj = {
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'lr_scheduler': lr_schdlr.state_dict(),
                    'prob_list': prob_list,
                    'metrics': {'best_acc': best_acc, 'best_epoch': best_epoch},
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(output_dir, 'ckp.latest'))
            else:
                print("Current acc:",best_row.top1)
                logger.info(
                    f"Accuracy did not improve by +0.02 from previous best ({best_acc:.3f}). "
                    "Stopping training early."
                )
                break  # Stop the entire training loop here.

            logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                        format(epoch, best_row.top1, ema_top1, best_acc, best_epoch))
            logger.info("Epoch {}. F0.5: {:.4f}. Ema-F0.5: {:.4f}".format(epoch, f05_model, f05_ema))
                

       
    
    print("Break program")
    sys.exit(0)
    if 'cifar' in args.dataset.lower():



        for idx_exp, expert in enumerate(experts_train):
            print("Expert:",idx_exp)
            predictions, accs = predict_cifar_acc(model, None, emb_model, dl_x_eval, dl_u_eval, dlval,expert,
                                                  experts_train_bin[idx_exp],train_cntx_sampler,val_cntx_sampler,idx_exp)   
            # logger.info(f"Train_x accuracy: {accs['train_x']:.4f}")
            logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
            logger.info(f"Validation accuracy: {accs['val']:.4f}")
            if not os.path.exists('./artificial_expert_labels/'):
                os.makedirs('./artificial_expert_labels/')
            pred_file = f'expert_{idx_exp + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
            with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                json.dump(predictions, f)
        # generate predictions for last 5 test experts
        for idx_exp, expert_test in enumerate(experts_test[-5:]):

            predictions, accs = predict_cifar_acc(model,None, emb_model, dl_x_eval, dl_u_eval, dlval,expert_test,
                                                  experts_test_bin[idx_exp+5],train_cntx_sampler,val_cntx_sampler,idx_exp+15)   
            # logger.info(f"Train_x accuracy: {accs['train_x']:.4f}")
            logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
            logger.info(f"Validation accuracy: {accs['val']:.4f}")
            if not os.path.exists('./artificial_expert_labels/'):
                os.makedirs('./artificial_expert_labels/')
            pred_file = f'expert_{idx_exp + 10 + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
            with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                json.dump(predictions, f)
        

    elif 'nih' in args.dataset.lower():
        predictions = predict_nih(model, ema_model, emb_model, dltrain_x, dltrain_u, dlval)

    elif 'gtsrb' in args.dataset.lower():
        for idx_exp, expert in enumerate(experts_train):
                print("Expert:",idx_exp)
                predictions, accs = predict_gtsrb_acc(model, None, emb_model, dl_x_eval, dl_u_eval, dlval,expert,
                                                    experts_train_bin[idx_exp],train_cntx_sampler,val_cntx_sampler,idx_exp)   
        
                if not os.path.exists('./artificial_expert_labels/'):
                    os.makedirs('./artificial_expert_labels/')
                pred_file = f'expert_{idx_exp + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
                logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
                logger.info(f"Validation accuracy: {accs['val']:.4f}")
                with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                    json.dump(predictions, f)
            # generate predictions for last 5 test experts
        for idx_exp, expert_test in enumerate(experts_test[-5:]):

            predictions, accs = predict_gtsrb_acc(model,None, emb_model, dl_x_eval, dl_u_eval, dlval,expert_test,
                                                    experts_test_bin[idx_exp+5],train_cntx_sampler,val_cntx_sampler,idx_exp+15)   

            if not os.path.exists('./artificial_expert_labels/'):
                os.makedirs('./artificial_expert_labels/')
            pred_file = f'expert_{idx_exp + 10 + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
            logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
            logger.info(f"Validation accuracy: {accs['val']:.4f}")
            with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                json.dump(predictions, f)
            

    elif 'ham10000' in args.dataset.lower():
        predictions , accs = predict_ham10000_acc(model, ema_model, emb_model, dltrain_x, dltrain_u, dlval)
        logger.info(f"Train_x accuracy: {accs['train_x']:.4f}")
        logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
        logger.info(f"Validation accuracy: {accs['val']:.4f}")

    elif 'fashion' in args.dataset.lower():
        for idx_exp, expert in enumerate(experts_train):
            print("Expert:",idx_exp)
            print("type of model",type(model))
            print("type of emb_model",type(emb_model))
            print("type of dl_x_eval",type(dl_x_eval))
            print("type of dl_u_eval",type(dl_u_eval))
            print("type of dlval",type(dlval))
            print("type of expert",type(expert))
            print("type of experts_train_bin",type(experts_train_bin[idx_exp]))
            print("type of train_cntx_sampler",type(train_cntx_sampler))
            print("type of val_cntx_sampler",type(val_cntx_sampler))
            print("type of idx_exp",type(idx_exp))
            
            predictions, accs = predict_fashion_acc(model, None, emb_model, dl_x_eval, dl_u_eval, dlval,expert,
                                                  experts_train_bin[idx_exp],train_cntx_sampler,val_cntx_sampler)   
     
            if not os.path.exists('./artificial_expert_labels/'):
                os.makedirs('./artificial_expert_labels/')
            pred_file = f'expert_{idx_exp + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
            with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                json.dump(predictions, f)
            logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
            logger.info(f"Validation accuracy: {accs['val']:.4f}")
        # generate predictions for last 5 test experts
        for idx_exp, expert_test in enumerate(experts_test[-5:]):

            predictions, accs = predict_fashion_acc(model,None, emb_model, dl_x_eval, dl_u_eval, dlval,expert_test,
                                                  experts_test_bin[idx_exp+5],train_cntx_sampler,val_cntx_sampler)   

            if not os.path.exists('./artificial_expert_labels/'):
                os.makedirs('./artificial_expert_labels/')
            pred_file = f'expert_{idx_exp + 10 + 1}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
            with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
                json.dump(predictions, f)
            logger.info(f"Train_u accuracy: {accs['train_u']:.4f}")
            logger.info(f"Validation accuracy: {accs['val']:.4f}")
        

    logger.info("***** Generate Predictions *****")
    # if not os.path.exists('./artificial_expert_labels/'):
    #     os.makedirs('./artificial_expert_labels/')
    # pred_file = f'{args.exp_dir}_{args.dataset.lower()}_expert{args.ex_strength}.{args.seed}@{args.n_labeled}_predictions.json'
    # with open(f'artificial_expert_labels/{pred_file}', 'w') as f:
    #     json.dump(predictions, f)
    # with open(os.getcwd()[:-len('Embedding-Semi-Supervised')]+f'Learning-to-Defer-Algs/artificial_expert_labels/{pred_file}', 'w') as f:
    #     json.dump(predictions, f)


if __name__ == '__main__':
    main()
