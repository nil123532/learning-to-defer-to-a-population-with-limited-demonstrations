import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from lib.utils import AverageMeter, accuracy
from sklearn.metrics import fbeta_score, confusion_matrix
import pandas as pd
from itertools import product
import random
import copy 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns



# def find_best_lr_and_steps(model,emb_model,loader,expert,experts_bin,cntx_sampler):

#     steps_grid = [1,2,5,10,20]
#     lr_grid = [1e-1,1e-2,1e-3,1e-4,1e-5]
#     records = []    
    
#     criterion = torch.nn.CrossEntropyLoss()
#     orig_state = copy.deepcopy(model.state_dict())
        

#     for finetune_steps, lr_finetune in product(steps_grid, lr_grid):

#         model.eval()

#         top1_meter = AverageMeter()
#         model_preds = []
#         true_labels = []


        
#         for ims, lbs, im_id in loader:
#             ims, lbs_orig = ims.cuda(), lbs.cuda()
#             lbs = torch.tensor(experts_bin(labels=lbs_orig), dtype=torch.long).cuda()
#             embedding = emb_model.get_embedding(batch=ims)
#             expert_cntx = cntx_sampler.sample(n_experts = 1)
#             cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
#             exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
#             expert_cntx.mc = exp_preds.unsqueeze(0)
#             E , NC = expert_cntx.xc.shape[:2]
#             C , H , W = expert_cntx.xc.shape[-3:]
#             xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
#             xc_flat = xc_flat.cuda()
#             em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
#             em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
#             expert_cntx.em = em

#             model.train()
#             for _ in range(finetune_steps):
#                 output = model(em.squeeze(0),expert_cntx).squeeze(0)
#                 targets = torch.tensor(experts_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
#                 loss = criterion(output, targets)
#                 model.zero_grad()
#                 loss.backward()
#                 with torch.no_grad():
#                     for param in model.parameters():
#                             # if param.requires_grad:
#                             new_param = param - lr_finetune * param.grad
#                             param.copy_(new_param)
#             model.eval()

#             with torch.no_grad():
#                 logits = model(embedding, expert_cntx).squeeze(0)
#                 scores = torch.softmax(logits, dim=1)
#                 top1 = accuracy(scores,lbs, (1,))
#                 top1_meter.update(top1.item())  
#                 model_preds.extend(torch.argmax(scores, dim=1).cpu().tolist())

#                 true_labels.extend(lbs.cpu().tolist())

#             model.load_state_dict(orig_state, strict=True)

#         model.eval()

#         model_f05 = fbeta_score(true_labels,model_preds, beta=0.5)
#         cm_model = confusion_matrix(true_labels,model_preds)

#         acc = top1_meter.avg
#         records.append({ 'steps': finetune_steps,
#                         'lr': lr_finetune,
#                         'acc': acc,
#                         'f05': model_f05,
#                         'cm_model': cm_model
#                     })
#     df = pd.DataFrame(records)
#     best_idx = df['acc'].idxmax()
#     best_row = df.loc[best_idx]

#     return best_row['steps'], best_row['lr']


def freeze_except_fc(model):
    """
    Freeze all layers except the final fully connected layer.
    """
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

def find_best_lr_and_steps(model,emb_model,loader,expert,experts_bin,cntx_sampler):


    steps_grid = [0,1,2,5,10,20]
    lr_grid = [1e-1,1e-2,1e-3]
    losses = []
    # head_factor = 1
    records = []    
    
    criterion = torch.nn.CrossEntropyLoss()
    orig_state = copy.deepcopy(model.state_dict())
        

    for finetune_steps, lr_finetune in product(steps_grid, lr_grid):

        model.eval()

        top1_meter = AverageMeter()
        model_preds = []
        true_labels = []


        
        for ims, lbs, im_id in loader:
            ims, lbs_orig = ims.cuda(), lbs.cuda()
            lbs = torch.tensor(experts_bin(labels=lbs_orig), dtype=torch.long).cuda()
            embedding = emb_model.get_embedding(batch=ims)
            expert_cntx = cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em

            ### Whole model finetuning
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune)
            # model.train()
            # for _ in range(finetune_steps):
            #     optimizer.zero_grad()
            #     output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(
            #         experts_bin(None, expert_cntx.yc.squeeze(0), None)
            #     ).cuda()
            #     loss = criterion(output, targets)
            #     loss.backward()
            #     optimizer.step()
            # model.eval()

            ### head finetuning
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune)    
            model.train()
            for _ in range(finetune_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    experts_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            model.eval()


            with torch.no_grad():
                logits = model(embedding, expert_cntx).squeeze(0)
                scores = torch.softmax(logits, dim=1)
                top1 = accuracy(scores,lbs, (1,))
                top1_meter.update(top1.item())  
                model_preds.extend(torch.argmax(scores, dim=1).cpu().tolist())

                true_labels.extend(lbs.cpu().tolist())

            model.load_state_dict(orig_state, strict=True)

        model.eval()

        model_f05 = fbeta_score(true_labels,model_preds, beta=0.5)
        cm_model = confusion_matrix(true_labels,model_preds)

        # print(f"Finetune Steps:{finetune_steps}, LR: {lr_finetune}, loss : {np.mean(losses)}")

        acc = top1_meter.avg
        records.append({ 'steps': finetune_steps,
                        'lr': lr_finetune,
                        'acc': acc,
                        'f05': model_f05,
                        'cm_model': cm_model
                    })
    df = pd.DataFrame(records)
    best_idx = df['acc'].idxmax()
    best_row = df.loc[best_idx]

    return best_row['steps'], best_row['lr']

def predict_cifar_acc(model, ema_model, emb_model, trainloader_x, trainloader_u, testloader,
                      expert,expert_bin,train_cntx_sampler,test_cntx_sampler,id,args):
    """
    Generate predictions for CIFAR (storing each sample exactly once),
    and compute "unique" accuracy for each loader (train_x, train_u, val).

    Returns:
      predictions_dict: {'train': [...], 'test': [...]}
        - For CIFAR-10/100, 'train' is length 50k, 'test' is length 10k
        - Each index i in 'train' or 'test' is the final predicted label for image i
      accuracies: dict of float
        - keys = 'train_x', 'train_u', 'val'
    """
    unique_idx_train = set()
    unique_idx_test = set()
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Arrays to hold final predictions for all train/test indices:
    predictions_dict = {
        'train': np.zeros(50000, dtype=int),  # CIFAR-10 train size
        'test': np.zeros(10000, dtype=int)    # CIFAR-10 test size
    }
    accuracies = {}

    # ---------------------------
    # 1) TRAIN LABELED (dltrain_x)
    # ---------------------------
    if trainloader_x is not None:
        final_labels_x = {}
        final_preds_x  = {}
        idxes = []
        for ims, lbs, im_id in trainloader_x:
            ims = ims.cuda()
            lbs = lbs.cuda()

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            expert_labels = expert(labels=lbs)

            for i, idx in enumerate(im_id_cpu):
                final_labels_x[idx] = lbs_cpu[i]
                final_preds_x[idx]  = expert_labels[i]
                predictions_dict['train'][idx] = expert_labels[i]
                idxes.append(idx)
                unique_idx_train.add(idx)

        # Compute unique-sample accuracy for labeled train
        correct = sum(final_preds_x[k] == final_labels_x[k] for k in final_labels_x)
        total = len(final_labels_x)
     
        accuracies['train_x'] = correct / total if total > 0 else 0.0

        



    # ---------------------------
    # 2) TRAIN UNLABELED (dltrain_u)
    # ---------------------------
    best_steps, best_lr = 0, 0
    if args.finetune:
        best_steps, best_lr = find_best_lr_and_steps(model,emb_model,trainloader_u,expert,expert_bin,train_cntx_sampler)
        print(f"Best steps: {best_steps}, Best lr: {best_lr}")

    orig_state = copy.deepcopy(model.state_dict())
    if trainloader_u is not None:
        final_labels_u = {}
        final_preds_u  = {}
        for ims_weak, lbs, im_id in trainloader_u:
            ims_weak = ims_weak.cuda()
            lbs = lbs.cuda()

            #context stuff
            expert_cntx = train_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em
            model.train()
            
            ### USING MANUAL UPDATE ###
            #finetuning
            for _ in range(best_steps):
                output = model(em.squeeze(0),expert_cntx).squeeze(0)
                targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
                loss = torch.nn.CrossEntropyLoss()(output, targets)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        new_param = param - best_lr * param.grad
                        param.copy_(new_param)

            ### USING OPTIMIZER ###
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            model.eval()



            # Forward pass - Evaluation model
            model.eval()
            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims_weak)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))  

            for i, idx in enumerate(im_id_cpu):
                final_labels_u[idx] = lbl_processed[i]
                final_preds_u[idx]  = preds_cpu[i]

                if preds_cpu[i] > 0:
                    pred = lbs_cpu[i]
                else:
                    pred = random.randint(0,9)
      
                final_preds_x[idx]  = pred
                predictions_dict['train'][idx] = pred
                unique_idx_train.add(idx)
            
            model.load_state_dict(orig_state, strict=True)  # Restore original model state 



        # Compute unique-sample accuracy for unlabeled train
        correct = sum(final_preds_u[k] == final_labels_u[k] for k in final_labels_u)
        total = len(final_labels_u)

        accuracies['train_u'] = correct / total if total > 0 else 0.0

    # ---------------------------
    # 3) VALIDATION/TEST
    # ---------------------------

    predictions = []

    if testloader is not None:
        count = 0
        final_labels_val = {}
        final_preds_val  = {}
   
        for ims, lbs, im_id in testloader:
            ims = ims.cuda()
            lbs = lbs.cuda()

            #context stuff
           
            expert_cntx = test_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em


            ### USING MANUAL UPDATE ###
            #finetune stuff
            # model.train()
            # for _ in range(best_steps):
            #     output = model(em.squeeze(0),expert_cntx).squeeze(0)
            #     targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            #     loss = torch.nn.CrossEntropyLoss()(output, targets)
            #     model.zero_grad()
            #     loss.backward()
            #     with torch.no_grad():
            #         for param in model.parameters():
            #             new_param = param - best_lr * param.grad
            #             param.copy_(new_param)

            ####### USING OPTIMIZER #######
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            model.eval()
            

            # Forward pass - Evaluation
            model.eval()
            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_true_expert_labels = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))

            for i, idx in enumerate(im_id_cpu):
                final_labels_val[idx] = lbl_processed[i]
                final_preds_val[idx]  = preds_cpu[i]

                predictions.append(preds_cpu[i])  

                if preds_cpu[i] > 0:
                    pred = lbs_cpu[i]
                else:
                    pred = random.randint(0,9)

                # Put it into predictions['test']
                count += 1 if lbl_true_expert_labels[i] == lbs_cpu[i] else 0
                predictions_dict['test'][idx] = pred
                # predictions_dict['test'][idx] = lbl_expert[i]
                unique_idx_test.add(idx)




            model.load_state_dict(orig_state, strict=True)  # Restore original model state


        # Compute unique-sample accuracy for validation/test
        correct = sum(final_preds_val[k] == final_labels_val[k] for k in final_labels_val)
        total = len(final_labels_val)
        accuracies['val'] = correct / total if total > 0 else 0.0


        # -----------------------------
        # Labels from the val cntx set
        # -----------------------------

        for imgs, lbls, idxs in test_cntx_sampler.iterate_all():
                # move to CPU so we can turn them into plain ints
                idxs_np = idxs.cpu().numpy()
                # lbls_np = lbls.cpu().numpy()
                
                expert_preds = expert(labels=lbls) 
                expert_bin_preds = expert_bin(labels=lbls)

                preds_np  = np.asarray(expert_preds)
                lbls  = lbls.cpu().numpy()

        
                for i, idx in enumerate(idxs_np):

                    final_labels_val[idx] = expert_bin_preds[i]
                    final_preds_val[idx]  = expert_bin_preds[i]
                    # Also place it into predictions['train']
                    predictions_dict['test'][idx] = expert_preds[i]
                    unique_idx_test.add(idx)
                    predictions.append(expert_bin_preds[i])


    return {'train':predictions_dict['train'].tolist(), 'test':predictions_dict['test'].tolist()}, accuracies


def predict_gtsrb_acc(model, ema_model, emb_model, trainloader_x, trainloader_u, testloader,
                      expert,expert_bin,train_cntx_sampler,test_cntx_sampler,id,args):
    """
    Generate predictions for GTSRB (storing each sample exactly once),
    and compute "unique" accuracy for each loader (train_x, train_u, val).

    Returns:
      predictions_dict: {'train': [...], 'test': [...]}  
        - 'train' is length 26640, 'test' is length 12630
      accuracies: dict of float
        - keys = 'train_x', 'train_u', 'val'
    """
    unique_idx_train = set()
    unique_idx_test = set()
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Preallocate space for each split
    predictions_dict = {
        'train': np.zeros(26640, dtype=int),
        'test':  np.zeros(12630, dtype=int)
    }
    accuracies = {}

    # ---------------------------
    # 1) TRAIN LABELED (dltrain_x)
    # ---------------------------
    if trainloader_x is not None:
        final_labels_x = {}
        final_preds_x  = {}
        idxes = []
        for ims, lbs, im_id in trainloader_x:
            ims = ims.cuda()
            lbs = lbs.cuda()

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            expert_labels = expert(labels=lbs)

            for i, idx in enumerate(im_id_cpu):
                final_labels_x[idx] = lbs_cpu[i]
                final_preds_x[idx]  = expert_labels[i]
                # Also place it into predictions['train']
                predictions_dict['train'][idx] = expert_labels[i]
                idxes.append(idx)
                unique_idx_train.add(idx)

        # Compute unique-sample accuracy for labeled train
        correct = sum(final_preds_x[k] == final_labels_x[k] for k in final_labels_x)
        total = len(final_labels_x)
 
        accuracies['train_x'] = correct / total if total > 0 else 0.0
        

    #best lr and steps 


    # ---------------------------
    # 2) TRAIN UNLABELED (dltrain_u)
    # ---------------------------
    best_steps, best_lr = 0, 0
    if args.finetune:
        best_steps, best_lr = find_best_lr_and_steps(model,emb_model,trainloader_u,expert,expert_bin,train_cntx_sampler)
        print(f"Best steps: {best_steps}, Best lr: {best_lr}")


    orig_state = copy.deepcopy(model.state_dict())

    if trainloader_u is not None:
        final_labels_u = {}
        final_preds_u  = {}
        for ims_weak, lbs, im_id in trainloader_u:
            ims_weak = ims_weak.cuda()
            lbs = lbs.cuda()

            #finetune stuff
            
            expert_cntx = train_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em


            ### USING MANUAL UPDATE ###
            # model.train()
            # for _ in range(best_steps):
            #     output = model(em.squeeze(0),expert_cntx).squeeze(0) if ema_model is None else ema_model(em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            #     loss = torch.nn.CrossEntropyLoss()(output, targets)
            #     model.zero_grad()
            #     loss.backward()
            #     with torch.no_grad():
            #         for param in model.parameters():
            #             # if param.requires_grad:
            #                 # Update only trainable parameters
            #             new_param = param - best_lr * param.grad
            #             param.copy_(new_param)
        
            # # Forward pass
            # model.eval()

            

            ## USING OPTIMIZER ###
            criterion = torch.nn.CrossEntropyLoss()
            ## Whole model finetuning
            # optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            # model.train()
            # for _ in range(best_steps):
            #     optimizer.zero_grad()
            #     output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(
            #         expert_bin(None, expert_cntx.yc.squeeze(0), None)
            #     ).cuda()
            #     loss = criterion(output, targets)
            #     loss.backward()
            #     optimizer.step()
            # model.eval()

             ### head finetuning
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)    
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            model.eval()

            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims_weak)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))  

            for i, idx in enumerate(im_id_cpu):
                final_labels_u[idx] = lbl_processed[i]
                final_preds_u[idx]  = preds_cpu[i]

                if preds_cpu[i] > 0:
                    pred = lbs_cpu[i]
                else:
                    while True:
                        pred = random.randint(0,42)
                        if pred != lbs_cpu[i]:
                            break

      
                final_preds_x[idx]  = pred
                # Also put it into predictions['train']
                predictions_dict['train'][idx] = pred
                # predictions_dict['train'][idx] = lbl_expert[i] 
                unique_idx_train.add(idx)
            
            model.load_state_dict(orig_state, strict=True)  # Restore original model state  



        # Compute unique-sample accuracy for unlabeled train
        correct = sum(final_preds_u[k] == final_labels_u[k] for k in final_labels_u)
        total = len(final_labels_u)

        accuracies['train_u'] = correct / total if total > 0 else 0.0

    # ---------------------------
    # 3) VALIDATION/TEST
    # ---------------------------
    predictions = []


        
    if testloader is not None:
        count = 0
        final_labels_val = {}
        final_preds_val  = {}
   
        for ims, lbs, im_id in testloader:
            ims = ims.cuda()
            lbs = lbs.cuda()

            #finetune stuff
           
            expert_cntx = test_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em

            ### USING MANUAL UPDATE ###
            # model.train()
            # for _ in range(best_steps):
            #     output = model(em.squeeze(0),expert_cntx).squeeze(0) if ema_model is None else ema_model(em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            #     loss = torch.nn.CrossEntropyLoss()(output, targets)
            #     model.zero_grad()
            #     loss.backward()
        
            #     with torch.no_grad():
            #         for param in model.parameters():
            #             # if param.requires_grad:
            #             new_param = param - best_lr * param.grad
            #             param.copy_(new_param)
            # # Forward pass
            # model.eval()

            ####### USING OPTIMIZER #######
            # optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            # model.train()
            # for _ in range(best_steps):
            #     optimizer.zero_grad()
            #     output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(
            #         expert_bin(None, expert_cntx.yc.squeeze(0), None)
            #     ).cuda()
            #     loss = criterion(output, targets)
            #     loss.backward()
            #     optimizer.step()
            # model.eval()

            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)    
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            model.eval()

            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_true_expert_labels = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))

            for i, idx in enumerate(im_id_cpu):
                final_labels_val[idx] = lbl_processed[i]
                final_preds_val[idx]  = preds_cpu[i]

                predictions.append(preds_cpu[i])  

                if preds_cpu[i] > 0:
                    pred = lbs_cpu[i]
                else:
                    while True:
                        pred = random.randint(0,42)
                        if pred != lbs_cpu[i]:
                            break

                count += 1 if lbl_true_expert_labels[i] == lbs_cpu[i] else 0
                predictions_dict['test'][idx] = pred
                unique_idx_test.add(idx)




            model.load_state_dict(orig_state, strict=True)  # Restore original model state  


        # Compute unique-sample accuracy for validation/test
        correct = sum(final_preds_val[k] == final_labels_val[k] for k in final_labels_val)
        total = len(final_labels_val)
        accuracies['val'] = correct / total if total > 0 else 0.0


        # -----------------------------
        # Labels from the val cntx set
        # -----------------------------
        for imgs, lbls, idxs in test_cntx_sampler.iterate_all():
                # move to CPU so we can turn them into plain ints
                idxs_np = idxs.cpu().numpy()
                
                expert_preds = expert(labels=lbls) 
                expert_bin_preds = expert_bin(labels=lbls)

                preds_np  = np.asarray(expert_preds)
                lbls  = lbls.cpu().numpy()

                for i, idx in enumerate(idxs_np):
                    final_labels_val[idx] = expert_bin_preds[i]
                    final_preds_val[idx]  = expert_bin_preds[i]
                    predictions_dict['test'][idx] = expert_preds[i]
                    unique_idx_test.add(idx)
                    predictions.append(expert_bin_preds[i])
    return {'train':predictions_dict['train'].tolist(), 'test':predictions_dict['test'].tolist()}, accuracies








def predict_fashion_acc(model, ema_model, emb_model, trainloader_x, trainloader_u, testloader,
                      expert,expert_bin,train_cntx_sampler,test_cntx_sampler,args):
    """
    Generate predictions for GTSRB (storing each sample exactly once),
    and compute "unique" accuracy for each loader (train_x, train_u, val).

    Returns:
      predictions_dict: {'train': [...], 'test': [...]}  
        - 'train' is length 26640, 'test' is length 12630
      accuracies: dict of float
        - keys = 'train_x', 'train_u', 'val'
    """
    unique_idx_train = set()    
    unique_idx_test = set()
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Preallocate space for each split
    predictions_dict = {
        'train': np.zeros(60000, dtype=int),
        'test':  np.zeros(10000, dtype=int)
    }
    accuracies = {}
    # ---------------------------
    # 1) TRAIN LABELED (dltrain_x)
    # ---------------------------
    if trainloader_x is not None:
        final_labels_x = {}
        final_preds_x  = {}
        idxes = []
        for ims, lbs, im_id in trainloader_x:
            ims = ims.cuda()
            lbs = lbs.cuda()

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            expert_labels = expert(labels=lbs)

            for i, idx in enumerate(im_id_cpu):
                final_labels_x[idx] = lbs_cpu[i]
                final_preds_x[idx]  = expert_labels[i]
                predictions_dict['train'][idx] = expert_labels[i]
                idxes.append(idx)
                unique_idx_train.add(idx)

        # Compute unique-sample accuracy for labeled train
        correct = sum(final_preds_x[k] == final_labels_x[k] for k in final_labels_x)
        total = len(final_labels_x)

        
        accuracies['train_x'] = correct / total if total > 0 else 0.0

        #Plotting confusion matrix  
        

    #best lr and steps 


    # ---------------------------
    # 2) TRAIN UNLABELED (dltrain_u)
    # ---------------------------
    best_steps, best_lr = 0, 0
    if args.finetune:
        best_steps, best_lr = find_best_lr_and_steps(model,emb_model,trainloader_u,expert,expert_bin,train_cntx_sampler)


    orig_state = copy.deepcopy(model.state_dict())

    
    if trainloader_u is not None:
        final_labels_u = {}
        final_preds_u  = {}
        for ims_weak, lbs, im_id in trainloader_u:
            ims_weak = ims_weak.cuda()
            lbs = lbs.cuda()

            #finetune stuff
            
            expert_cntx = train_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em
            model.train()

            ### USING MANUAL UPDATE ###
            # for _ in range(best_steps):
            #     output = model(em.squeeze(0),expert_cntx).squeeze(0) if ema_model is None else ema_model(em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            #     loss = torch.nn.CrossEntropyLoss()(output, targets)
            #     model.zero_grad()
            #     loss.backward()
            #     with torch.no_grad():
            #         for param in model.parameters():
            #             new_param = param - best_lr * param.grad
            #             param.copy_(new_param)

            ### USING OPTIMIZER ###
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            criterion = torch.nn.CrossEntropyLoss() 
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

            # Forward pass
            model.eval()
            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims_weak)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            # Overwrite final predictions & store ground-truth
            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))  

            for i, idx in enumerate(im_id_cpu):
                final_labels_u[idx] = lbl_processed[i]
                final_preds_u[idx]  = preds_cpu[i]

                pred = lbs_cpu[i] if preds_cpu[i] > 0 else random.randint(0,9) #give the right label if the model is confident, else random between 0 and 9 inclusive
                final_preds_x[idx]  = pred
                # Also put it into predictions['train']
                predictions_dict['train'][idx] = pred
                # predictions_dict['train'][idx] = lbl_expert[i] 
                unique_idx_train.add(idx)
            
            model.load_state_dict(orig_state, strict=True)  # Restore original model state  

        #compute train_u accuracy
        correct = sum(final_preds_u[k] == final_labels_u[k] for k in final_labels_u)
        total = len(final_labels_u)
        accuracies['train_u'] = correct / total if total > 0 else 0.0


    # ---------------------------
    # 3) VALIDATION/TEST
    # ---------------------------

    if testloader is not None:
        count = 0
        final_labels_val = {}
        final_preds_val  = {}
   
        for ims, lbs, im_id in testloader:
            ims = ims.cuda()
            lbs = lbs.cuda()

            #finetune stuff
            expert_cntx = test_cntx_sampler.sample(n_experts = 1)
            cntx_yc_index = None if expert_cntx.yc_index is None else expert_cntx.yc_index.squeeze(0)
            exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_index)).cuda()
            expert_cntx.mc = exp_preds.unsqueeze(0)
            E , NC = expert_cntx.xc.shape[:2]
            C , H , W = expert_cntx.xc.shape[-3:]
            xc_flat = expert_cntx.xc.flatten(0, 1)   # [E*Nc,3,32,32]
            xc_flat = xc_flat.cuda()
            em_flat = emb_model.get_embedding(xc_flat) # [E*Nc,1280]
            em    = em_flat.view(E, NC, -1) # [E,Nc,1280]
            expert_cntx.em = em

            ### USING MANUAL UPDATE ###
            # model.train()
            # for _ in range(best_steps):
            #     output = model(em.squeeze(0),expert_cntx).squeeze(0) if ema_model is None else ema_model(em.squeeze(0), expert_cntx).squeeze(0)
            #     targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            #     loss = torch.nn.CrossEntropyLoss()(output, targets)
            #     model.zero_grad()
            #     loss.backward()
            #     with torch.no_grad():
            #         for param in model.parameters():
            #             new_param = param - best_lr * param.grad
            #             param.copy_(new_param)
            
            ####### USING OPTIMIZER #######
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            criterion = torch.nn.CrossEntropyLoss() 
            model.train()
            for _ in range(best_steps):
                optimizer.zero_grad()
                output  = model(expert_cntx.em.squeeze(0), expert_cntx).squeeze(0)
                targets = torch.tensor(
                    expert_bin(None, expert_cntx.yc.squeeze(0), None)
                ).cuda()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            model.eval()

            # Forward pass
            model.eval()
            with torch.no_grad():
                embeddings = emb_model.get_embedding(ims)
                logits = model(embeddings, expert_cntx).squeeze(0) if ema_model is None else ema_model(embeddings, expert_cntx).squeeze(0)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            im_id_cpu = im_id.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            lbs_cpu   = lbs.cpu().numpy()
            lbl_processed = expert_bin(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_true_expert_labels = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))
            lbl_expert = expert(labels=torch.tensor(lbs_cpu, dtype=torch.long, device='cuda'))

            for i, idx in enumerate(im_id_cpu):
                final_labels_val[idx] = lbl_processed[i]
                final_preds_val[idx]  = preds_cpu[i]
                pred = lbs_cpu[i] if preds_cpu[i] > 0 else random.randint(0,9) 


                # Put it into predictions['test']
                count += 1 if lbl_true_expert_labels[i] == lbs_cpu[i] else 0
                predictions_dict['test'][idx] = pred
                unique_idx_test.add(idx)




            model.load_state_dict(orig_state, strict=True)  # Restore original model state  

        # Compute unique-sample accuracy for validation/test
        correct = sum(final_preds_val[k] == final_labels_val[k] for k in final_labels_val)
        total = len(final_labels_val)
        accuracies['val'] = correct / total if total > 0 else 0.0
        

        # -----------------------------
        # Labels from the val cntx set
        # -----------------------------

        for imgs, lbls, idxs in test_cntx_sampler.iterate_all():
                # move to CPU so we can turn them into plain ints
                idxs_np = idxs.cpu().numpy()
                
                expert_preds = expert(labels=lbls) 
                preds_np  = np.asarray(expert_preds)
                lbls  = lbls.cpu().numpy()

                for i, idx in enumerate(idxs_np):
                    final_labels_val[idx] = lbls[i]
                    final_preds_val[idx]  = expert_preds[i]
                    predictions_dict['test'][idx] = expert_preds[i]
                    unique_idx_test.add(idx)



    return {'train':predictions_dict['train'].tolist(), 'test':predictions_dict['test'].tolist()}, accuracies
