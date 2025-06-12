
import random
import copy
import torch
from sklearn.metrics import fbeta_score, confusion_matrix
from lib.utils import AverageMeter, accuracy
from sklearn.metrics import fbeta_score, confusion_matrix

def evaluate_merged(model, ema_model, emb_model, dataloader, criterion, beta=0.5,experts_test=None,cntx=None,experts_test_bin=None,steps=20,lr_finetune=0.01,with_attn='mlp'):
    """Evaluate model and ema_model on validation set, computing accuracy and F0.5 score.

    Args:
        model: The main model to evaluate.
        ema_model: The EMA model to evaluate (optional).
        emb_model: Embedding model used to get input features.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function (not used here, kept for compatibility).
        beta: Beta parameter for F-beta score calculation.

    Returns:
        tuple: Contains:
            - model_top1 (float): Main model's average top1 accuracy.
            - ema_top1 (float): EMA model's average top1 accuracy (None if not present).
            - model_f05 (float): Main model's F0.5 score.
            - ema_f05 (float): EMA model's F0.5 score (None if not present).
    """
    choice = 8  # Randomly choose an expert for evaluation
    expert = experts_test[choice]
    expert_bin = experts_test_bin[choice]


    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Initialize meters and prediction lists
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter() if ema_model is not None else None
    model_preds = []
    ema_preds = [] if ema_model is not None else None
    true_labels = []

    #finetune stuff 
    initial_state = copy.deepcopy(model.state_dict())  # Store initial model state   
    finetune_steps = steps
    lr_finetune = lr_finetune

    orig_state = {k: v.detach().clone()      # true copy, no shared storage
              for k, v in model.state_dict().items()}

    for ims, lbs, im_id in dataloader:
        ims, lbs_orig = ims.cuda(), lbs.cuda()
        lbs = torch.tensor(expert_bin(labels=lbs_orig), dtype=torch.long).cuda()  # Convert labels to expert labels
        embedding = emb_model.get_embedding(batch=ims)

        expert_cntx = cntx.sample(n_experts = 1)
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
        for _ in range(finetune_steps):
            output = model(em.squeeze(0),expert_cntx).squeeze(0)
            targets = torch.tensor(expert_bin(None, expert_cntx.yc.squeeze(0), None)).cuda()
            loss = criterion(output, targets)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    new_param = param - lr_finetune * param.grad
                    param.copy_(new_param)
        model.eval()

        with torch.no_grad():
            # Main model processing
            logits = model(embedding,expert_cntx).squeeze(0)
            scores = torch.softmax(logits, dim=1)
            top1 = accuracy(scores, lbs, (1,))
            top1_meter.update(top1.item())
            model_preds.extend(torch.argmax(scores, dim=1).cpu().tolist())

            #Recover model
            # model = model_backup
            model.load_state_dict(initial_state, strict=True)
            # EMA model processing
            if ema_model is not None:
                ema_logits = ema_model(embedding,expert_cntx).squeeze(0)
                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_top1 = accuracy(ema_scores, lbs, (1,))
                ema_top1_meter.update(ema_top1.item())
                ema_preds.extend(torch.argmax(ema_scores, dim=1).cpu().tolist())

            true_labels.extend(lbs.cpu().tolist())

    model.load_state_dict(orig_state,strict=True)

    model.eval()

    # Calculate F0.5 scores
    model_f05 = fbeta_score(true_labels, model_preds, beta=beta)
    ema_f05 = None
    ema_top1_avg = None
    
    if ema_model is not None:
        ema_f05 = fbeta_score(true_labels, ema_preds, beta=beta)
        ema_top1_avg = ema_top1_meter.avg

    cm_model = confusion_matrix(true_labels, model_preds)

    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.state_dict().values(), initial_state.values())), "Model not restored correctly!"

    return (
        top1_meter.avg,
        ema_top1_avg,
        model_f05,
        ema_f05,
        cm_model
    )
