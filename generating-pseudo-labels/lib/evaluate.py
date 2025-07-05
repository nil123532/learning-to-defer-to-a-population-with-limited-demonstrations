
import random
import copy
import torch
from sklearn.metrics import fbeta_score, confusion_matrix
from lib.utils import AverageMeter, accuracy
from sklearn.metrics import fbeta_score, confusion_matrix

def evaluate_merged(model, ema_model, emb_model, dataloader, criterion, beta=0.5,experts_test=None,cntx=None,experts_test_bin=None):
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
    choice = 0  # fix expert
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
    


      
        with torch.no_grad():
            # Main model processing
            try:
                logits = model(embedding, expert_cntx).squeeze(0)
            except Exception as e:
                logits = model(embedding)
            # logits = model(embedding,expert_cntx).squeeze(0)
            scores = torch.softmax(logits, dim=1)
            top1 = accuracy(scores, lbs, (1,))
            top1_meter.update(top1.item())
            model_preds.extend(torch.argmax(scores, dim=1).cpu().tolist())

            # EMA model processing
            if ema_model is not None:
                ema_logits = ema_model(embedding,expert_cntx).squeeze(0)
                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_top1 = accuracy(ema_scores, lbs, (1,))
                ema_top1_meter.update(ema_top1.item())
                ema_preds.extend(torch.argmax(ema_scores, dim=1).cpu().tolist())

            true_labels.extend(lbs.cpu().tolist())


    model.eval()

    # Calculate F0.5 scores
    model_f05 = fbeta_score(true_labels, model_preds, beta=beta)
    ema_f05 = None
    ema_top1_avg = None
    
    if ema_model is not None:
        ema_f05 = fbeta_score(true_labels, ema_preds, beta=beta)
        ema_top1_avg = ema_top1_meter.avg

    cm_model = confusion_matrix(true_labels, model_preds)


    return (
        top1_meter.avg,
        ema_top1_avg,
        model_f05,
        ema_f05,
        cm_model
    )
