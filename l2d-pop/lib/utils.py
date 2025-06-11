import logging
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from attrdict import AttrDict


ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BetaScore:
    def __init__(self, prox_coef = 0.5):
        self.prox_coef = prox_coef
    
    #NR - updated lambda function
    def update_prox_w_ess_factor(self,cs_model, x, beta=None):
            '''
                This function calculates effective sample size (ESS):
                ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
                ESS = ESS / n where n is number of samples to normalize
                x: is (n, D)
            '''
            n = x.shape[0]
            if beta is not None:
                # beta results should be same as using cs_model.predict_proba(x)[:,0] if no clipping
                w = ((torch.sum(beta)**2) /(torch.sum(beta**2) +  np.float32(1e-7) ) )/n
                ess_factor = np.float32(w.numpy())

            else:
                # step 1: get prob class 1
                p0 = cs_model.predict_proba(x)[:,0]
                w =  p0 / ( 1 - p0 +  np.float32(1e-7) )
                w = (np.sum(w)**2) / (np.sum(w**2) +  np.float32(1e-7) )
                ess_factor = np.float32(w) / n

            # since we assume task_i is class -1, and replay buffer is 1, then
            ess_prox_factor = 1.0 - ess_factor

            if np.isnan(ess_prox_factor) or np.isinf(ess_prox_factor) or ess_prox_factor <=  np.float32(1e-7) : # make sure that it is valid
                return 0.55

            else:
                return ess_prox_factor

    #NR - get propensity score
    def get_propensity_score(self,cs_model,ctxt,beta_clip=1.0):
        
        #step 1: f(X)
        ctxt = ctxt.cpu().numpy()
        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_
        
        #step 2: convert to torch 
        f_prop = torch.from_numpy(f_prop).float()
        
        #clip it 
        f_prop = f_prop.clamp(min=-beta_clip)
        
        # step 3: exp(-f(X)), f_score: N * 1
        f_score = torch.exp(-f_prop)
        f_score[f_score < 0.1]  = 0 # for numerical stability
        
        self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)
        
        return f_score



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_logger(filename, mode='a'):
    try:
        os.remove(filename)
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger




def get_balanced_cntx(ctx: AttrDict, replace=False, generator=None) -> AttrDict:
    """
    Return a new AttrDict where every expert contributes the same number
    of samples.  The overall total is unchanged.
    """
    E = ctx.xc.size(0)
    counts = [ctx.xc[i].size(0) for i in range(E)]
    total  = sum(counts)
    N      = total // E           # target per expert

    keep_sparse = 'yc_sparse' in ctx and ctx.yc_sparse is not None
    keep_mc     = 'mc'        in ctx and ctx.mc is not None

    # collect into plain lists
    xc_list = []
    yc_list = []
    sparse_list = [] if keep_sparse else None
    mc_list     = [] if keep_mc     else None

    for i in range(E):
        n_avail = counts[i]
        if n_avail < N and not replace:
            raise ValueError(
                f"Expert {i} only has {n_avail} samples; need {N}.  "
                f"Either relax the divisibility assumption or set replace=True."
            )

        idx = (
            torch.randint(n_avail, (N,), generator=generator, device=ctx.xc.device)
            if replace else
            torch.randperm(n_avail, generator=generator, device=ctx.xc.device)[:N]
        )

        xc_list.append(ctx.xc[i, idx])
        yc_list.append(ctx.yc[i, idx])

        if keep_sparse:
            ys_i = ctx.yc_sparse[i]
            sparse_list.append(None if ys_i is None else ys_i[idx])
        if keep_mc:
            mc_list.append(ctx.mc[i, idx])

    # stack into tensors
    xc_final = torch.stack(xc_list)   # [E, N, C, H, W]
    yc_final = torch.stack(yc_list)   # [E, N]

    # build final AttrDict in one go
    balanced = AttrDict(xc=xc_final, yc=yc_final)

    if keep_sparse:
        # if every slice was non-None, stack; else leave as list of None/arrays
        if all(v is not None for v in sparse_list):
            balanced.yc_sparse = torch.stack(sparse_list)
        else:
            balanced.yc_sparse = sparse_list

    if keep_mc:
        balanced.mc = torch.stack(mc_list)

    return balanced

def train_cs(test_cntx, train_cntx, model):
    """
    Trains a covariate shift correction model using expert contexts.
    
    Args:
        test_cntx (Context): Test context containing (xc, yc, mc) for the target expert.
                             Shapes: (1, num_test_samples, ...)
        train_cntx (Context): Train contexts containing data for all train experts.
                              Shapes: (num_train_experts, samples_per_expert, ...)
        num_classes (int): Number of classes in the dataset.
    
    Returns:
        model (LogisticRegression): Trained logistic regression model.
        info (tuple): Information about training (pos_samples, neg_samples, score).
    """
    # Extract features and labels from the test context
    balanced_cntx = get_balanced_cntx(train_cntx)
    pos_features = model.get_context_encoding(test_cntx).cpu().numpy()
    neg_features = model.get_context_encoding(balanced_cntx).cpu().numpy()

  
    # Combine data and labels
    X = np.concatenate([pos_features, neg_features])
    y = np.concatenate([
        -np.ones(pos_features.shape[0]), 
        np.ones(neg_features.shape[0])
    ])

    # Train logistic regression
    model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    model.fit(X, y)
    score = model.score(X, y)
    
    #Model Accuracy 
    # print("Model Accuracy: ", score)
    
    
    return model

def get_prox_penalty(model_t, model_target):
    '''
        This function calculates ||theta - theta_t||
    '''
    param_prox = []
    for p, q in zip(model_t.parameters(), model_target.parameters()):
        # q should ne detached
        param_prox.append((p - q.detach()).norm()**2)

    result = sum(param_prox)

    return result


