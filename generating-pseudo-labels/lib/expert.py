import random
import numpy as np
import sys
from sklearn.metrics import accuracy_score
import pandas as pd
import torch

# expert correct in class_oracle with prob. p_in; correct on other classes with prob. p_out
class SyntheticExpertOverlap():
    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1,binary=True):
        self.expert_static = True
        self.classes_oracle = classes_oracle
        if isinstance(self.classes_oracle, int):
            self.classes_oracle = [self.classes_oracle]
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out
        self.binary = binary

    def __call__(self, images = None, labels = None, labels_sparse=None):
        use_arr = False
        if isinstance(labels,list):
            use_arr = True

        batch_size = labels.size()[0] if not use_arr else len(labels) 
        outs = [0] * batch_size
        for i in range(0, batch_size):
            
            if not use_arr:
                if labels[i].item() in self.classes_oracle:
                    coin_flip = np.random.binomial(1, self.p_in)
                    if coin_flip == 1:
                        outs[i] = labels[i].item() if not self.binary else 1
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
                else:
                    coin_flip = np.random.binomial(1, self.p_out)
                    if coin_flip == 1:
                        outs[i] = labels[i].item()
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1) if not self.binary else 0
            else:
               
                if labels[i] in self.classes_oracle:
                    coin_flip = np.random.binomial(1, self.p_in)
                    if coin_flip == 1:
                        outs[i] = labels[i] if not self.binary else 1
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
                else:
                    coin_flip = np.random.binomial(1, self.p_out)
                    if coin_flip == 1:
                        outs[i] = labels[i]
                    if coin_flip == 0:
                        while True:
                            outs[i] = random.randint(0, self.n_classes-1)
                            if outs[i] != labels[i]:  # ensure not the same as the true label
                                break
                        
                        outs[i] = outs[i] if not self.binary else 0

        return outs
    

class CIFAR100Expert:
    # fine-label (0–99) → coarse-label (0–19) map
    fine_to_coarse = {
        0: 4,  1: 1,  2:14,  3: 8,  4: 0,  5: 6,  6: 7,  7: 7,  8:18,  9: 3,
        10: 3, 11:14, 12: 9, 13:18, 14: 7, 15:11, 16: 3, 17: 9, 18: 7, 19:11,
        20: 6, 21:11, 22: 5, 23:10, 24: 7, 25: 6, 26:13, 27:15, 28: 3, 29:15,
        30: 0, 31:11, 32: 1, 33:10, 34:12, 35:14, 36:16, 37: 9, 38:11, 39: 5,
        40: 5, 41:19, 42: 8, 43: 8, 44:15, 45:13, 46:14, 47:17, 48:18, 49:10,
        50:16, 51: 4, 52:17, 53: 4, 54: 2, 55: 0, 56:17, 57: 4, 58:18, 59:17,
        60:10, 61: 3, 62: 2, 63:12, 64:12, 65:16, 66:12, 67: 1, 68: 9, 69:19,
        70: 2, 71:10, 72: 0, 73: 1, 74:16, 75:12, 76: 9, 77:13, 78:15, 79:13,
        80:16, 81:19, 82: 2, 83: 4, 84: 6, 85:19, 86: 5, 87: 5, 88: 8, 89:19,
        90:18, 91: 1, 92: 2, 93:15, 94: 6, 95: 0, 96:17, 97: 8, 98:14, 99:13
    }

    def __init__(self,
                 classes_oracle,      # int or list/tuple of coarse IDs (0-19)
                 *,
                 n_coarse: int = 20,  # keep the same default signature
                 p_in: float = 1.0,
                 p_out: float = 0.1,
                 binary: bool = True):
        if isinstance(classes_oracle, int):
            classes_oracle = [classes_oracle]
        self.classes_oracle = set(classes_oracle)

        self.n_coarse = n_coarse
        self.p_in     = p_in
        self.p_out    = p_out
        self.binary   = binary
        self.probs = np.load('data/mistake_probs.npy')  # mistake probabilities
        # no extra RNG object – SyntheticExpertOverlap uses the global numpy RNG

    # ------------------------------------------------------------------ #
    def __call__(self, images=None, labels=None, labels_sparse=None):
        """
        Make the expert behave like SyntheticExpertOverlap:

        • With prob p_in (inside oracle) / p_out (outside) it predicts the
          *correct coarse label*; otherwise it produces a random label.
        • In binary mode it outputs 1 for oracle classes, 0 otherwise.
        """
        if labels is None:
            raise ValueError("labels must be provided")

        # Accept torch tensor, numpy array or list
        if torch.is_tensor(labels):
            labels = labels.cpu().tolist()
        elif isinstance(labels, np.ndarray):
            labels = labels.tolist()

        batch_size = len(labels)
        outs = [0] * batch_size
        np.random.seed(123)  # for reproducibility
        # for i, fine_y in enumerate(labels):
        #     if fine_y in self.classes_oracle:
        #         if np.random.uniform(0,1) < self.p_in:
        #             # correct prediction
        #             if self.binary:
        #                 outs[i] = 1
        #             else:
        #                 coarse_y = self.fine_to_coarse[fine_y]
        #                 outs[i] = coarse_y
        #         else:
        #             if self.binary:
        #                 outs[i] = 0 
        #             else:
        #                 outs[i] = self.fine_to_coarse[np.random.choice(range(100), 1, p=self.probs[fine_y])[0]]  # random coarse label
        #     else:
        #         if np.random.uniform(0,1) < self.p_out:
        #             if self.binary:
        #                 outs[i] = 1
        #             else:
        #                 coarse_y = self.fine_to_coarse[fine_y]
        #                 outs[i] = coarse_y
        #         else:
        #             if self.binary:
        #                 outs[i] = 0 
        #             else:
        #                 outs[i] = self.fine_to_coarse[np.random.choice(range(100), 1, p=self.probs[fine_y])[0]]  # random coarse label
       
        for i, fine_y in enumerate(labels):
            # coarse_y  = self.fine_to_coarse[fine_y]
            # in_window = coarse_y in self.classes_oracle
            in_window = fine_y in self.classes_oracle   

            # --- decide whether the expert is correct this time -------------
            coin_flip = np.random.binomial(1, self.p_in if in_window
                                              else self.p_out)

            if coin_flip == 1:       # correct prediction
                if self.binary:
                    outs[i] = 1 
                else:
                    outs[i] = self.fine_to_coarse[fine_y]
            else:                    # incorrect prediction
                if self.binary:
                    # Mirror SyntheticExpertOverlap: random int when inside,
                    # zero when outside (so negatives stay 0)
                    if in_window:
                        print("Never here")
                        outs[i] = random.randint(0, self.n_coarse - 1)
                    else:
                        outs[i] = 0
                else:
                    # draw a *wrong* coarse label
                    # wrong = coarse_y
                    # while wrong == coarse_y:
                    # wrong = random.randint(0, self.n_coarse - 1)
                    # outs[i] = wrong
                    outs[i] = self.fine_to_coarse[np.random.choice(
                        range(100), 1, p=self.probs[fine_y])[0]]

        return outs
    

def get_oracles_classes():
    probs = np.load('data/mistake_probs.npy')
    strength_base = random.randint(0, 100)
    strengths_drawn = np.random.choice(range(100), 59, replace=False, p=probs[strength_base])
    print(f"Base strength: {strength_base}, Drawn strengths: {len(strengths_drawn)}")
    return np.append(strength_base, strengths_drawn)

