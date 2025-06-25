import random
import numpy as np
import sys
from sklearn.metrics import accuracy_score
import pandas as pd

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