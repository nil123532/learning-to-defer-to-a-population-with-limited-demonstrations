import random
import numpy as np

from lib.datasets import coarse2sparse



#NR - Creating expert class to return precomputed label when given index

class PreComputedExpert():
    def __init__(self, expert_labels_1d):
            """
            expert_labels_1d a 1D array of length 50,000 or 10,000
            containing the precomputed labels for exactly one expert.
            """
            self.expert_labels_1d = expert_labels_1d
        
    def __call__(self, images, true_labels, indices=None):
        """
        images:       [B, C, H, W]  (unused here for label lookup)
        true_labels:  [B]           (unused here for label lookup)
        indices:      [B]           (actual sample indices)
        Return the expert’s label for each index in the batch.
        """
        if indices is None:
            raise ValueError("No indices provided to expert; cannot fetch precomputed labels.")
        

        batch_indices = indices.cpu().numpy()

        # Return the expert's label for each index in the batch
        return self.expert_labels_1d[batch_indices]

# expert correct in class_oracle with prob. p_in; correct on other classes with prob. p_out
class SyntheticExpertOverlap():
    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1):
        self.expert_static = True
        self.classes_oracle = classes_oracle
        if isinstance(self.classes_oracle, int):
            self.classes_oracle = [self.classes_oracle]
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out

    def __call__(self, images, labels, labels_sparse=None):
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
                        outs[i] = labels[i].item()
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
                else:
                    coin_flip = np.random.binomial(1, self.p_out)
                    if coin_flip == 1:
                        outs[i] = labels[i].item()
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
            else:
                if labels[i] in self.classes_oracle:
                    coin_flip = np.random.binomial(1, self.p_in)
                    if coin_flip == 1:
                        outs[i] = labels[i]
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
                else:
                    coin_flip = np.random.binomial(1, self.p_out)
                    if coin_flip == 1:
                        outs[i] = labels[i]
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
        return outs

