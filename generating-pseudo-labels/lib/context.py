from attrdict import AttrDict
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.cifar import VisionDataset
import pandas as pd 
from PIL import Image
import os
from torchvision.transforms.functional import to_pil_image

ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])


# --- NEW: exact-balance batch sampler ---------------------------------
class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Yields a list of indices such that every batch contains 
    the same number of examples from each class.
    """
    def __init__(self, labels, batch_size: int):
        self.labels   = np.asarray(labels)
        self.classes  = np.unique(self.labels)
        self.n_cls    = len(self.classes)

        if batch_size % self.n_cls:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple "
                f"of #classes ({self.n_cls}) to get perfect balance."
            )

        self.k = batch_size // self.n_cls        # samples / class / batch
        self.lookup = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __iter__(self):
        while True:                              # endless generator
            batch_idx = []
            for c in self.classes:
                idx = np.random.choice(
                    self.lookup[c],
                    self.k,
                    replace=len(self.lookup[c]) < self.k   # oversample if needed
                )
                batch_idx.extend(idx)
            np.random.shuffle(batch_idx)         # keep batches i.i.d.
            yield batch_idx

    def __len__(self):                           # not really used
        return len(self.labels) // (self.k * self.n_cls)


class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform,original_indices=None):
        super().__init__(ROOT+'/data', transform=transform)
        self.data, self.targets = images, labels
        self.targets = torch.asarray(self.targets, dtype=torch.int64)
        self.original_indices = original_indices
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        # img = img.numpy()
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if isinstance(img,str):
                img = Image.open(img).convert('RGB')

                
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.original_indices is not None:
            original_index = self.original_indices[index]
            return img, target, original_index

        return img, target

    def __len__(self):
        return len(self.data)


class ContextSampler():
    def __init__(self, images, labels, transform, original_indices=None, n_cntx_pts=50, device='cpu',use_balanced_sampler=True, **kwargs):
        self.n_cntx_pts = n_cntx_pts
        self.device = device
        self.with_additional_label = False
        if original_indices is not None:
            self.with_additional_label = True


        dataset = MyVisionDataset(images, labels, transform,original_indices=original_indices)

        if use_balanced_sampler:
            batch_sampler = BalancedBatchSampler(labels, batch_size=self.n_cntx_pts)
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

        else:    
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.n_cntx_pts, shuffle=True, drop_last=True, **kwargs)
        
        self.data_iter = iter(self.dataloader)

    def _balanced_sample(self):
        try:
            data_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data_batch = next(self.data_iter)

        if self.with_additional_label:
            input_all, target_all, indices = data_batch
            input_all, target_all, indices = input_all.to(self.device), target_all.to(self.device), indices.to(self.device)
            return input_all, target_all, indices
        else:
            input_all, target_all = data_batch
            input_all, target_all = input_all.to(self.device), target_all.to(self.device)
            #print distribution of target_all
            
            return input_all, target_all
        
       


    def sample(self, n_experts=1):
        # Not resample for multiple experts (at train-time)
        cntx = AttrDict()
        if self.with_additional_label:
            input, target, indices = self._balanced_sample()
            cntx.yc_index = indices.unsqueeze(0).repeat(n_experts,1)
        else:
            input, target = self._balanced_sample()
            cntx.yc_index = None
        cntx.xc = input.unsqueeze(0).repeat(n_experts,1,1,1,1)
        cntx.yc = target.unsqueeze(0).repeat(n_experts,1)

        return cntx
    
    def reset(self):
        self.data_iter = iter(self.dataloader)
    def iterate_all(self, batch_size = None):
        """
        Generator that yields every sample in the dataset exactly once.

        Yields
        ------
        imgs  : torch.Tensor  – shape [B, C, H, W]
        lbls  : torch.Tensor  – shape [B]
        idxs  : torch.Tensor  – shape [B]  (original indices / sparse labels)
        """
        bs = batch_size or self.n_cntx_pts
        full_loader = torch.utils.data.DataLoader(
            self.dataloader.dataset,
            batch_size=bs,
            shuffle=False,       # deterministic pass
            drop_last=False      # keep the final partial batch
        )
        #return indices here always
        for data_batch in full_loader:
            imgs, lbls , idxs = data_batch

            yield imgs.to(self.device), lbls.to(self.device) , idxs