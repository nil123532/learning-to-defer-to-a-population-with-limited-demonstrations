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
from lib.utils import ROOT

#NR - To split based on patient id 
def extract_patient_id(file_path):
    """Extract first 6 characters after 'train/' as patient ID."""
    filename = os.path.basename(file_path)  # e.g., "535989-IMG015x013-3.JPG"
    patient_id = filename[:6]  # First 6 characters
    return patient_id

#NR - Added original indices
class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform, original_indices=None):
        super().__init__(ROOT+'/data', transform=transform)
        self.data, self.targets = images, labels
        self.targets = torch.asarray(self.targets, dtype=torch.int64)
        self.original_indices = original_indices

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
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
            return img,target,self.original_indices[index]
            
        else:
            return img, target

    def __len__(self):
        return len(self.data)



class ContextSampler():
    def __init__(self, images, labels, transform, n_cntx_pts=50, device='cpu', original_indices=None, **kwargs):
        self.n_cntx_pts = n_cntx_pts
        self.device = device
        self.with_additional_label = False
        if original_indices is not None:
            self.with_additional_label = True

        dataset = MyVisionDataset(images, labels, transform,original_indices)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.n_cntx_pts, shuffle=True, drop_last=True, **kwargs)
        self.data_iter = iter(self.dataloader)

    def _balanced_sample(self):
        try:
            data_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data_batch = next(self.data_iter)

        if self.with_additional_label:
            input_all, target_all, target_all_sparse = data_batch
            input_all, target_all, target_all_sparse = input_all.to(self.device), target_all.to(self.device), target_all_sparse.to(self.device)
            return input_all, target_all, target_all_sparse
        else:
            input_all, target_all = data_batch
            input_all, target_all = input_all.to(self.device), target_all.to(self.device)
            return input_all, target_all


    def sample(self, n_experts=1):
        # Not resample for multiple experts (at train-time)
        cntx = AttrDict()
        if self.with_additional_label:
            input, target, target_sparse = self._balanced_sample()
            cntx.yc_sparse = target_sparse.unsqueeze(0).repeat(n_experts,1)
        else:
            input, target = self._balanced_sample()
            cntx.yc_sparse = None
        cntx.xc = input.unsqueeze(0).repeat(n_experts,1,1,1,1)
        cntx.yc = target.unsqueeze(0).repeat(n_experts,1)

        return cntx
    
    def reset(self):
        self.data_iter = iter(self.dataloader)


# From https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def coarse2sparse(targets):
    sparse_labels = np.array([[4, 30, 55, 72, 95],
                              [1, 32, 67, 73, 91],
                              [54, 62, 70, 82, 92],
                              [9, 10, 16, 28, 61],
                              [0, 51, 53, 57, 83],
                              [22, 39, 40, 86, 87],
                              [5, 20, 25, 84, 94],
                              [6, 7, 14, 18, 24],
                              [3, 42, 43, 88, 97],
                              [12, 17, 37, 68, 76],
                              [23, 33, 49, 60, 71],
                              [15, 19, 21, 31, 38],
                              [34, 63, 64, 66, 75],
                              [26, 45, 77, 79, 99],
                              [2, 11, 35, 46, 98],
                              [27, 29, 44, 78, 93],
                              [36, 50, 65, 74, 80],
                              [47, 52, 56, 59, 96],
                              [8, 13, 48, 58, 90],
                              [41, 69, 81, 85, 89]])
    return sparse_labels[targets,:]

#NR - added expert_type 
def load_cifar(variety='10', data_aug=False, seed=0, train_split=0.9,expert_type=None):
    assert variety in ['10','20_100']
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if variety == '10':
        train_dataset_all = datasets.CIFAR10(root=ROOT+'/data', train=True, download=True, transform=transform_train)        
        #NR - Changed dataset to return indices such that we can query expert later on in training
        original_indices = np.arange(len(train_dataset_all))
        images_train, images_val, targets_train, targets_val, indices_train , indices_val = \
            train_test_split(train_dataset_all.data, train_dataset_all.targets, original_indices, train_size=train_split, random_state=seed, stratify=train_dataset_all.targets)
        
        
        if expert_type is None:
            train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
            val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
            test_dataset = datasets.CIFAR10(root=ROOT+'/data', train=False, download=True, transform=transform_test)

        else:
            print("Data for noisy experts")
            train_dataset = MyVisionDataset(images_train,targets_train,transform_train,indices_train)
            val_dataset = MyVisionDataset(images_val,targets_val,transform_test,indices_val)
            test_dataset = datasets.CIFAR10(root=ROOT+'/data', train=False, download=True, transform=transform_test)
            original_indices = np.arange(len(test_dataset))
            test_dataset = MyVisionDataset(test_dataset.data,test_dataset.targets,transform_test,original_indices)
    

    return train_dataset, val_dataset, test_dataset



def load_gtsrb(expert_type=None):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [87.1, 79.7, 82.0]],
                                    std=[x / 255.0 for x in [69.8, 66.5, 67.9]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transform_train

    train_dataset_all = datasets.GTSRB(root=ROOT+'/data', split='train', download=True)
    transform_resize = transforms.Resize((32, 32)) # Resize all images to 32x32 (originals are variable size)

    train_images_all = np.vstack([np.array(transform_resize(train_dataset_all[i][0]))[None,:] for i in range(len(train_dataset_all))])
    train_targets_all = [train_dataset_all[i][1] for i in range(len(train_dataset_all))]
    original_train_indices = np.arange(len(train_dataset_all)) 

    test_dataset = datasets.GTSRB(root=ROOT+'/data', split='test', download=True)
    print("Length of test dataset: ", len(test_dataset))
    test_images_all = np.vstack([np.array(transform_resize(test_dataset[i][0]))[None,:] for i in range(len(test_dataset))])
    test_targets_all = [test_dataset[i][1] for i in range(len(test_dataset))]
    original_test_indices = np.arange(len(test_dataset))    

    # # Extract 10,000 examples from full train set + seeded
    # images_train, _, targets_train, _ , indices_train , _ = \
    #     train_test_split(train_images_all, train_targets_all, original_train_indices, train_size=10000, random_state=0, stratify=train_targets_all)

    #Use all training data for gtsrb 
    images_train = train_images_all
    targets_train = train_targets_all
    indices_train = original_train_indices


    # 50/50 split into val/test (unseeded)
    images_val, images_test, targets_val, targets_test, indices_val , indices_test = \
        train_test_split(test_images_all, test_targets_all, original_test_indices, train_size=0.5, random_state=0, stratify=test_targets_all)
    


    if expert_type is None:
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)


    else:
        print("Data for generated experts")
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train, indices_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test, indices_val)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test, indices_test)
    
    
    print("Length of train indices: ", len(indices_train))
    print("Length of val indices: ", len(indices_val))
    print("Length of test indices: ", len(indices_test))
    return train_dataset, val_dataset, test_dataset



def load_fashion_mnist(expert_type=None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
            
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_test = transform_train

    train_dataset_all = datasets.FashionMNIST(root=ROOT+'/data', train=True, download=True)
    transform_resize = transforms.Resize((32, 32)) # Resize all images to 32x32 (originals are variable size)

    train_images_all = np.vstack([np.array(transform_resize(train_dataset_all[i][0]))[None,:] for i in range(len(train_dataset_all))])
    train_targets_all = [train_dataset_all[i][1] for i in range(len(train_dataset_all))]
    original_train_indices = np.arange(len(train_dataset_all)) 

    test_dataset = datasets.FashionMNIST(root=ROOT+'/data', train=False, download=True)
    test_images_all = np.vstack([np.array(transform_resize(test_dataset[i][0]))[None,:] for i in range(len(test_dataset))])
    test_targets_all = [test_dataset[i][1] for i in range(len(test_dataset))]
    
    original_test_indices = np.arange(len(test_dataset))  

    images_test = test_images_all   
    targets_test = test_targets_all
    indices_test = original_test_indices  

    # 80/20 split into train/val (unseeded)
    images_train, images_val, targets_train, targets_val , indices_train , indices_val = \
        train_test_split(train_images_all, train_targets_all, original_train_indices, train_size=0.9, random_state=0, stratify=train_targets_all)


    if expert_type is None:
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
    else:
        print("Data for generated experts")
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train,  indices_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test, indices_val)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test, indices_test)
    
    return train_dataset, val_dataset, test_dataset

