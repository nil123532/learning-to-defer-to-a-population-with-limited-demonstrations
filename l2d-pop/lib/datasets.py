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
from lib.utils import ROOT,BalancedBatchSampler
# from medmnist import PathMNIST , OrganAMNIST , OrganCMNIST , BloodMNIST
from torchvision.datasets import ImageFolder
from pathlib import Path
from collections import Counter


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
            if isinstance(img, torch.Tensor):              # (3,32,32) uint8
                img = transforms.functional.to_pil_image(img)   # channel‑first ok

                
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
    def __init__(self, images, labels, transform, n_cntx_pts=50, device='cpu', original_indices=None, balanced=False, **kwargs):
        self.n_cntx_pts = n_cntx_pts
        self.device = device
        self.with_additional_label = False
        if original_indices is not None:
            self.with_additional_label = True

        dataset = MyVisionDataset(images, labels, transform,original_indices)

        if balanced:
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

        elif expert_type == "limited_demo":
            images_train, _, targets_train, _, indices_train, _ = train_test_split(images_train,targets_train,indices_train,train_size=500,random_state=seed,stratify=targets_train) 
            print("Data for  limited demo")
            train_dataset = MyVisionDataset(images_train,targets_train,transform_train,indices_train)
            val_dataset = MyVisionDataset(images_val,targets_val,transform_test,indices_val)
            test_dataset = datasets.CIFAR10(root=ROOT+'/data', train=False, download=True, transform=transform_test)
            original_indices = np.arange(len(test_dataset))
            test_dataset = MyVisionDataset(test_dataset.data,test_dataset.targets,transform_test,original_indices)
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
    # img_size = (16, 16)
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

    elif expert_type == "limited_demo": 
        images_train, _, targets_train, _ , indices_train , _ = \
            train_test_split(images_train,targets_train,indices_train,train_size=473,random_state=0,stratify=targets_train) 
        print("Data for  limited demo")
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train, indices_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test, indices_val)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test, indices_test)

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

    # 90/10 split into train/val (unseeded)
    images_train, images_val, targets_train, targets_val , indices_train , indices_val = \
        train_test_split(train_images_all, train_targets_all, original_train_indices, train_size=0.9, random_state=0, stratify=train_targets_all)


    if expert_type is None:
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)

    elif expert_type == "limited_demo": 
        images_train, _, targets_train, _ , indices_train , _ = \
            train_test_split(images_train,targets_train,indices_train,train_size=500,random_state=0,stratify=targets_train) 
        print("Data for  limited demo")
        train_dataset = MyVisionDataset(images_train,targets_train,transform_train,indices_train)
        val_dataset = MyVisionDataset(images_val,targets_val,transform_test,indices_val)
        test_dataset = MyVisionDataset(images_test,targets_test,transform_test,indices_test)
    else:
        print("Data for generated experts")
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train,  indices_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test, indices_val)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test, indices_test)
    
    return train_dataset, val_dataset, test_dataset

def load_medical(expert_type=None):
    # train_all = PathMNIST(split='train', download=True, root=ROOT+'/data')
    # val_all = PathMNIST(split='val', download=True, root=ROOT+'/data')
    # test_all = PathMNIST(split='test', download=True, root=ROOT+'/data')
    
    train_all = OrganAMNIST(split='train', download=True, root=ROOT+'/data')
    val_all = OrganAMNIST(split='val', download=True, root=ROOT+'/data')
    test_all = OrganAMNIST(split='test', download=True, root=ROOT+'/data')
    #do we need to squeeze the labels?
    images_train, targets_train = train_all.imgs, train_all.labels
    

    images_val, targets_val = val_all.imgs, val_all.labels
    images_test, targets_test = test_all.imgs, test_all.labels

    idx_train = np.arange(len(images_train))
    #use 10k from images train 
    images_train, _ , targets_train, _ , idx_train , _ = \
        train_test_split(images_train, targets_train, idx_train, train_size=10000, random_state=0, stratify=targets_train)
    idx_val = np.arange(len(images_val))
    idx_test = np.arange(len(images_test))

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
        transforms.Resize((32,32)),  # Resize all images to 32x32
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_test = transform_train
    
    if expert_type is None:
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
    else:
        print("Data for generated experts")
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train, idx_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test, idx_val)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test, idx_test)

    return train_dataset, val_dataset, test_dataset



# ------------------------------------------------------------------

def load_crc(expert_type=None, split_ratio=(0.8, 0.1, 0.1), seed=42):
    I_ROOT  = ROOT + "/data"                 # same ROOT you use elsewhere
    IMG_DIR = "crc/crc_test"                 # nine tissue sub‑folders live here
    """
    Returns (train_ds, val_ds, test_ds) where each is a MyVisionDataset
    wrapping the same CRC‑VAL‑HE‑7K image list but with its own index mask.
    """
    # -------- 2. read file paths & integer labels -----------------
    full_ds   = ImageFolder(Path(I_ROOT) / IMG_DIR, transform=None)   # CRC folder
    paths     = np.array([p for p, _ in full_ds.imgs])                # shape (7180,)
    labels    = np.array(full_ds.targets)                             # same shape
    orig_idx  = np.arange(len(paths))                                 # 0…7179

    paths_tr, paths_tmp, lbl_tr, lbl_tmp, idx_tr, idx_tmp = train_test_split(
            paths, labels, orig_idx,
            test_size=0.20,
            stratify=labels,
            random_state=seed, shuffle=True)                      # :contentReference[oaicite:2]{index=2}

    # second split: 10 % val, 10 % test
    paths_val, paths_te, lbl_val, lbl_te, idx_val, idx_te = train_test_split(
            paths_tmp, lbl_tmp, idx_tmp,
            test_size=0.50,
            stratify=lbl_tmp,
            random_state=seed, shuffle=True)                      # :contentReference[oaicite:3]{index=3}

    # -------- 4. common transform pipeline -----------------------
    tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.70,.50,.75], [.15,.18,.14]),
    ])

    train_ds = MyVisionDataset(paths_tr, lbl_tr, tfm, idx_tr)
    val_ds   = MyVisionDataset(paths_val, lbl_val, tfm, idx_val)
    test_ds  = MyVisionDataset(paths_te, lbl_te, tfm, idx_te)



    return train_ds, val_ds, test_ds

def load_bmc(expert_type=None):
    I_ROOT  = ROOT + "/data"                 # same ROOT you use elsewhere
    IMG_DIR = "bone_marrow_cell_dataset"                 # nine tissue sub‑folders live here
    """
    Returns (train_ds, val_ds, test_ds) where each is a MyVisionDataset
    wrapping the same BMC image list but with its own index mask.
    """
    # -------- 2. read file paths & integer labels -----------------
    full_ds   = ImageFolder(Path(I_ROOT) / IMG_DIR, transform=None)   # BMC folder
    paths     = np.array([p for p, _ in full_ds.imgs])                # shape (7180,)
    labels    = np.array(full_ds.targets)                             # same shape
    orig_idx  = np.arange(len(paths))                                 # 0…7179

    paths_tr, paths_tmp, lbl_tr, lbl_tmp, idx_tr, idx_tmp = train_test_split(
            paths, labels, orig_idx,
            test_size=0.20,
            stratify=labels,
            random_state=42, shuffle=True)                      # :contentReference[oaicite:2]{index=2}

    print("Length of training paths: ", len(paths_tr))
    print("Length of temporary paths: ", len(paths_tmp))
    # second split: 10 % val, 10 % test
    paths_val, paths_te, lbl_val, lbl_te, idx_val, idx_te = train_test_split(
            paths_tmp, lbl_tmp, idx_tmp,
            test_size=0.50,
            # stratify=lbl_tmp,
            random_state=42, shuffle=True)                      # :contentReference[oaicite:3]{index=3}
    
    def pretty_counts(lbls, split_name):
        from collections import Counter
        counts = Counter(lbls)
        print(f"\n{split_name} split:")
        for i, cls in enumerate(full_ds.classes):
            print(f"{cls:4s}: {counts.get(i, 0):6d}")
        print(f"TOTAL: {len(lbls):6d}")

    pretty_counts(lbl_tr,  "Train")
    pretty_counts(lbl_val, "Val")
    pretty_counts(lbl_te,  "Test")
    # -------- 4. common transform pipeline -----------------------
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),  # Resize all images to 32x32    
        transforms.Normalize([.70,.50,.75], [.15,.18,.14]),
        ])

    train_ds = MyVisionDataset(paths_tr, lbl_tr, tfm, idx_tr)
    val_ds   = MyVisionDataset(paths_val, lbl_val, tfm, idx_val)
    test_ds  = MyVisionDataset(paths_te, lbl_te, tfm, idx_te)

    return train_ds, val_ds, test_ds

# ------------------------------------------------------------------
#  l2d_pop/data/irma/
#  ├── ImageCLEFmed2009_train.02/
#  │   └── ImageCLEFmed2009_train.02/   ← PNG files (00001877.png …)
#  ├── ImageCLEFmed2009_test.03/
#  │   └── ImageCLEFmed2009_test.03/
#  ├── ImageCLEFmed2009_train_codes.02.csv
#  └── ImageCLEFmed2009_test_codes.03.csv
# ------------------------------------------------------------------

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms

def load_irma(label_col="05_class", img_size=32, val_frac=0.10, seed=42):
    """
    Returns (train_ds, val_ds, test_ds) for IRMA 2009.
    
    Args
    ----
    label_col : str
        Which column in the CSV to use as the flat label.
        Options:  "05_class", "06_class", "irma_code", etc.
    img_size  : int or tuple
        Final spatial resolution fed to the network.
    val_frac  : float
        Fraction of the training set held out for validation.
    seed      : int
        Random seed for the sk‑learn splits.
    """
    I_ROOT  = Path(ROOT) / "data" / "irma"                # ROOT defined in your env
    TRAIN_CSV = I_ROOT / "ImageCLEFmed2009_train_codes.02.csv"
    TEST_CSV  = I_ROOT / "ImageCLEFmed2009_test_codes.03.csv"
    TRAIN_DIR = I_ROOT / "ImageCLEFmed2009_train.02" / "ImageCLEFmed2009_train.02"
    TEST_DIR  = I_ROOT / "ImageCLEFmed2009_test.03"  / "ImageCLEFmed2009_test.03"

    # ---------- 1. read training CSV ----------------------------------------
    df_tr  = pd.read_csv(TRAIN_CSV, sep=";", na_values="\\N")
    df_tr = df_tr[df_tr[label_col].notna()]               # drop unlabeled rows
    img_ids = df_tr["image_id"].astype(int).to_numpy()
    labels  = df_tr[label_col].astype(int).to_numpy()
    paths   = np.array([TRAIN_DIR / f"{iid:08d}.png" for iid in img_ids])
    orig_idx = np.arange(len(paths))

    # ---------- 2. split train → (train,val) -------------------------------
    paths_tr, paths_val, lbl_tr, lbl_val, idx_tr, idx_val = train_test_split(
        paths, labels, orig_idx,
        test_size=val_frac,
        stratify=labels, random_state=seed, shuffle=True)

    # ---------- 3. read test CSV (labels may be missing) -------------------
    df_te  = pd.read_csv(TEST_CSV,  sep=";", na_values="\\N")
    img_ids_te = df_te["image_id"].astype(int).to_numpy()
    paths_te = np.array([TEST_DIR / f"{iid:08d}.png" for iid in img_ids_te])
    # some challenges hide test labels → set to -1 if absent
    lbl_te = df_te[label_col].fillna(-1).astype(int).to_numpy()
    idx_te = np.arange(len(paths_te))

    # ---------- 4. sanity print --------------------------------------------
    from collections import Counter
    def pretty_counts(lbls, split):
        cnt = Counter(lbls)
        print(f"\n{split} split:")
        for c in sorted(cnt):
            print(f"class {c:3d}: {cnt[c]:5d}")
        print(f"TOTAL    : {len(lbls):5d}")

    pretty_counts(lbl_tr,  "Train")
    pretty_counts(lbl_val, "Val")
    pretty_counts(lbl_te,  "Test (labels may be -1)")

    # ---------- 5. transforms ----------------------------------------------
    tfm = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32) if hasattr(transforms, "ConvertImageDtype") else transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB‑ified later
    ])

    # ---------- 6. wrap in MyVisionDataset ---------------------------------
    train_ds = MyVisionDataset(paths_tr, lbl_tr,  tfm, idx_tr)
    val_ds   = MyVisionDataset(paths_val, lbl_val, tfm, idx_val)
    test_ds  = MyVisionDataset(paths_te, lbl_te,  tfm, idx_te)

    return train_ds, val_ds, test_ds


def load_mice(expert_type=None):
    I_ROOT  = ROOT + "/data"                 # same ROOT you use elsewhere
    IMG_DIR = "mice"                 # nine tissue sub‑folders live here
    """
    Returns (train_ds, val_ds, test_ds) where each is a MyVisionDataset
    wrapping the same BMC image list but with its own index mask.
    """
    # -------- 2. read file paths & integer labels -----------------
    full_ds   = ImageFolder(Path(I_ROOT) / IMG_DIR, transform=None)   # BMC folder
    paths     = np.array([p for p, _ in full_ds.imgs])                # shape (7180,)
    labels    = np.array(full_ds.targets)                             # same shape
    orig_idx  = np.arange(len(paths))                                 # 0…7179

    paths_tr, paths_tmp, lbl_tr, lbl_tmp, idx_tr, idx_tmp = train_test_split(
            paths, labels, orig_idx,
            test_size=0.20,
            stratify=labels,
            random_state=42, shuffle=True)                      # :contentReference[oaicite:2]{index=2}

    print("Length of training paths: ", len(paths_tr))
    print("Length of temporary paths: ", len(paths_tmp))
    # second split: 10 % val, 10 % test
    paths_val, paths_te, lbl_val, lbl_te, idx_val, idx_te = train_test_split(
            paths_tmp, lbl_tmp, idx_tmp,
            test_size=0.50,
            # stratify=lbl_tmp,
            random_state=42, shuffle=True)                      # :contentReference[oaicite:3]{index=3}
    
    def pretty_counts(lbls, split_name):
        from collections import Counter
        counts = Counter(lbls)
        print(f"\n{split_name} split:")
        for i, cls in enumerate(full_ds.classes):
            print(f"{cls:4s}: {counts.get(i, 0):6d}")
        print(f"TOTAL: {len(lbls):6d}")

    pretty_counts(lbl_tr,  "Train")
    pretty_counts(lbl_val, "Val")
    pretty_counts(lbl_te,  "Test")
    # -------- 4. common transform pipeline -----------------------
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),  # Resize all images to 32x32    
        transforms.Normalize([.70,.50,.75], [.15,.18,.14]),
        ])

    train_ds = MyVisionDataset(paths_tr, lbl_tr, tfm, idx_tr)
    val_ds   = MyVisionDataset(paths_val, lbl_val, tfm, idx_val)
    test_ds  = MyVisionDataset(paths_te, lbl_te, tfm, idx_te)

    return train_ds, val_ds, test_ds


def load_mice_pt(expert_type=None):
    """
    Returns (train_ds, val_ds, test_ds) using the pre‑packed mice_32x32.pt file.
    Splits are 80 % / 10 % / 10 % stratified on class labels.
    """
    pt_file = Path(ROOT) / "data" / "mice"/ "nafld_32x32.pt"
    blob = torch.load(pt_file, map_location="cpu")   # expects keys 'x', 'y'
    images, labels = blob["x"], blob["y"]            # (N,3,32,32), (N,)

    # ------------- split indices the same way as before -----------------
    all_idx = np.arange(len(images))
    idx_tr, idx_tmp, lbl_tr, lbl_tmp = train_test_split(
        all_idx, labels, test_size=0.20,
        stratify=labels, random_state=42, shuffle=True)

    idx_val, idx_te, lbl_val, lbl_te = train_test_split(
        idx_tmp, lbl_tmp, test_size=0.50,
        random_state=42, shuffle=True)

    # ------------- keep only ToTensor() as requested --------------------
    tfm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([.70, .50, .75], [.15, .18, .14])
                               ])

    train_ds = MyVisionDataset(images[idx_tr], lbl_tr, tfm, idx_tr)
    val_ds   = MyVisionDataset(images[idx_val], lbl_val, tfm, idx_val)
    test_ds  = MyVisionDataset(images[idx_te],  lbl_te,  tfm, idx_te)

    return train_ds, val_ds, test_ds