import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import GTSRB, FashionMNIST, CIFAR10
from torch.utils.data import Dataset

from lib import transform as T
from lib.randaugment import RandomAugment
from lib.sampler import RandomSampler, BatchSampler , WeightedRandomSampler
from lib.context import ContextSampler
from sklearn.model_selection import train_test_split

class TwoCropsTransform:
    """Take 2 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong: Transform for the strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong: Transform for the strong augmentation
    """

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong0: Transform for the first strong augmentation
    :param trans_strong1: Transform for the second strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong0: Transform for the first strong augmentation
    :ivar trans_strong1: Transform for the second strong augmentation
    """

    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]



class GTSRB_Dataset(GTSRB):
    """GTSRB that returns (img, target, index) like CIFAR100_Dataset."""
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target, index
    
class FASHIONMNIST_Dataset(FashionMNIST):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target, index
    


def  load_data_train(L=250, dataset='CIFAR10', dspth='./data'):
    """Load the train dataset

    :param L: Number of labeled instances
    :param dataset: Name of the dataset
    :param dspth: Path of the dataset

    :return: tuple
        - data_x: Images of the labeled set
        - label_x: Label of the labeled set
        - data_u: Images of the unlabeled set
        - label_u: Label of the unlabeled set
    """

    if dataset == 'CIFAR10':
        #download cifar10 dataset
        _ = CIFAR10(root=dspth, train=True, download=True)
        _ = CIFAR10(root=dspth, train=False, download=True)
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [10, 20, 100, 60 , 40, 80,  400, 1000, 200, 250, 2500,500, 4000,40000,400,60000,50000]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 20
        assert L in [None, 40, 80, 120, 200, 400, 1000, 5000]
    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':

        # load images and labels
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
                
        data = np.concatenate(data, axis=0)     # shape = [50000, 3072] for CIFAR-10
        labels = np.concatenate(labels, axis=0) # shape = [50000]
        labels_coarse = transform_to_coarse(labels) if dataset=='CIFAR100' else labels

        # Let's create an array of "global indices" 0..49999
        all_indices = np.arange(len(labels))

        if L is None:
            # If no L, everything is unlabeled (or something).
            data = [el.reshape(3, 32, 32).transpose(1, 2, 0) for el in data]
            return data, labels, None, None, all_indices, None
        else:
            # We split them into labeled vs. unlabeled
            n_class = 10 if dataset=='CIFAR10' else 20 # for CIFAR-10
            n_labels_per_class = L // n_class
            
            data_x, label_x, index_x = [], [], []
            data_u, label_u, index_u = [], [], []
            for i in range(n_class):
                inds_for_class = np.where(labels == i)[0] if dataset=='CIFAR10' else np.where(labels_coarse == i)[0]
                np.random.shuffle(inds_for_class)

                inds_labeled = inds_for_class[:n_labels_per_class]
                inds_unlabeled = inds_for_class[n_labels_per_class:]

                # Labeled subset
                for idx in inds_labeled:
                    img = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
                    data_x.append(img)
                    label_x.append(labels[idx])
                    index_x.append(all_indices[idx])  # store the global index

                # Unlabeled subset
                for idx in inds_unlabeled:
                    img = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
                    data_u.append(img)
                    label_u.append(labels[idx])
                    index_u.append(all_indices[idx])  # store the global index
                
       

            return data_x, label_x, data_u, label_u, index_x, index_u
        
    if dataset == 'GTSRB':
        #Load data
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # 2) load full train & test
        train_full = GTSRB_Dataset(root='./data', split='train',
                                download=True, transform=transform)
        
        n_class = 43 

        all_indices = np.arange(len(train_full))
        targets = np.array([train_full[idx][1] for idx in all_indices])

        if L is None:
            # everything goes into "unlabeled"
            data = [train_full[idx][0] for idx in all_indices]
            return data, targets.tolist(), None, None, all_indices.tolist(), None
        else:
            # how many labels per class
            n_labels_per_class = L // n_class

            data_x, label_x, index_x = [], [], []
            data_u, label_u, index_u = [], [], []

            for cls in range(n_class):
                # indices of this class
                inds_cls = all_indices[targets == cls]
                np.random.shuffle(inds_cls)

                # split
                inds_labeled   = inds_cls[:n_labels_per_class]
                inds_unlabeled = inds_cls[n_labels_per_class:]

                # collect labeled
                for idx in inds_labeled:
                    img, lbl, orig_idx = train_full[idx]
                    data_x.append(img)
                    label_x.append(lbl)
                    index_x.append(orig_idx)

                # collect unlabeled
                for idx in inds_unlabeled:
                    img, lbl, orig_idx = train_full[idx]
                    data_u.append(img)
                    label_u.append(lbl)
                    index_u.append(orig_idx)

        return data_x, label_x, data_u, label_u, index_x, index_u
    
    if dataset == "FASHION":
    

        # 2) load full train & test
        train_full = FASHIONMNIST_Dataset(root='./data', train=True,
                                download=True)
        
        n_class = 10 

        all_indices = np.arange(len(train_full))
        targets = np.array([train_full[idx][1] for idx in all_indices])

        if L is None:
            # everything goes into "unlabeled"
            data = [train_full[idx][0] for idx in all_indices]
            return data, targets.tolist(), None, None, all_indices.tolist(), None
        else:
            # how many labels per class
            n_labels_per_class = L // n_class

            data_x, label_x, index_x = [], [], []
            data_u, label_u, index_u = [], [], []

            for cls in range(n_class):
                # indices of this class
                inds_cls = all_indices[targets == cls]
                np.random.shuffle(inds_cls)

                # split
                inds_labeled   = inds_cls[:n_labels_per_class]
                inds_unlabeled = inds_cls[n_labels_per_class:]

                # collect labeled
                for idx in inds_labeled:
                    img, lbl, orig_idx = train_full[idx]
                    data_x.append(img)
                    label_x.append(lbl)
                    index_x.append(orig_idx)

                # collect unlabeled
                for idx in inds_unlabeled:
                    img, lbl, orig_idx = train_full[idx]
                    data_u.append(img)
                    label_u.append(lbl)
                    index_u.append(orig_idx)

        return data_x, label_x, data_u, label_u, index_x, index_u

        

def load_data_val(dataset='CIFAR10', dspth='./data', cntx_per_class=None):
    """
    Load *test* split and optionally carve out a balanced context subset (x-split).

    Returns
    -------
    data_x, label_x, data_u, label_u, idx_x, idx_u
        data_x   – context  images  (empty list if cntx_per_class is None)
        label_x  – context  labels
        data_u   – test     images  (all remaining samples)
        label_u  – test     labels
        idx_x    – global   indices of x
        idx_u    – global   indices of u
    """
    print("Loading test data for dataset:", dataset)
    if dataset == 'CIFAR10':
        datalist = [osp.join(dspth, 'cifar-10-batches-py', 'test_batch')]
        n_class  = 10
    elif dataset == 'CIFAR100':
        datalist = [osp.join(dspth, 'cifar-100-python', 'test')]
        n_class  = 20   # coarse classes
    elif dataset == 'FASHION':
        n_class  = 10
    elif dataset == 'GTSRB':    
        n_class  = 43

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        # ------------- load raw record arrays -------------
        data, labels = [], []
        for f in datalist:
            with open(f, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels']
                data.append(entry['data'])
                labels.append(lbs)

        data   = np.concatenate(data, axis=0)     # shape [N, 3072]
        labels = np.concatenate(labels, axis=0)   # shape [N]

        # coarse labels for CIFAR-100
        if dataset == 'CIFAR100':
            labels_coarse = transform_to_coarse(labels)
        else:
            labels_coarse = labels

        # reshape to HWC uint8 images
        imgs = [arr.reshape(3, 32, 32).transpose(1, 2, 0) for arr in data]
        all_idx = np.arange(len(imgs))

        # ------------- balanced context split -------------
        if cntx_per_class is None or cntx_per_class == 0:
            # everything is "u" (test) set; context set is empty
            data_x   , label_x   , idx_x = [], [], []
            data_u   , label_u   , idx_u = imgs, labels, all_idx
        else:
            data_x, label_x, idx_x = [], [], []
            data_u, label_u, idx_u = [], [], []
            for cls in range(n_class):
                cls_mask = (labels_coarse == cls)
                cls_inds = np.where(cls_mask)[0]
                if len(cls_inds) < cntx_per_class:
                    raise ValueError(f"Class {cls}: only {len(cls_inds)} samples available, "
                                    f"but cntx_per_class={cntx_per_class}")

                np.random.shuffle(cls_inds)
                x_inds = cls_inds[:cntx_per_class]
                u_inds = cls_inds[cntx_per_class:]

                # context
                for idx in x_inds:
                    data_x.append(imgs[idx])
                    label_x.append(labels[idx])
                    idx_x.append(all_idx[idx])

                # remaining → test
                for idx in u_inds:
                    data_u.append(imgs[idx])
                    label_u.append(labels[idx])
                    idx_u.append(all_idx[idx])

            return data_x, label_x, data_u, label_u, idx_x, idx_u  
        
    if dataset == 'GTSRB':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        test_full  = GTSRB_Dataset(root='./data', split='test',
                               download=True, transform=transform)
        
        all_indices = np.arange(len(test_full))
        targets = np.array([test_full[idx][1] for idx in all_indices])
        L = cntx_per_class * 43
        if L is None:
            # everything goes into "unlabeled"
            data = [test_full[idx][0] for idx in all_indices]
            return data, targets.tolist(), None, None, all_indices.tolist(), None
        else:
            # how many labels per class
            n_labels_per_class = L // n_class

            data_x, label_x, index_x = [], [], []
            data_u, label_u, index_u = [], [], []

            for cls in range(n_class):
                # indices of this class
                inds_cls = all_indices[targets == cls]
                np.random.shuffle(inds_cls)

                # split
                inds_labeled   = inds_cls[:n_labels_per_class]
                inds_unlabeled = inds_cls[n_labels_per_class:]

                # collect labeled
                for idx in inds_labeled:
                    img, lbl, orig_idx = test_full[idx]
                    data_x.append(img)
                    label_x.append(lbl)
                    index_x.append(orig_idx)

                # collect unlabeled
                for idx in inds_unlabeled:
                    img, lbl, orig_idx = test_full[idx]
                    data_u.append(img)
                    label_u.append(lbl)
                    index_u.append(orig_idx)

        return data_x, label_x, data_u, label_u, index_x, index_u
    
    if dataset == 'FASHION':

        test_full  = FASHIONMNIST_Dataset(root='./data', train=False,
                               download=True)
        
        
        all_indices = np.arange(len(test_full))
        targets = np.array([test_full[idx][1] for idx in all_indices])
        L = cntx_per_class * 10
        if L is None:
            # everything goes into "unlabeled"
            data = [test_full[idx][0] for idx in all_indices]
            return data, targets.tolist(), None, None, all_indices.tolist(), None
        else:
            # how many labels per class
            n_labels_per_class = L // n_class

            data_x, label_x, index_x = [], [], []
            data_u, label_u, index_u = [], [], []

            for cls in range(n_class):
                # indices of this class
                inds_cls = all_indices[targets == cls]
                np.random.shuffle(inds_cls)

                # split
                inds_labeled   = inds_cls[:n_labels_per_class]
                inds_unlabeled = inds_cls[n_labels_per_class:]

                # collect labeled
                for idx in inds_labeled:
                    img, lbl, orig_idx = test_full[idx]
                    data_x.append(img)
                    label_x.append(lbl)
                    index_x.append(orig_idx)

                # collect unlabeled
                for idx in inds_unlabeled:
                    img, lbl, orig_idx = test_full[idx]
                    data_u.append(img)
                    label_u.append(lbl)
                    index_u.append(orig_idx)

        return data_x, label_x, data_u, label_u, index_x, index_u

def compute_mean_var():
    """Compute mean and variance of the images from the train set

    :return:
    """
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class FASHIONMNISTDATASET(Dataset):    
    def __init__(self, data, labels, mode, imsize,indices=None):
        super(FASHIONMNISTDATASET, self).__init__()
        self.counter = 0
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
            
        self.indices = indices 
        trans_weak = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            transforms.Normalize(mean, std),
        ])

        trans_strong0 = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            transforms.Normalize(mean, std),
        ])

        trans_strong1 = transforms.Compose([
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            transforms.Normalize(mean, std),
        ])

        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            self.trans = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
                transforms.Normalize(mean, std),
            ])

    def __getitem__(self, idx):

        im, lb = self.data[idx], self.labels[idx]
        
        # The global index is self.indices[idx] (if provided)
        if self.indices is not None:
            global_id = self.indices[idx]
        else:
            global_id = idx  # fallback if we don't have them

        return self.trans(im), lb, global_id
    
    def __len__(self):
        leng = len(self.data)
        return leng

class Cifar(Dataset):
    """Class representing the CIFAR dataset

    :param dataset: Name of the dataset
    :param data: Images
    :param labels: Labels
    :param mode: Mode
    :param imsize: Image size

    :ivar data: Images
    :ivar labels: Labels
    :ivar mode: Mode
    """
    def __init__(self, dataset, data, labels, mode, imsize,indices=None):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            
        self.indices = indices  # store the global indices array here

        # trans_weak = T.Compose([
        #     T.Resize((imsize, imsize)),
        #     T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.Normalize(mean, std),
        #     T.ToTensor(),
        # ])
        # trans_strong0 = T.Compose([
        #     T.Resize((imsize, imsize)),
        #     T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
        #     T.RandomHorizontalFlip(p=0.5),
        #     RandomAugment(2, 10),
        #     T.Normalize(mean, std),
        #     T.ToTensor(),
        # ])
        # trans_strong1 = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])

        trans_weak = transforms.Compose([
            transforms.ToPILImage(),                      # Convert numpy.ndarray to PIL Image
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trans_strong0 = transforms.Compose([
            transforms.ToPILImage(),                      # Convert numpy.ndarray to PIL Image
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # trans_weak = trans_strong0 #NR - Need to change later

        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            self.trans = T.Compose([
                T.Resize((imsize, imsize)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        
        # The global index is self.indices[idx] (if provided)
        if self.indices is not None:
            global_id = self.indices[idx]
        else:
            global_id = idx  # fallback if we don't have them

        return self.trans(im), lb, global_id
    
    def __len__(self):
        leng = len(self.data)
        return leng
    


def get_train_loader(dataset, expert, batch_size, mu, n_iters_per_epoch, L, root='data', method='comatch', imsize=32,weighted=False):
    """Get data loader for the train set

    :param dataset: Name of the dataset
    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param mu: Factor of train batch size of unlabeled samples
    :param n_iters_per_epoch: Number of iteration per epoch
    :param L: Number of labeled instances
    :param root: Path of the dataset
    :param method: Training algorithm (either comatch or fixmatch)
    :param imsize: Size of images

    :return: tuple
        - dl_x: Dataloader for the labeled instances
        - dl_u: Dataloader for the unlabeled instances
    """
    data_x, label_x, data_u, label_u , global_ind_x , global_ind_u = load_data_train(L=L, dataset=dataset, dspth=root)
    
    #NR- Expert labelling done through callable expert class in the main now

    # if expert is not None:
    #     label_x = expert.generate_expert_labels(label_x, binary=True)
    #     if label_u is not None:
    #         label_u = expert.generate_expert_labels(label_u, binary=True)

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        if dataset == 'CIFAR10':
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,

            ])
        cntx_sampler = ContextSampler(data_x,label_x,transform_train,n_cntx_pts=20,device='cuda')
        ds_x = Cifar(
            dataset=dataset,
            data=data_x,
            labels=label_x,
            mode='train_x',
            imsize=imsize,
            indices=global_ind_x  # pass the global indices to the dataset
        )  # return an iter of num_samples length (all indices of samples)
        
        print("unique labels in ds_x: ", np.unique(ds_x.labels))
        sampler_weights = get_class_weights_sampler(ds_x,2) if weighted else None
        if sampler_weights is not None:
            print("Weighted sampler")
            sampler_x = WeightedRandomSampler(sampler_weights, num_samples=n_iters_per_epoch * batch_size, replacement=True)
        else:
            print("Normal Sampler")
            sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )
        if data_u is None:
            return dl_x
        else:
            ds_u = Cifar(
                dataset=dataset,
                data=data_u,
                labels=label_u,
                mode='train_u_%s' % method,
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )
            sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
            # sampler_u = RandomSampler(ds_u, replacement=False)
            batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
            dl_u = torch.utils.data.DataLoader(
                ds_u,
                batch_sampler=batch_sampler_u,
                num_workers=2,
                pin_memory=True
            )


            dl_x_eval = torch.utils.data.DataLoader(
                ds_x,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )

            dl_u_eval = Cifar(
                dataset=dataset,
                data=data_u,
                labels=label_u,
                mode='test',
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )

            dl_u_pred = torch.utils.data.DataLoader(
                dl_u_eval,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )



            return dl_x, dl_u,cntx_sampler,dl_x_eval,dl_u_pred
        
    elif dataset == 'GTSRB':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [87.1, 79.7, 82.0]],
                                    std=[x / 255.0 for x in [69.8, 66.5, 67.9]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        cntx_sampler = ContextSampler(data_x,label_x,transform_train,n_cntx_pts=86,device='cuda')
        ds_x = GTSRBDataset(
            data=data_x,
            labels=label_x,
            mode='train_x',
            imsize=imsize,
            indices=global_ind_x  # pass the global indices to the dataset
        )  # return an iter of num_samples length (all indices of samples)
        
        print("unique labels in ds_x: ", np.unique(ds_x.labels))
        sampler_weights = get_class_weights_sampler(ds_x,2) if weighted else None
        if sampler_weights is not None:
            print("Weighted sampler")
            sampler_x = WeightedRandomSampler(sampler_weights, num_samples=n_iters_per_epoch * batch_size, replacement=True)
        else:
            print("Normal Sampler")
            sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )
        if data_u is None:
            return dl_x
        else:
            ds_u = GTSRBDataset(
                data=data_u,
                labels=label_u,
                mode='train_u_%s' % method,
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )
            sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
            # sampler_u = RandomSampler(ds_u, replacement=False)
            batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
            dl_u = torch.utils.data.DataLoader(
                ds_u,
                batch_sampler=batch_sampler_u,
                num_workers=2,
                pin_memory=True
            )


            dl_x_eval = torch.utils.data.DataLoader(
                ds_x,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )

            dl_u_eval = GTSRBDataset(
                data=data_u,
                labels=label_u,
                mode='other',
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )

            dl_u_pred = torch.utils.data.DataLoader(
                dl_u_eval,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )



            return dl_x, dl_u,cntx_sampler,dl_x_eval,dl_u_pred
    
    elif dataset == 'ham10000': 
        # Labeled dataset
   
        ds_x = HAM10000Dataset(
            data=data_x,
            labels=label_x,
            mode='train_x',
            imsize=imsize,
            indices=global_ind_x
        )
        print("unique labels in ds_x: ", np.unique(ds_x.labels))
        sampler_weights = get_class_weights_sampler(ds_x, 2) if weighted else None
        if sampler_weights is not None:
            print("Weighted sampler")
            sampler_x = WeightedRandomSampler(
                sampler_weights,
                num_samples=n_iters_per_epoch * batch_size,
                replacement=True
            )
        else:
            print("Normal Sampler")
            sampler_x = RandomSampler(
                ds_x,
                replacement=True,
                num_samples=n_iters_per_epoch * batch_size
            )
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )

        # If no unlabeled data, return only the labeled loader
        if data_u is None:
            return dl_x

        # Unlabeled dataset
        ds_u = HAM10000Dataset(
            data=data_u,
            labels=label_u,
            mode=f'train_u_{method}',
            imsize=imsize,
            indices=global_ind_u
        )
        sampler_u = RandomSampler(
            ds_u,
            replacement=True,
            num_samples=mu * n_iters_per_epoch * batch_size
        )
        batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
        dl_u = torch.utils.data.DataLoader(
            ds_u,
            batch_sampler=batch_sampler_u,
            num_workers=2,
            pin_memory=True
        )
    
        return dl_x, dl_u
    
    elif dataset == 'FASHION':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            normalize,
        ])
        cntx_sampler = ContextSampler(data_x,label_x,transform_train,n_cntx_pts=20,device='cuda')
        ds_x = FASHIONMNISTDATASET(
            data=data_x,
            labels=label_x,
            mode='train_x',
            imsize=imsize,
            indices=global_ind_x  # pass the global indices to the dataset
        )  # return an iter of num_samples length (all indices of samples)
        
        print("unique labels in ds_x: ", np.unique(ds_x.labels))
        sampler_weights = get_class_weights_sampler(ds_x,2) if weighted else None
        if sampler_weights is not None:
            print("Weighted sampler")
            sampler_x = WeightedRandomSampler(sampler_weights, num_samples=n_iters_per_epoch * batch_size, replacement=True)
        else:
            print("Normal Sampler")
            sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )
        if data_u is None:
            return dl_x
        else:
            ds_u = FASHIONMNISTDATASET(
                data=data_u,
                labels=label_u,
                mode='train_u_%s' % method,
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )
            sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
            # sampler_u = RandomSampler(ds_u, replacement=False)
            batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
            dl_u = torch.utils.data.DataLoader(
                ds_u,
                batch_sampler=batch_sampler_u,
                num_workers=2,
                pin_memory=True
            )


            dl_x_eval = torch.utils.data.DataLoader(
                ds_x,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )

            dl_u_eval = FASHIONMNISTDATASET(
                data=data_u,
                labels=label_u,
                mode='train_x',
                imsize=imsize,
                indices=global_ind_u  # pass the global indices to the dataset
            )

            dl_u_pred = torch.utils.data.DataLoader(
                dl_u_eval,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                num_workers=2,
                pin_memory=True
            )



            return dl_x, dl_u,cntx_sampler,dl_x_eval,dl_u_pred





def get_val_loader(dataset, expert, batch_size, num_workers, pin_memory=True, root='data', imsize=32):
    """Get data loader for the validation set

    :param dataset: Name of the dataset
    :param expert: Synthetic cifar expert
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param pin_memory: Pin memory
    :param root: Path of the dataset
    :param imsize: Size of images

    :return: Dataloader
    """
    
    data_x, labels_x , data_test , labels_test , indices_x , indices_test = load_data_val(dataset=dataset, dspth=root,cntx_per_class=50)

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # #split 0.05 of data for context sampler and the rest for validation
        # data_test, data_cntx , labels_test , labels_cntx, original_indices_test, original_indices_cntx = train_test_split(data, labels, original_indices, test_size=0.05,                                                                                               random_state=0,shuffle=True,stratify=labels)

        
        print("Length of context set:", len(data_x))
        print("Length of test set:", len(data_test))    
        
        cntx_sampler = ContextSampler(data_x,labels_x,transform_test,original_indices=indices_x,n_cntx_pts=20,device='cuda')
        ds = Cifar(
            dataset=dataset,
            data=data_test,
            labels=labels_test,
            mode='test',
            imsize=imsize,
            indices=indices_test  # pass the global indices to the dataset
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return dl , cntx_sampler
    
    if dataset == 'GTSRB':
        data_x, labels_x , data_test , labels_test , indices_x , indices_test = load_data_val(dataset=dataset, dspth=root,cntx_per_class=11)
        #need new load data here, loading <500 labels
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [87.1, 79.7, 82.0]],
                                    std=[x / 255.0 for x in [69.8, 66.5, 67.9]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        print("Length of context set:", len(data_x))
        print("Length of test set:", len(data_test))    
        
        cntx_sampler = ContextSampler(data_x,labels_x,transform_test,original_indices=indices_x,n_cntx_pts=86,device='cuda')
        ds = GTSRBDataset(
            data=data_test,
            labels=labels_test,
            mode='test',
            imsize=imsize,
            indices=indices_test  # pass the global indices to the dataset
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return dl , cntx_sampler

    
    if dataset == 'ham10000':
        ds = HAM10000Dataset(
            data=data,
            labels=labels,
            mode='test',
            imsize=imsize
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return dl
    
    if dataset == 'FASHION':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            normalize,
        ])
        print("Length of context set:", len(data_x))
        print("Length of test set:", len(data_test))    
        
        cntx_sampler = ContextSampler(data_x,labels_x,transform_test,original_indices=indices_x,n_cntx_pts=20,device='cuda')
        ds = FASHIONMNISTDATASET(
            data=data_test,
            labels=labels_test,
            mode='test',
            imsize=imsize,
            indices=indices_test  # pass the global indices to the dataset
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return dl , cntx_sampler




def transform_to_coarse(targets):
    """Transforms fine targets into coarse targets

    :param targets: Fine targets
    :return: Coarse targets
    """
    coarse = np.array([fine_id_coarse_id()[t] for t in targets])
    return coarse


def fine_id_coarse_id():
    """Mapping between fine and coarse labels

    :return: Mapping as dictionary
    """
    return {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7,
            15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3,
            29: 15, 30: 0,
            31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15,
            45: 13, 46: 14,
            47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10,
            61: 3, 62: 2,
            63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9,
            77: 13, 78: 15,
            79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2,
            93: 15, 94: 6, 95: 0,
            96: 17, 97: 8, 98: 14, 99: 13}

class GTSRBDataset(Dataset):
    """Class representing the GTSRB dataset

    :param data: list or array of images (H×W×C numpy arrays)
    :param labels: list or array of integer labels
    :param mode: one of 'train_x', 'train_u_comatch', 'train_u_fixmatch', or 'test'
    :param imsize: output image size (int)
    :param indices: optional list/array of global indices
    """
    def __init__(self, data, labels, mode, imsize, indices=None):
        super().__init__()
        assert len(data) == len(labels), "data and labels must be same length"
        self.data = data
        self.labels = labels
        self.mode = mode
        self.indices = indices  # global indices mapping, if provided

        # GTSRB normalization stats
        mean = [87.1/255.0, 79.7/255.0, 82.0/255.0]
        std  = [69.8/255.0, 66.5/255.0, 67.9/255.0]

        # define transforms
        trans_weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trans_strong0 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trans_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # choose which transform to apply
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            # returns three crops: [weak, strong0, strong1]
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_fixmatch':
            # returns two crops: [weak, strong0]
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:  # 'test' or any other mode
            self.trans = trans_test

    def __getitem__(self, idx):
        img = self.data[idx]
        lbl = self.labels[idx]
        # determine global index
        global_id = self.indices[idx] if self.indices is not None else idx
        return self.trans(img), lbl, global_id

    def __len__(self):
        return len(self.data)
    