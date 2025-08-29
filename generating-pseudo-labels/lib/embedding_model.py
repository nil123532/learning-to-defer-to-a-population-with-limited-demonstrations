import os
import sys
import torch
import timm
import torchvision.transforms as T
from torchvision.models.resnet import resnet18,resnet34
import torch.nn as nn

from lib.wideresnet import WideResNetBase



class EmbeddingModel:
    def __init__(self, train_dir, dataset):
        self.train_dir = train_dir
        self.dataset = dataset.lower()
        print('Dataset:', self.dataset)
        if self.dataset == 'cifar100':
            self.args = {'dataset': self.dataset, 'fe_model': 'efficientnet_b1', 'num_classes': 20, 'batch': 800}
        elif self.dataset == 'cifar10':  # NEW CASE
            self.args = {'dataset': self.dataset, 'fe_model': 'wideresnet', 'num_classes': 10, 'batch': 800}
        elif self.dataset == 'nih':
            self.args = {'dataset': self.dataset, 'fe_model': 'resnet18', 'num_classes': 2, 'batch': 800}
        elif self.dataset == 'gtsrb':
            self.args = {'dataset': self.dataset, 'fe_model': 'wideresnet', 'num_classes': 43, 'batch': 800}
        elif self.dataset == 'ham10000':
            self.args = {'dataset': self.dataset, 'fe_model': 'resnet18', 'num_classes': 7, 'batch': 800}
        elif self.dataset == 'fashion':
            self.args = {'dataset': self.dataset, 'fe_model': 'wideresnet', 'num_classes': 10, 'batch': 800}
        
        else:
            print(f'Dataset {self.dataset} not implemented')
            sys.exit()
        self.device = get_device()
        self.emb_model = self.get_emb_model(os.getcwd())

    def get_emb_model(self, wkdir):
        """Initialize base model

        :param wkdir:
        :return: model
        """
      
        if self.dataset == 'cifar10':
            model = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type='batchnorm')
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model.linear = nn.Identity()

        elif self.dataset == 'gtsrb':
            model = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type='batchnorm',n_classes=self.args['num_classes'])
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model.linear = nn.Identity()
       
        elif self.dataset == 'fashion':
            model = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type='batchnorm')
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model.linear = nn.Identity()

        elif self.dataset == 'cifar100':
            model = timm.create_model(self.args['fe_model'], pretrained=True, num_classes=self.args['num_classes'])
            model = self.load_emb_net_from_checkpoint(model, wkdir)
            model = torch.nn.Sequential(*list(model.children())[:-1])


        model = to_device(model, self.device)
        return model

    def get_embedding(self, batch):
        """Resize CIFAR-10 images to 224x224 for EfficientNet"""
        if self.dataset in ['cifar100']:  # Modified condition for efficientnet
            batch = T.Resize((224, 224))(batch)
        self.emb_model.eval()
    
        return self.emb_model(batch)

    def get_emb_net_dir(self, wkdir):
        """Get training directory of the embedding net

        :param wkdir: Working directory
        :return: base_cnn_dir
        """
        args_base = {'model': self.args['fe_model'],
                     'num_classes': self.args['num_classes'],
                     'batch': 64,
                     'dataset': self.dataset
                     }

        base_cnn_dir = get_train_dir(wkdir, args_base, 'base_net')
        return base_cnn_dir

    def load_emb_net_from_checkpoint(self, emb_model, wkdir, mode='best'):
        """Load base model weights from checkpoint

        :param emb_model: Initialized base model
        :param wkdir: Working directory
        :param mode: Checkpoint to load (best or latest)
        :return: base model
        """
        # get checkpoint
        cp_dir = self.get_emb_net_dir(wkdir) + 'checkpoints/checkpoint.' + mode
        try:
            # load state dict from checkpoint
            checkpoint = torch.load(cp_dir,weights_only=False)
            emb_model.load_state_dict(checkpoint['model_state_dict'])
            print('Found base net checkpoint at', cp_dir)
        except FileNotFoundError:
            print('No base net Checkpoint found at', cp_dir)
            sys.exit()

        # freeze base model layers
        for param in emb_model.parameters():
            param.requires_grad = False

        return emb_model


def get_device():
    """Get active device

    :return: device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    """Load to device

    :param data: Data
    :param device: Device
    :return: Data loaded to device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def concat_args(args, mode):
    """Concatenate args to string

    :param args: Args
    :param mode: Mode
    :return: String
    """
    args_string = mode + '@'
    for key in args:
        if key not in ['batch', 'epochs', 'input_shape']:
            args_string += str(key) + '-' + str(args[key]) + '-'
    return args_string[:-1]


def get_train_dir(wkdir, args, mode):
    """Dynamic path based on dataset"""
    dataset = args.get('dataset', 'cifar100').upper()  # Default to CIFAR100 if missing
    path = os.path.join(wkdir, dataset, concat_args(args, mode)) + '/'
    os.makedirs(path, exist_ok=True)  # Create dirs recursively
    return path

