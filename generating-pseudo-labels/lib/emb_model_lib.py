import json
import os
import torch
import timm
import numpy as np
import math
import torch.nn as nn

from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.models.resnet import resnet50, resnet18 , resnet34


import lib.data_loading as prep

from lib.wideresnet import WideResNetBase,WideResNet
from lib.utils import get_train_dir, printProgressBar
from lib.metrics import get_confusion_matrix


class EmbeddingModel:
    """Class representing the embedding model

    :param args: Training arguments for the embedding model
    :param wkdir: Working directory
    :param writer: Tensorboard writer

    :ivar global_step: Global setp
    :ivar args: Training arguments
    :ivar writer: Tensorboard writer
    :ivar device: Active device
    :ivar train_dir: Training directory of the embedding model
    :ivar model: Embedding model
    :ivar optimizer: Optimizer
    :ivar scheduler: Learning rate scheduler
    :ivar loss_function: Loss function
    :ivar train_data: Train dataset
    :ivar test_data: Test dataset
    :ivar val_data: Validation dataset
    :ivar train_loader: Train dataloader
    :ivar test_loader: Test dataloader
    :ivar val_loader: Validation dataloader
    """
    def __init__(self, args, wkdir, writer):
        self.global_step = 0
        self.dataset = args['dataset']
        self.args = args
        self.writer = writer
        self.device = prep.get_device()
        self.train_dir = get_train_dir(wkdir, args, args['name'])
        self.model = self.get_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args['lr'], weight_decay=5e-4, momentum=0.9,
                                         nesterov=True)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.75,
                                                                        min_lr=1e-6,threshold=0.005,threshold_mode='abs',patience=10)
        self.val_acc = 0
       
        
        self.train_data, self.test_data, self.val_data = prep.get_train_val_test_data(expert=None,
                                                                                      model=args['model'],
                                                                                      dataset=args['dataset'])
        print('Train data:', len(self.train_data))
        print('Test data:', len(self.test_data))
        print('Val data:', len(self.val_data))
        
        self.train_loader, self.test_loader, self.val_loader, device = prep.get_data_loader(self.train_data,
                                                                                            self.test_data,
                                                                                            self.val_data,
                                                                                            batch_size=args['batch'])
                                                                                            
        self.loss_function = nn.CrossEntropyLoss().cuda()


        self.save_model_args()

    def get_class_weights(self,dltrain_x, n_classes):
        all_labels = []
        for batch in dltrain_x:
            img, lbs, _ = batch
            all_labels.extend(lbs.numpy())
        all_labels = np.array(all_labels)

        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(n_classes), 
            y=all_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

        return class_weights
    
    def get_model(self):
        """Initialize model

        :return: model
        """
        # load model
        if self.args['model'] == 'wideresnet':
            model = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0, norm_type='batchnorm',n_classes=self.args['num_classes'])
 


        elif self.args['model'] == 'resnet18':
            model = Resnet(self.args['num_classes'])
        elif self.args['model'] == 'resnet34':
            model = Resnet34(self.args['num_classes'])
        
        elif self.args['model'].startswith('vit'):
             model = timm.create_model(self.args['model'] , pretrained=True,num_classes=self.args['num_classes'],global_pool='token')

        elif self.args['model'] == 'resnet20':
            print('Loading ResNet20')   
            model = resnet20(n_classes=self.args['num_classes'])

        else:
            model = timm.create_model(self.args['model'], pretrained=True, num_classes=self.args['num_classes'])
        print('Loaded Model', self.args['model'])
        # load model to device
        model = prep.to_device(model, self.device)

        return model

    def train_one_epoch(self, epoch):
        """Train one epoch

        :param epoch: Epoch
        :return: loss
        """
        self.epoch = epoch
        self.model.train()
        for ii, (data, target, index) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            target = target.long()
            pred = self.model(data)

           
            loss = self.loss_function(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            printProgressBar(ii + 1, math.ceil((len(self.train_data) / self.args['batch'])),
                             prefix='Train Epoch ' + str(epoch + 1) + ':',
                             suffix='Complete', length=40)
        self.scheduler.step(self.val_acc)
        self.global_step += len(index)
        self.writer.add_scalar('LR/lr', self.optimizer.param_groups[0]["lr"], self.epoch)
        self.writer.add_scalar('Loss/total', loss, self.epoch)
        

        return loss

    def get_validation_accuracy(self, epoch =None, return_acc=False, print_acc=True):
        """Get validation accuracy

        :param epoch: Epoch
        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.val_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # get model artificial_expert_labels
            with torch.no_grad():
                output = self.model(data)

            # get predicted classes from model output
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()

            for p in predicted_class:
                predict.append(p)
            for t in target:
                targets.append(t.item())

        # calculate accuracy score
        acc = accuracy_score(targets, predict)
        if print_acc:
            if epoch is None:
                print('Val-Accuracy:', acc)
            else:
                print('Epoch:', epoch + 1, '- Val-Accuracy:', acc)
        if return_acc: 
            self.val_acc = acc
            self.writer.add_scalar('Validation Accuracy', acc, self.epoch)
            return acc

    def get_test_accuracy(self, return_acc=False, print_acc=True):
        """Get test accuracy

        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # get model artificial_expert_labels
            with torch.no_grad():
                output = self.model(data)

            # get predicted classes from model output
            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()
            for p in predicted_class:
                predict.append(p)
            for t in target:
                targets.append(t.item())

        # calculate accuracy score
        acc = accuracy_score(targets, predict)

        cm_true = get_confusion_matrix(targets, predict)
        cat_acc = cm_true.diagonal() / cm_true.sum(axis=1)
        if print_acc: print('Test-Accuracy:', acc, '\nTest-Acc-Class', cat_acc)
        if return_acc:
            self.writer.add_scalar('Test Accuracy',acc,self.epoch)
            return acc

    def predict_test_data(self):
        """Predict test data

        :return: artificial_expert_labels
        """
        predict = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.test_loader):
            data.to(self.device)
            target.to(self.device)
            with torch.no_grad():
                output = self.model(data)

            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(m(output), dim=1).cpu().numpy()
            for p in predicted_class:
                predict.append(int(p))

        return predict

    def load_from_checkpoint(self, mode='best'):
        """Load from checkpoint

        :param mode: Checkpoint to load (best or latest)
        :return: epoch
        """
        cp_dir = self.train_dir + '/checkpoints/checkpoint.' + mode
        try:
            checkpoint = torch.load(cp_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            epoch = checkpoint['epoch']
            print('Found latest checkpoint at', cp_dir)
            print('Continuing in epoch', epoch + 1)
        except:
            epoch = 0
            print('No Checkpoint found')
            print('Starting new from epoch', epoch + 1)

        return epoch

    def save_to_checkpoint(self, epoch, loss, acc):
        with open(self.train_dir + '/logs/exp_log.json', 'r') as f:
            log = json.load(f)


        if acc >= np.max(log['valid_acc']):
            torch.save({'epoch': epoch,
                        'global_step': self.global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.best')

        torch.save({'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.latest')

    def save_model_args(self):
        with open(self.train_dir + 'args/model_args.json', 'w') as f:
            json.dump(self.args, f)


class Resnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(pretrained=True)
        # del self.resnet.fc

        print('load Resnet-18 pretrained on ImageNet')

        # for param in self.resnet.parameters():
        #    param.requires_grad = False

        # self.training = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # print(self.resnet)

    def load_my_state_dict(self, state_dict, strict=True):
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.resnet.load_state_dict(pretrained_dict, strict=strict)

    def forward(self, x, return_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        features = torch.flatten(x, 1)
        if return_features:
            return features
        else:
            out = self.resnet.fc(features)
            # out = nn.Softmax(dim=1)(out)
            return out


class Resnet34(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet34(pretrained=True)
        # del self.resnet.fc

        try:
            print('load Resnet-34 checkpoint')
            self.load_my_state_dict(
                torch.load(
                    os.getcwd()[:-len('Embedding-Semi-Supervised')] + "/nih_images/checkpoint.pretrain"),
                strict=False)
        except FileNotFoundError:
            print('load Resnet-34 pretrained on ImageNet')

        # for param in self.resnet.parameters():
        #    param.requires_grad = False

        # self.training = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # print(self.resnet)

    def load_my_state_dict(self, state_dict, strict=True):
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.resnet.load_state_dict(pretrained_dict, strict=strict)

    def forward(self, x, return_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        features = torch.flatten(x, 1)
        if return_features:
            return features
        else:
            out = self.resnet.fc(features)
            # out = nn.Softmax(dim=1)(out)
            return out