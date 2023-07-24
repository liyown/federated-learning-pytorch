# McMahan et al., 2016; 1,663,370 parameters
import copy
from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn


class FedAvgCNN(nn.Module):
    def __init__(self, dataset: str = "cifar10"):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "fmnist": (1, 1024, 10),
            "cifar10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
        }
        self.features = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                linear1=nn.Linear(config[dataset][1], 512),
                activation1=nn.ReLU(),
                linear2=nn.Linear(512, config[dataset][2]),
            )
        )

        self.dropout = [
            module
            for module in list(self.features.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: torch.Tensor):
        return self.classifier(self.features(x))

    def get_final_features(self, x: torch.Tensor, detach=True) -> torch.Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.features(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)


class CnnWithBatch(nn.Module):
    def __init__(self, backbone=None, feature_dim=1600):
        '''输入为3*32*32，尺寸减半是因为池化层'''
        super(CnnWithBatch, self).__init__()
        self.net = eval(backbone)()
        self.flatten = nn.Flatten()
        self.features = self.net.features
        self.classifier = self.net.classifier
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=2, dim_feedforward=feature_dim,
                                                       dropout=0.2)

    def batchFormer(self, x, y, isTrain):
        if not isTrain:
            return x, y
        pre_x = x
        x = self.encoderLayer(x.unsqueeze(1)).squeeze(1)
        x = torch.cat([x, pre_x], dim=0)
        y = torch.cat([y, y], dim=0)
        return x, y

    def forward(self, x, y, isTrain=True):
        x = self.features(x)
        x = self.flatten(x)
        x, y = self.batchFormer(x, y, isTrain)
        x = self.classifier(x)
        return x, y


class CnnWithEncoder(nn.Module):
    def __init__(self, backbone=None, feature_dim=1600):
        super(CnnWithEncoder, self).__init__()
        self.net = eval(backbone)()
        self.flatten = nn.Flatten()
        self.feature = self.net.features
        self.classifier = self.net.classifier
        self.featureRefer = None
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=feature_dim,
                                                       dropout=1)
        self.setRefer()

    def batchFormer(self, x):
        x = self.encoderLayer(x.unsqueeze(1)).squeeze(1)
        return x

    def setRefer(self):
        self.featureRefer = copy.deepcopy(self.feature)
        for param in self.featureRefer.parameters():
            param.requires_grad = False

    def forward(self, x, y, isTrain=True):
        if not isTrain:
            x = self.feature(x)
            x = self.flatten(x)
            x = self.classifier(x)
            return x, y
        self.setRefer()
        x_l, x_g = self.feature(x), self.featureRefer(x)
        x_l, x_g = self.flatten(x_l), self.flatten(x_g)
        x_l_g = torch.cat([x_l, x_g], dim=0)
        xe_l_g = self.batchFormer(x_l_g)
        xe_l_g_l = torch.cat([xe_l_g, x_l_g], dim=0)
        y = torch.cat([y, y, y, y], dim=0)
        x = self.classifier(xe_l_g_l)
        return x, y


class CnnWithFusion(nn.Module):
    def __init__(self, backbone=None, feature_dim=64 * 6 * 6):
        super(CnnWithFusion, self).__init__()
        self.flatten = nn.Flatten()
        self.feature = eval(backbone)().features
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        self.classifier = eval(backbone)().classifier
        self.featureFusion = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0)

    def setRefer(self):
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        for param in self.featureWithNoneGrad.parameters():
            param.requires_grad = False

    def forward(self, x, y, isTrain=True):
        self.setRefer()
        x, x_g = self.feature(x), self.featureWithNoneGrad(x)
        x = torch.cat([x, x_g], dim=1)
        x = self.featureFusion(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x, y
