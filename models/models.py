# McMahan et al., 2016; 1,663,370 parameters
import copy
from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Linear


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
                # mnist out: 32*12*12 fmnist out: 32*12*12  cifar10 out: 32*28*28  cifar100 out: 32*28*28
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                # mnist:  64*4*4  fmnist: 64*4*4  cifar10: 64*6*6  cifar100: 64*6*6
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

    def forward(self, x: torch.Tensor):
        return self.classifier(self.features(x))

class CnnWithBatchFormer(nn.Module):
    def __init__(self, backbone=None, modelConfig=None):
        super(CnnWithBatchFormer, self).__init__()
        self.flatten = nn.Flatten()
        self.feature = eval(backbone)(**modelConfig).features
        self.featureWithNoneGrad = copy.deepcopy(self.feature)

        self.batchFormer = BatchFormer(64, 4)

        self.classifier = eval(backbone)(**modelConfig).classifier

    def setRefer(self):
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        for param in self.featureWithNoneGrad.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.setRefer()
        x, x_g = self.feature(x), self.featureWithNoneGrad(x)
        x = self.batchFormer(x, x_g)
        # x = self.feature(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class CnnWithFusion(nn.Module):
    def __init__(self, backbone=None, modelConfig=None):
        super(CnnWithFusion, self).__init__()
        self.flatten = nn.Flatten()
        self.feature = eval(backbone)(**modelConfig).features
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        self.classifier = eval(backbone)(**modelConfig).classifier
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


class BatchFormer(nn.Module):
    def __init__(self, feature_chanel=512, feature_dim=24):
        super(BatchFormer, self).__init__()
        self.q = Linear(in_features=feature_chanel, out_features=feature_chanel, bias=False)
        self.k = Linear(in_features=feature_chanel, out_features=feature_chanel, bias=False)

        self.v = nn.Conv2d(in_channels=feature_chanel, out_channels=feature_chanel, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=feature_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.lambda1 = nn.Parameter(torch.tensor(0.1))

        self.fla = nn.Flatten()

    def forward(self, x_local, x_global):
        """
        batch attention implement
        Args:
            x_local: size(batch_size, feature_chanel, feature_dim, feature_dim)
            x_global:  size(batch_size, feature_chanel, feature_dim, feature_dim)
        Returns: size(batch_size, feature_chanel, feature_dim, feature_dim)
        """
        x_local_embed = self.fla(x_local)

        x_global_embed = self.fla(x_global)

        attention = torch.matmul(x_local_embed, x_global_embed.t())

        attention = attention / (1e-9 + torch.sqrt(torch.tensor(x_local_embed.size(-1), dtype=torch.float32)))

        attention = self.softmax(attention)

        attention_feature = torch.matmul(attention, x_global.view(x_global.size(0), -1)).view(x_global.size())

        out = self.lambda1 * self.v(attention_feature) + (1 - self.lambda1) * self.v(x_local)


        return out

if __name__ == '__main__':
    pass

