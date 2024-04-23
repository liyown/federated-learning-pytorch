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
                # mnist out: 32*12*12 fmnist out: 32*12*12  cifar10 out: 32*14*14  cifar100 out: 32*14*14
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                # mnist:  64*4*4  fmnist: 64*4*4  cifar10: 64*5*5  cifar100: 64*5*5
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
        feature_config = {
            "mnist": (64, 4),
            "fmnist": (64, 4),
            "cifar10": (64, 5),
            "cifar100": (64, 5),
        }
        self.flatten = nn.Flatten()
        self.feature = eval(backbone)(**modelConfig).features
        self.featureWithNoneGrad = copy.deepcopy(self.feature)

        self.batchFormer = BatchFormer(feature_chanel=feature_config[modelConfig["dataset"]][0],
                                        feature_dim=feature_config[modelConfig["dataset"]][1])

        self.classifier = eval(backbone)(**modelConfig).classifier

    def setRefer(self):
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        for param in self.featureWithNoneGrad.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.setRefer()
        x, x_g = self.feature(x), self.featureWithNoneGrad(x)
        x = self.batchFormer(x, x_g)
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

        self.single = nn.Parameter(torch.tensor(0.1))

        self.W = nn.Parameter(torch.randn(1, 64, 1, 1))
    def setRefer(self):
        self.featureWithNoneGrad = copy.deepcopy(self.feature)
        for param in self.featureWithNoneGrad.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.setRefer()
        x, x_g = self.feature(x), self.featureWithNoneGrad(x)
        # x = self.single * x + (1 - self.single) * x_g # single
        x = self.W * x + (1 - self.W) * x_g # multi
        # x = self.featureFusion(torch.cat([x, x_g], dim=1)) # conv
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class BatchFormer(nn.Module):
    def __init__(self, feature_chanel=512, feature_dim=24):
        super(BatchFormer, self).__init__()
        self.q = Linear(in_features=feature_chanel*feature_dim*feature_dim, out_features=feature_chanel, bias=False)
        self.k = Linear(in_features=feature_chanel*feature_dim*feature_dim, out_features=feature_chanel, bias=False)

        self.v = nn.Sequential(
            nn.Conv2d(in_channels=feature_chanel, out_channels=feature_chanel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

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
        resdual = x_local

        x_local_embed = self.q(self.fla(x_local))

        x_global_embed = self.k(self.fla(x_global))

        attention = torch.matmul(x_local_embed, x_global_embed.t())

        attention = attention / (1e-9 + torch.sqrt(torch.tensor(self.q.out_features, dtype=torch.float32)))

        attention = self.softmax(attention)

        attention_feature = torch.matmul(attention, x_global.view(x_global.size(0), -1)).view(x_global.size())

        out = self.lambda1 * self.v(attention_feature) + (1 - self.lambda1) * self.v(x_local)

        out = out + resdual

        return out


if __name__ == '__main__':
    pass
