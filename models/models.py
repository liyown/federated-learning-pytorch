import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose, Normalize, ToTensor


#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        '''输入为3*32*32，尺寸减半是因为池化层'''
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 10))
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=32 * 8 * 8, nhead=4, dim_feedforward=32 * 8 * 8, dropout=0.2)

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
    def __init__(self, backbone=None, feature_dim=32 * 8 * 8):
        super(CnnWithEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.cnnOnlyFeature = eval(backbone)().features
        self.cnnOnlyFeatureWithNoneGrad = copy.deepcopy(self.cnnOnlyFeature)
        self.cnnOnlyClassifier = eval(backbone)().classifier
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=feature_dim, dropout=0.2)

    def batchFormer(self, x, y, isTrain):
        if not isTrain:
            return x, y
        pre_x = x
        x = self.encoderLayer(x.unsqueeze(1)).squeeze(1)
        x = torch.cat([x, pre_x], dim=0)
        y = torch.cat([y, y], dim=0)
        return x, y

    def forward(self, x, y, isTrain=True):
        if not isTrain:
            x = self.cnnOnlyFeature(x)
            x = self.flatten(x)
            # x = self.encoderLayer(x.unsqueeze(1)).squeeze(1)
            x = self.cnnOnlyClassifier(x)
            return x, y
        self.cnnOnlyFeatureWithNoneGrad = copy.deepcopy(self.cnnOnlyFeature)
        for param in self.cnnOnlyFeatureWithNoneGrad.parameters():
            param.requires_grad = False
        x, x_g = self.cnnOnlyFeature(x), self.cnnOnlyFeatureWithNoneGrad(x)
        x, x_g = self.flatten(x), self.flatten(x_g)
        x = torch.cat([x, x_g], dim=0)
        y = torch.cat([y, y], dim=0)
        x, y = self.batchFormer(x, y, isTrain)
        x = self.cnnOnlyClassifier(x)
        return x, y


if __name__ == '__main__':
    models = CnnWithEncoder("Cifar10CNN", 32 * 8 * 8).to("cuda")
    # models = Cifar10CNN().to("cuda")
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    trainDataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    testDataloader = DataLoader(testset, batch_size=32, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.Adam(models.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    result = []
    for e in range(30):
        loss = 0
        acc = 0
        models.train()
        for x_, y_ in trainDataloader:
            x_, y_ = x_.to("cuda"), y_.to("cuda")
            optimizer.zero_grad()
            output = models(x_)
            loss = criterion(output, y_)
            loss.backward()
            optimizer.step()
            acc += (output.argmax(dim=1) == y_).sum().item()
        # 计算一个epoch的训练准确率和损失
        print("train epoch:{},loss:{},acc:{}".format(e, loss, acc / (len(dataset))))
        # 计算一个epoch的测试准确率和损失

        loss = 0
        acc = 0
        with torch.no_grad():
            models.eval()
            for x_, y_ in testDataloader:
                x_, y_ = x_.to("cuda"), y_.to("cuda")
                output = models(x_)
                loss = criterion(output, y_)
                acc += (output.argmax(dim=1) == y_).sum().item()
        print("eval epoch:{},loss:{},acc:{}".format(e, loss, acc / (len(testset))))
        # 保存acc和loss到result,并且写入json文件
        result.append({"epoch": e, "train_loss": loss.item(), "train_acc": acc / (len(testset))})
        with open("result_encoder11.json", "w") as f:
            json.dump(result, f)


