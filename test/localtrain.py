# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import time

import torch
import torchvision
import yaml
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

from model.models import Cifar10CNN

train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
tudui = Cifar10CNN().cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器

learning_rate = 1e-3
optimizer = torch.optim.Adam(list(tudui.parameters()), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
BeginTrainTime = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss / (test_data_size / 64)))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    total_test_step = total_test_step + 1
    TrainTime = time.time() - BeginTrainTime
    with open("../result/local_train.txt", mode='a', encoding="utf-8") as txt:
        txt.write("communicate round " + str(epoch) + " " + "{}".format(configs["global_config"]["record_id"]))
        txt.write('accuracy:' + str(float(total_accuracy / test_data_size)) + " ")
        txt.write('loss: ' + str(float(total_test_loss)) + " ")
        txt.write('time: ' + str(float(TrainTime)) + "\n")

    # torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
