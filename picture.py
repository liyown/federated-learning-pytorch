import json

import matplotlib.pyplot as plt

# 从JSON文件中加载数据
with open('fedavg/result/fedavg_cifar10_dirichlet_1.json', 'r') as file:
    data_avg = json.load(file)
    loss_avg = data_avg['loss']
    acc_avg = data_avg['accuracy']

with open('fedtransfermer/result/fusion_cifar10_dirichlet_100.json', 'r') as file:
    data_transfermer = json.load(file)
    loss_transfermer = data_transfermer['loss']
    acc_transfermer = data_transfermer['accuracy']

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_avg, label='FedAvg')
plt.plot(loss_transfermer, label='FedTransfermer')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# 绘制accuracy曲线
plt.figure(figsize=(10, 5))
plt.plot(acc_avg, label='FedAvg')
plt.plot(acc_transfermer, label='FedTransfermer')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
