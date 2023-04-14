import json

import matplotlib.pyplot as plt

# with open("../result/local_train.txt", mode='r') as f:
#     data = f.readlines()

# for i in range(10):
#     data[i] = data[i].replace(' ', ':').split(':')
#     accuracy.append(float(data[i][4]))
#     loss.append(float(data[i][6]))

import numpy as np

with open("../result/cifar.json") as f:
    data = json.load(f)

cifar1_acc = np.array(data["cifar1_acc"])
cifar1_acc = cifar1_acc[:, 2]

cifar2_acc = np.array(data["cifar2_acc"])
cifar2_acc = cifar2_acc[:, 2]

cifar3_acc = np.array(data["cifar3_acc"])
cifar3_acc = cifar3_acc[:, 2]
fig = plt.figure(figsize=(5, 4), dpi=100)

# 绘制三条曲线
plt.plot(range(100), cifar1_acc, color='red', linewidth=1, linestyle='-', label='Model 1')
plt.plot(range(100), cifar2_acc, color='green', linewidth=1, linestyle='--', label='Model 2')
plt.plot(range(100), cifar3_acc, color='blue', linewidth=1, linestyle=':', label='Model 3')

# 设置坐标轴标签、范围和刻度
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xlim(0, 100)
plt.ylim(0.35, 0.7)
plt.xticks(np.arange(-5, 100, 10), fontsize=10)
plt.yticks(np.arange(0.35, 0.7, 0.05), fontsize=10)

# 添加图例
plt.legend(loc='lower right', fontsize=10)

# 添加标题和作者信息
plt.title('Accuracy Comparison', fontsize=14)

# 调整子图间距
plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.92, wspace=0.2, hspace=0.2)

# 保存图形
# plt.savefig('cvpr_figure.png', dpi=300)

# 显示图形
plt.show()
