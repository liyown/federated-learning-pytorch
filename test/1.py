#encoding = utf-8
# 创建数据
import numpy as np
from matplotlib import pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y1 = np.array([2, 3, 4, 3, 2])
y2 = np.array([1, 4, 2, 5, 3])
y3 = np.array([3, 2, 1, 4, 5])

# 绘制堆积图
plt.bar(x, y1, label='A')
plt.bar(x, y2, bottom=y1, label='B')
plt.bar(x, y3, bottom=y1+y2, label='C')

# 添加图例和标签
plt.legend()
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('柱状堆积图')

# 显示图形
plt.show()