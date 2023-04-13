import matplotlib.pyplot as plt

with open("../result/local_train.txt", mode='r') as f:
    data = f.readlines()

ref_accuracy = []
ref_loss = []
accuracy = []
loss = []
for i in range(10):
    data[i] = data[i].replace(' ', ':').split(':')
    accuracy.append(float(data[i][4]))
    loss.append(float(data[i][6]))

plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(range(len(loss)), loss, 'b', label="Loss")
lns2 = ax2.plot(range(len(accuracy)), accuracy, 'r', label="Accuracy")

ax1.set_xlabel('iteration')
ax1.set_ylabel('testing loss')
ax2.set_ylabel('testing accuracy')

# 合并图例
lns = lns1 + lns2
labels = ["Loss", "Accuracy"]

plt.legend(lns, labels, loc=7)
# plt.savefig(f"{file}-epoch_{filename['epoch']}.png")
plt.show()
