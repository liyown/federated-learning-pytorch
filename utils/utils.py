import os
import random

import numpy as np
import pandas as pd
import torch
import yagmail
from matplotlib import pyplot as plt

from dataset.datasets.partitioned_cifar import PartitionCIFAR
from dataset.utils.functional import partition_report


def launch_tensor_board(log_path, port):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the tensorboard is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port}")
    return True


def seed_torch(seed=3027):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def draw(PartitionDataset: PartitionCIFAR, root):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    csv_dir = os.path.join(root,
                           f"{PartitionDataset.dataName}_{PartitionDataset.numClients}_{PartitionDataset.partitioner.partition}_{PartitionDataset.partitioner.balance}.csv")
    partition_report(PartitionDataset.trainDatasets.targets, PartitionDataset.partitioner.client_dict,
                     class_num=np.unique(PartitionDataset.trainDatasets.targets).shape[0],
                     verbose=False, file=csv_dir)

    hetero_dir_part_df = pd.read_csv(csv_dir, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(np.unique(PartitionDataset.trainDatasets.targets).shape[0])]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
    plt_dir = os.path.join(root,
                           f"{PartitionDataset.dataName}_{PartitionDataset.numClients}_{PartitionDataset.partitioner.partition}_{PartitionDataset.partitioner.balance}.png")
    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(plt_dir, dpi=400)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.lossVal = 0
        self.lossAvg = 0
        self.lossSum = 0
        self.lossCount = 0

        self.accVal = 0
        self.accAvg = 0
        self.accSum = 0
        self.accCount = 0

    def updateLoss(self, val, n=1):
        self.lossVal = val
        self.lossSum += val * n
        self.lossCount += n
        self.lossAvg = self.lossSum / self.lossCount

    def updateAcc(self, val, n=1):
        self.accVal = val
        self.accSum += val * n
        self.accCount += n
        self.accAvg = self.accSum / self.accCount

    def __str__(self):
        """打印平均lossAvg和accAvg"""
        fmtstr = '\n{name}\tloss {lossAvg:.4f} \t acc {accAvg:.2f}%\n'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def sendMail(func):
    """Send the results to the specified mailbox."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        contents = [f"Test Loss: {result['loss'][-1]}", f"Test Accuracy: {result['accuracy'][-1]}"]
        yagmail.SMTP(user="liuyaowen_smile@126.com", password='BWLALEHLTNVNWLHX', host='smtp.126.com').send(
            to='1536727925@qq.com', subject='Send', contents=contents)
        print("Mail Send successfully!")
        return result

    return wrapper


def setupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    label = torch.tensor([2, 5, 8, 9])
    pre = torch.tensor([[0.7082, 0.5335, 0.9494, 0.7792, 0.3288, 0.6303, 0.0335, 0.6918, 0.0778,
                         0.6404],
                        [0.3881, 0.8676, 0.7700, 0.6266, 0.8843, 0.8902, 0.4336, 0.5385, 0.8372,
                         0.1204],
                        [0.9717, 0.2727, 0.9086, 0.7797, 0.1216, 0.4793, 0.1149, 0.1544, 0.7292,
                         0.0459],
                        [0.0424, 0.0809, 0.1597, 0.4177, 0.4798, 0.7107, 0.9683, 0.7502, 0.1536,
                         0.3994]])

    c = accuracy(pre, label, [1, 2, 3, 4])
    print(c)
