import copy
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.init as init
import yagmail
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from tqdm import tqdm

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


class Evaluation(object):
    """Evaluate the performance of the model on the test dataset."""

    def __init__(self, model=None, testDataloader=None, device="cuda"):
        self.model = model
        self.testDataloader = testDataloader
        self.device = device

    def evaluate(self, printFlag=True):
        self.model.eval()
        self.model.to(self.device)
        testLoss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.testDataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                if self.model.__class__.__name__ == "CnnWithEncoder":
                    outputs, labels = self.model(data, labels, isTrain=False)
                    testLoss += torch.nn.CrossEntropyLoss()(outputs, labels).item()
                else:
                    outputs, labels = self.model(data, labels, isTrain=False)
                    testLoss += torch.nn.CrossEntropyLoss()(outputs, labels).item()
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 如果设备是cuda，清理缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        testLoss = testLoss / len(self.testDataloader)
        testAccuracy = correct / (len(self.testDataloader) * self.testDataloader.batch_size)
        if printFlag:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(testLoss, testAccuracy))
        return testLoss, testAccuracy


def sendMail(func):
    """Send the results to the specified mailbox."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        contents = [f"Test Loss: {result['loss'][-1]}", f"Test Accuracy: {result['accuracy'][-1]}"]
        yagmail.SMTP(user="liuyaowen_smile@126.com", password='BWLALEHLTNVNWLHX', host='smtp.126.com').send(to='1536727925@qq.com', subject='Send', contents=contents)
        print("Mail Send successfully!")
        return result
    return wrapper

