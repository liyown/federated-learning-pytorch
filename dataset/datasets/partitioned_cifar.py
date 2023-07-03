# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torchvision
from torch.utils.data import DataLoader

from .basic_dataset import FedDataset, CIFARSubset
from ..partition import CIFAR10Partitioner, CIFAR100Partitioner


class PartitionCIFAR(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        dataName (str): "cifar10" or "cifar100"
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
        partition (str, optional): Partition type, only ``"iid"``, ``shards``, ``"dirichlet"`` are supported. Default as ``"iid"``.
        unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
        num_shards (int, optional): Number of shards in non-iid ``"shards"`` partition. Only works if ``partition="shards"``. Default as ``None``.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        targetTransform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 root,
                 path,
                 dataName,
                 num_clients,
                 download=True,
                 preprocess=False,
                 balance=True,
                 partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 targetTransform=None) -> None:
        self.test_datasets = None
        self.train_datasets = None
        self.partitioner = None
        self.dataName = dataName
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targetTransform = targetTransform

        if preprocess:
            self.preprocess(balance=balance,
                            partition=partition,
                            unbalance_sgm=unbalance_sgm,
                            num_shards=num_shards,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download)

    def preprocess(self,
                   balance=True,
                   partition="iid",
                   unbalance_sgm=0,
                   num_shards=None,
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            # os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
        # train dataset partitioning
        if self.dataName == 'cifar10':
            trainDataset = torchvision.datasets.CIFAR10(root=self.root,
                                                        train=True,
                                                        download=download)

            testDataset = torchvision.datasets.CIFAR10(root=self.root,
                                                       train=False, transform=self.transform,
                                                       target_transform=self.targetTransform,
                                                       download=download)
            self.partitioner = CIFAR10Partitioner(trainDataset.targets,
                                                  self.num_clients,
                                                  balance=balance,
                                                  partition=partition,
                                                  unbalance_sgm=unbalance_sgm,
                                                  num_shards=num_shards,
                                                  dir_alpha=dir_alpha,
                                                  verbose=verbose,
                                                  seed=seed)
            print()

        elif self.dataName == 'cifar100':
            trainDataset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=download)
            testDataset = torchvision.datasets.CIFAR100(root=self.root,
                                                        train=False, transform=self.transform,
                                                        target_transform=self.targetTransform,
                                                        download=download)
            self.partitioner = CIFAR100Partitioner(trainDataset.targets,
                                                   self.num_clients,
                                                   balance=balance,
                                                   partition=partition,
                                                   unbalance_sgm=unbalance_sgm,
                                                   num_shards=num_shards,
                                                   dir_alpha=dir_alpha,
                                                   verbose=verbose,
                                                   seed=seed)
        else:
            raise ValueError(
                f"'dataName'={self.dataName} currently is not supported. Only 'cifar10', and 'cifar100' are supported."
            )

        self.train_datasets = trainDataset
        self.test_datasets = testDataset

        subsets = {
            cid: CIFARSubset(trainDataset,
                             self.partitioner.client_dict[cid],
                             transform=self.transform,
                             target_transform=self.targetTransform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def get_dataset(self, cid, type_="train"):
        """Load sub-dataset for client with client ID ``cid`` from local file.
        Args:
             cid (int): client id
             type_ (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        Returns:
            Dataset
        """
        dataset = None
        if type_ == "train":
            dataset = torch.load(os.path.join(self.path, type_, "data{}.pkl".format(cid)))
        elif type_ == "test":
            # todo 每个客户端的测试集
            pass
        else:
            pass
            # TODO 验证集

        return dataset

    def get_dataloader(self, cid=None, batch_size=None, type_="train"):
        """Return dataload for client with client ID ``cid``.
        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type_ (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        data_loader = None
        # 返回整体训练与测试集
        if cid is None:
            if type_ == "test":
                return DataLoader(self.test_datasets, batch_size=batch_size)
            elif type_ == "train":
                return DataLoader(self.test_datasets, batch_size=batch_size, shuffle=True)
        if type_ == "train":
            dataset = self.get_dataset(cid, type_)
            batch_size = len(dataset) if batch_size is None else batch_size
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        elif type_ == "test":
            # todo 每个客户端的测试集
            pass
        else:
            pass
            # TODO 验证集

        return data_loader
