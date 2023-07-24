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

from .basic_dataset import FedDataset, Subset
from dataset.utils.partition import MNISTPartitioner


class PartitionedMNIST(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        partition (str, optional): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 num_clients,
                 root="../data",
                 path="./clientdata",
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targetTransform = target_transform

        self.trainDatasets = torchvision.datasets.MNIST(root=self.root,
                                                        train=True,
                                                        download=download)
        self.testDatasets = torchvision.datasets.MNIST(root=self.root,
                                                       train=False, transform=self.transform,
                                                       target_transform=self.targetTransform,
                                                       download=download)

        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            # os.mkdir(os.path.join(self.path, "var"))
            # os.mkdir(os.path.join(self.path, "test"))

        partitioner = MNISTPartitioner(self.trainDatasets.targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       verbose=verbose,
                                       seed=seed)

        # partition
        subsets = {
            cid: Subset(self.trainDatasets.targets,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def getDataset(self, cid, type_="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

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

    def getDataloader(self, cid=None, batch_size=None, type_="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        data_loader = None
        # 返回整体训练与测试集
        if cid is None:
            if type_ == "test":
                return DataLoader(self.testDatasets,
                                  batch_size=len(self.testDatasets) if batch_size is None else batch_size)
            elif type_ == "train":
                return DataLoader(self.trainDatasets, batch_size=batch_size, shuffle=True)
        if type_ == "train":
            dataset = self.getDataset(cid, type_)
            batch_size = len(dataset) if batch_size is None else batch_size
            data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        elif type_ == "test":
            # todo 每个客户端的测试集
            pass
        else:
            pass
            # TODO 验证集

        return data_loader
