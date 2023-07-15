import argparse
import json

import torch
from torchvision.transforms import Normalize, ToTensor, Compose

from dataset.datasets.partitioned_cifar import PartitionCIFAR
from fedavg.server import Server
from utils.utils import draw


def arg_parse():
    parser = argparse.ArgumentParser()

    # global config
    parser.add_argument("--seed", type=int, default="5959")
    parser.add_argument("--recordId", type=str, default="cifar10_dirichlet_100")
    parser.add_argument("--recordPath", type=str, default="../tensorboard/")

    # dataset config
    parser.add_argument("--datasetName", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--balance", type=bool, default="True")
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "iid", "shards"])
    parser.add_argument("--unbalanceSgm", type=int, default=0)
    parser.add_argument("--numShards", type=int, default=200)
    parser.add_argument("--dirAlpha", type=int, default=100)

    # client config
    parser.add_argument("--model", type=str, default="Cifar10CNN", choices=["Cifar10CNN", "MnistCNN", "resnet34"])
    parser.add_argument("--modelConfig", type=dict, default={})
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--localEpochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="torch.optim.SGD", choices=["torch.optim.SGD", "torch.optim.Adam"])
    parser.add_argument("--optimConfig", type=dict, default={"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4})
    # server config
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--numClients", type=int, default=100)
    parser.add_argument("--numGlobalEpochs", type=int, default=2)

    # init_config
    parser.add_argument("--initType", type=str, default="kaiming", choices=["xavier", "kaiming"])
    parser.add_argument("--initSeed", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    configs = arg_parse()
    dataPartitioner = PartitionCIFAR(dataName=configs.datasetName, numClients=configs.numClients, download=True, preprocess=True,
                                     balance=True, partition="iid", unbalance_sgm=0, numShards=None, dirAlpha=None, verbose=True, seed=None,
                                     transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), targetTransform=None)
    draw(dataPartitioner, "./result")
    federatedServer = Server(dataPartitioner, configs)
    results = federatedServer.train()
    with open("result/{0}.json".format(configs.recordId), encoding="utf-8", mode="w") as f:
        json.dump(results, f)



