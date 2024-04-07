import argparse
import json
import os

from torchvision.transforms import ToTensor
from dataset.datasets.partitioned_mnist import PartitionedMNIST
from fedavg.server import FedavgServer
from utils.utils import draw, setupSeed, find_project_root


def arg_parse():
    parser = argparse.ArgumentParser()

    # global config
    parser.add_argument("--seed", type=int, default=925)
    parser.add_argument("--recordId", type=str, default="mnist_noniid-2label_1")

    # dataset config
    parser.add_argument("--datasetName", type=str, default="mnist", choices=["mnist"])
    parser.add_argument("--partition", type=str, default="noniid-#label",
                        choices=["noniid-#labe", "noniid-labeldir", "unbalance", "iid"])
    parser.add_argument("--majorClassesNum", type=int, default=2)
    parser.add_argument("--dirAlpha", type=int, default=None)

    # client config
    parser.add_argument("--model", type=str, default="FedAvgCNN")
    parser.add_argument("--modelConfig", type=dict, default={"dataset": "mnist"})
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--localEpochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="torch.optim.Adam",
                        choices=["torch.optim.SGD", "torch.optim.Adam"])
    parser.add_argument("--optimConfig", type=dict, default={"lr": 0.0003, "weight_decay": 0.0001, "betas": (0.9, 0.999)})
    # server config
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--numClients", type=int, default=100)
    parser.add_argument("--numGlobalEpochs", type=int, default=50)

    args = parser.parse_args()
    return args


def run(configs):
    dataPartitioner = PartitionedMNIST(numClients=configs.numClients, dataName=configs.datasetName,
                                       path="./clientdata/mnsit", download=True,
                                       partition=configs.partition, majorClassesNum=configs.majorClassesNum,
                                       dirAlpha=configs.dirAlpha, verbose=True, seed=260,
                                       transform=ToTensor(), targetTransform=None)

    if not os.path.exists("./clientdata/mnsit"):
        draw(dataPartitioner, "./clientdata/mnsit")

    print("avg")
    federatedServer = FedavgServer(dataPartitioner, configs)

    results = federatedServer.train()

    with open("result/{0}.json".format(configs.recordId), encoding="utf-8", mode="w") as f:
        json.dump(results, f)



if __name__ == '__main__':
    configs = arg_parse()
    # dataPrecess()
    for i in range(10):
        configs.recordId = "mnist_noniid-2label_{0}".format(i)
        run(configs)
    #     在任何文件获得项目的根目录



