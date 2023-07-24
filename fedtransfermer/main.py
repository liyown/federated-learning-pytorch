import argparse
import json

from torchvision.transforms import Normalize, ToTensor, Compose, RandomHorizontalFlip, CenterCrop, ColorJitter

from dataset.datasets.partitioned_cifar import PartitionCIFAR
from fedtransfermer.server import FedBatchServer
from utils.utils import draw, setupSeed


def arg_parse():
    parser = argparse.ArgumentParser()

    # global config
    parser.add_argument("--seed", type=int, default="0925")
    parser.add_argument("--recordId", type=str, default="encoder_cifar10_dirichlet_1")

    # dataset config
    """
    balance = None, partition = dirichlet, dirAlpha = 需要的参数         //  一般的dirichlet
    balance = True, partition = dirichlet, dirAlpha = 需要的参数         //  平衡的dirichlet
    balance = False, partition = dirichlet, dirAlpha = 需要的参数        //  不平衡的dirichlet
    
    balance = None, partition = shards, numShards = 需要的参数(200)      //  一般的shards
    
    balance = True, partition = iid                                     //  平衡的iid
    balance = False, partition = iid                                    //  不平衡的iid
    """
    parser.add_argument("--datasetName", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--balance", type=bool, default="True")
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "iid", "shards"])
    parser.add_argument("--unbalanceSgm", type=int, default=None)
    parser.add_argument("--numShards", type=int, default=None)
    parser.add_argument("--dirAlpha", type=int, default=0.1)

    # client config
    parser.add_argument("--model", type=str, default="CnnWithEncoder",
                        choices=["CnnWithEncoder", "CnnWithFusion", "CnnWithBatch"])
    parser.add_argument("--modelConfig", type=dict, default={"backbone": "FedAvgCNN", "feature_dim": 1600})
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--localEpochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="torch.optim.Adam",
                        choices=["torch.optim.SGD", "torch.optim.Adam"])
    parser.add_argument("--optimConfig", type=dict, default={"lr": 0.0003})
    # server config
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--numClients", type=int, default=100)
    parser.add_argument("--numGlobalEpochs", type=int, default=150)

    args = parser.parse_args()
    return args


def run():
    configs = arg_parse()
    # transform consists of cropping the images to 24x24, randomly flipping left-right and adjusting the contrast, brightness and whitening.
    transform = Compose(
        [RandomHorizontalFlip(), ColorJitter(0.5, 0.5), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    setupSeed(configs.seed)
    dataPartitioner = PartitionCIFAR(dataName=configs.datasetName, numClients=configs.numClients, download=True,
                                     path="./clientdata/cifar10", preprocess=False,
                                     balance=configs.balance, partition=configs.partition,
                                     unbalance_sgm=configs.unbalanceSgm, numShards=configs.numShards,
                                     dirAlpha=configs.dirAlpha, verbose=True, seed=120,
                                     transform=transform, targetTransform=None)
    # draw(dataPartitioner, "./result")
    print("transformer")
    federatedServer = FedBatchServer(dataPartitioner, configs)
    results = federatedServer.train()
    with open("result/{0}.json".format(configs.recordId), encoding="utf-8", mode="w") as f:
        json.dump(results, f)


def dataProcess():
    configs = arg_parse()
    transform = Compose(
        [RandomHorizontalFlip(), ColorJitter(0.5, 0.5), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    setupSeed(configs.seed)
    dataPartitioner = PartitionCIFAR(dataName=configs.datasetName, numClients=configs.numClients,
                                     path="./clientdata/cifar10", download=True, preprocess=True,
                                     balance=configs.balance, partition=configs.partition,
                                     unbalance_sgm=configs.unbalanceSgm, numShards=configs.numShards,
                                     dirAlpha=configs.dirAlpha, verbose=True, seed=260,
                                     transform=transform, targetTransform=None)
    draw(dataPartitioner, "./clientdata/cifar10")


if __name__ == '__main__':
    # dataProcess()
    run()
