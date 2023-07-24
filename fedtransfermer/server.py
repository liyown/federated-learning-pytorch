import torch

from abstractclass.server import Server
from utils.utils import sendMail, AverageMeter, accuracy
from .client import FedBatchClient
from models import *


class FedBatchServer(Server):
    def __init__(self, dataPartitioner, configs):
        """Initialize the server with the given configurations."""
        super().__init__(dataPartitioner, configs)
        self.testDataloader = self.dataPartitioner.getDataloader(cid=None, batch_size=32, type_="test")
        self.createClients(dataPartitioner, configs)

    @sendMail
    def train(self):
        """Train the global model using federated learning."""
        results = {"loss": [], "accuracy": []}
        for epoch in range(self.configs.numGlobalEpochs):
            print(f"Global epoch: {epoch + 1}/{self.configs.numGlobalEpochs}")
            self.selectClient()
            self.transmitModel()
            selectedTotalSize = self.updateSelectedClients()
            self.averageModel(selectedTotalSize)
            testLoss, testAccuracy = self.evaluate()
            results['loss'].append(testLoss)
            results['accuracy'].append(testAccuracy)
        return results

    def evaluate(self, printFlag=True):
        self.globalModel.eval().to(self.configs.device)
        lossAndAcc = AverageMeter("Test set:")
        with torch.no_grad():
            for data, labels in self.testDataloader:
                data, labels = data.float().to(self.configs.device), labels.long().to(self.configs.device)
                outputs, labels = self.globalModel(data, labels, isTrain=False)
                testLoss = torch.nn.CrossEntropyLoss()(outputs, labels).item()
                lossAndAcc.updateLoss(testLoss, data.size(0))
                lossAndAcc.updateAcc(accuracy(outputs, labels)[0], labels.size(0))
            # 如果设备是cuda，清理缓存
            if self.configs.device == "cuda":
                torch.cuda.empty_cache()
        self.globalModel.to("cpu")
        if printFlag:
            print(lossAndAcc)
        return lossAndAcc.lossAvg, lossAndAcc.accAvg

    def createClients(self, dataPartition, configs):
        """Create clients for the server."""
        self.allClients = [FedBatchClient(cid, dataPartition, configs) for cid in range(dataPartition.numClients)]
        print(f"Total {dataPartition.numClients} clients created.")
