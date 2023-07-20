from abstractclass.server import Server
from models.models import *
from utils.utils import sendMail
from .client import FedBatchClient


class FedBatchServer(Server):
    def __init__(self, dataPartitioner, configs):
        """Initialize the server with the given configurations."""
        super().__init__(dataPartitioner, configs)
        self.testDataloader = self.dataPartitioner.getDataloader(cid=None, batch_size=32, type_="test")
        self.allClients = FedBatchClient.createClients(FedBatchClient, self.dataPartitioner, self.configs)

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
        testLoss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.testDataloader:
                data, labels = data.float().to(self.configs.device), labels.long().to(self.configs.device)
                outputs, labels = self.globalModel(data, labels, isTrain=False)
                testLoss += torch.nn.CrossEntropyLoss()(outputs, labels).item()
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 如果设备是cuda，清理缓存
            if self.configs.device == "cuda":
                torch.cuda.empty_cache()
        self.globalModel.to("cpu")
        testLoss = testLoss / len(self.testDataloader)
        testAccuracy = correct / (len(self.testDataloader) * self.testDataloader.batch_size)
        if printFlag:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(testLoss, testAccuracy))
        return testLoss, testAccuracy
