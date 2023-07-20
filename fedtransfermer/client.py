import torch

from abstractclass.client import Client
from utils.utils import AverageMeter, accuracy


class FedBatchClient(Client):
    def __init__(self, clientId, dataPartition, configs):
        """Client object is initiated by the center server."""
        super().__init__(clientId, dataPartition, configs)

    def learningRateDecay(self):
        pass
        # TODO: add learning rate decay

    def clientUpdate(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        # self.learningRateDecay()
        for e in range(self.localEpoch):
            for data, labels in self.dataLoader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()
                outputs, labels = self.model(data, labels, isTrain=True)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                del data, labels, outputs, loss
        # 如果是cuda，清理缓存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.model.to("cpu")

    def clientEvaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        lossAndAcc = AverageMeter(f"Client {str(self.id).zfill(4)}] evaluation")
        with torch.no_grad():
            for data, labels in self.dataLoader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                testLoss = self.criterion(outputs, labels).item()
                lossAndAcc.updateLoss(testLoss, data.size(0))
                lossAndAcc.updateAcc(accuracy(outputs, labels)[0], labels.size(0))
        self.model.to("cpu")
        print(lossAndAcc)
        return lossAndAcc.lossAvg, lossAndAcc.accAvg
