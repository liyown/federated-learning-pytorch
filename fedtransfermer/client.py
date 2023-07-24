
import torch

from abstractclass.client import Client
from utils.utils import AverageMeter, accuracy


class FedBatchClient(Client):
    def __init__(self, clientId, dataPartition, configs):
        """Client object is initiated by the center server."""
        super().__init__(clientId, dataPartition, configs)

    def learningRateDecay(self):
        self.optimizer = eval(self.configs.optimizer)(self.model.parameters(), **self.optimConfig)
        pass
        # TODO: add learning rate decay

    def clientUpdate(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        self.learningRateDecay()
        for e in range(self.localEpoch):
            for data, labels in self.dataLoader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()
                outputs, labels = self.model(data, labels, isTrain=True)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        del self.optimizer
        self.model.to("cpu")
        # 如果是cuda，清理缓存
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def clientEvaluate(self):
        """"""
        pass
