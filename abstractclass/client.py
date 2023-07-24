from abc import ABC, abstractmethod
from collections import Counter
import torch
from models import *


class Client(ABC):
    """Abstract class for all clients."""

    def __init__(self, clientId, dataPartition, configs):
        """Client object is initiated by the center server."""
        self.configs = configs
        self.id = clientId
        self.model = eval(configs.model)(**configs.modelConfig)
        self.dataLoader = dataPartition.getDataloader(cid=self.id, batch_size=configs.batchSize, type_="train")
        self.localEpoch = configs.localEpochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimConfig = configs.optimConfig
        self.device = configs.device
        self.optimizer = eval(self.configs.optimizer)(self.model.parameters(), **self.optimConfig)

    @abstractmethod
    def learningRateDecay(self):
        """Learning rate decay."""

        pass

    @abstractmethod
    def clientUpdate(self):
        """Update local model using local dataset"""
        pass

    @abstractmethod
    def clientEvaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        pass

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataLoader) * self.dataLoader.batch_size

    def dataDistribute(self):
        # 统计每个标签的数量
        counter = Counter()
        for batch in self.dataLoader:
            _, labels = batch
            counter.update(labels.tolist())
        return counter
