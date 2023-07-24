from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from torch.nn import init
from tqdm import tqdm
from models import *


class Server(ABC):
    """Abstract class for the server."""

    def __init__(self, dataPartitioner, configs):
        """Initialize the server with the given configurations."""
        self.dataPartitioner = dataPartitioner
        self.configs = configs
        self.globalModel = eval(self.configs.model)(**self.configs.modelConfig)
        self.selectClients = None
        self.allClients = None

    @abstractmethod
    def train(self):
        """Train the global model using federated learning."""
        """
            1. Select clients to participate in the current global training round.
            2. Send the updated global model to selected/all clients.
            3. Call "clientUpdate" function of each selected client.
            4. Average the updated and transmitted parameters from each selected client.
            5. Evaluate the global model.
        """
        pass

    @abstractmethod
    def evaluate(self, printFlag=True):
        """Evaluate the global model."""
        """
            由于模型的定义不同，无法在抽象类中定义evaluate函数，因此在子类中定义
        """
        pass

    @abstractmethod
    def createClients(self, dataPartition, configs):
        """Initialize each Client instance."""
        pass

    def averageModel(self, selectedTotalSize):
        """Average the updated and transmitted parameters from each selected client."""
        mixingCoefficients = [len(client) / selectedTotalSize for client in self.selectClients]
        averagedWeights = OrderedDict()
        for it, client in enumerate(self.selectClients):
            localWeights = client.model.state_dict()
            for key in client.model.state_dict().keys():
                if it == 0:
                    averagedWeights[key] = mixingCoefficients[it] * localWeights[key]
                else:
                    averagedWeights[key] += mixingCoefficients[it] * localWeights[key]
        self.globalModel.load_state_dict(averagedWeights)

    def updateSelectedClients(self):
        """Call "client_update" function of each selected client."""
        selectedTotalSize = 0
        for client in tqdm(self.selectClients, desc="Client update"):
            client.clientUpdate()
            selectedTotalSize += len(client)
        return selectedTotalSize

    def selectClient(self):
        """Select clients to participate in the current global training round."""
        numSampledClients = max(int(self.configs.fraction * self.configs.numClients), 1)
        sampledClientIndices = sorted(
            np.random.choice(a=[i for i in range(self.configs.numClients)], size=numSampledClients,
                             replace=False).tolist())
        self.selectClients = [self.allClients[i] for i in sampledClientIndices]

    def transmitModel(self):
        """Send the updated global model to selected/all clients."""
        for client in self.selectClients:
            client.model.load_state_dict(self.globalModel.state_dict())

