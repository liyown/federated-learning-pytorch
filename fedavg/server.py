import copy
import sys
from collections import OrderedDict

import numpy as np
from torch.nn import init
from tqdm import tqdm

from client import Client
from models.models import Cifar10CNN
from utils.utils import Evaluation


class Server:
    def __init__(self, dataPartitioner, configs):
        """Initialize the server with the given configurations."""
        self.dataPartitioner = dataPartitioner
        self.configs = configs
        self.globalModel = self.initModelWeights(eval(self.configs.model)(**self.configs.modelConfig),
                                                 init_type=self.configs.initType,
                                                 init_gain=self.configs.initSeed)
        self.clients = Client.createClients(self.dataPartitioner, self.configs)
        self.evaluate = Evaluation()

    def train(self):
        """Train the global model using federated learning."""
        results = {"loss": [], "accuracy": []}
        for epoch in range(self.configs.numGlobalEpochs):
            print(f"Global epoch: {epoch + 1}/{self.configs.numGlobalEpochs}")
            selectClients = self.selectClients()
            self.transmitModel(selectClients)
            selectedTotalSize = self.updateSelectedClients(selectClients)
            self.averageModel(selectClients, selectedTotalSize)

            self.evaluate.model, self.evaluate.testDataloader = self.globalModel, self.dataPartitioner.get_dataloader(cid=None, batch_size=self.configs.batchSize, type_="test")
            testLoss, testAccuracy = self.evaluate.evaluate()
            results['loss'].append(testLoss)
            results['accuracy'].append(testAccuracy)
        return results

    def selectClients(self):
        """Select clients to participate in the current global training round."""
        numSampledClients = max(int(self.configs.fraction * self.configs.numClients), 1)
        sampledClientIndices = sorted(
            np.random.choice(a=[i for i in range(self.configs.numClients)], size=numSampledClients,
                             replace=False).tolist())
        return [self.clients[idx] for idx in sampledClientIndices]

    def transmitModel(self, selectClient):
        """Send the updated global model to selected/all clients."""
        for client in selectClient:
            client.model = copy.deepcopy(self.globalModel)

    def updateSelectedClients(self, selectClients):
        """Call "client_update" function of each selected client."""
        selectedTotalSize = 0
        for client in tqdm(selectClients, file=sys.stdout):
            client.clientUpdate()
            # client.clientEvaluate()
            selectedTotalSize += len(client)
        return selectedTotalSize

    def averageModel(self, selectClients, selectedTotalSize):
        """Average the updated and transmitted parameters from each selected client."""
        mixingCoefficients = [len(client) / selectedTotalSize for client in selectClients]
        averagedWeights = OrderedDict()
        for it, client in enumerate(selectClients):
            localWeights = client.model.state_dict()
            for key in client.model.state_dict().keys():
                if it == 0:
                    averagedWeights[key] = mixingCoefficients[it] * localWeights[key]
                else:
                    averagedWeights[key] += mixingCoefficients[it] * localWeights[key]
        self.globalModel.load_state_dict(averagedWeights)

    def initModelWeights(self, model, init_type='normal', init_gain=0.02):
        """Function for initializing network weights."""

        def init_func(m):
            class_name = m.__class__.__name__
            if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                else:
                    raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

            elif class_name.find('BatchNorm2d') != -1 or class_name.find('InstanceNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        model.apply(init_func)
        return model
