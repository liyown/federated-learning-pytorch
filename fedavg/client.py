import copy
from collections import Counter
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp

from models.models import Cifar10CNN
from models.ResNet import resnet18



class Client(object):

    def __init__(self, clientId, DataPartition, configs):
        """Client object is initiated by the center server."""
        self.optimizer = None
        self.configs = configs
        self.id = clientId
        self.model = eval(configs.model)(**configs.modelConfig)
        self.dataLoader = DataPartition.getDataloader(cid=self.id, batch_size=configs.batchSize, type_="train")
        self.localEpoch = configs.localEpochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimConfig = configs.optimConfig
        self.device = configs.device
        self.scaler = amp.GradScaler()

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
                # with autocast():
                #     outputs = self.model(data)
                #     loss = self.criterion(outputs, labels)
                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        self.model.to("cpu")

    def clientEvaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataLoader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += self.criterion(outputs, labels).item()
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
        self.model.to("cpu")
        test_loss = test_loss / len(self.dataLoader)
        test_accuracy = correct / (len(self.dataLoader) * self.dataLoader.batch_size)
        message = f"\t[Client {str(self.id).zfill(4)}] evaluation!\t=> Test loss: {test_loss:.4f}\t=> Test accuracy: {100. * test_accuracy:.2f}%"
        print(message)
        return test_loss, test_accuracy

    @staticmethod
    def createClients(DataPartition, configs):
        """Initialize each Client instance."""
        clients = []
        for k in range(DataPartition.numClients):
            client = Client(clientId=k, DataPartition=DataPartition, configs=configs)
            clients.append(client)

        print(f"successfully created all {DataPartition.numClients} clients!")
        return clients
