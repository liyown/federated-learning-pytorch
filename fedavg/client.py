import torch
from abstractclass.client import Client
from models.models import *


class FedavgClient(Client):
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
        for e in range(self.localEpoch):
            for data, labels in self.dataLoader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()
                outputs, labels = self.model(data, labels, isTrain=True)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        # 如果是cuda，清理缓存
        if self.device == "cuda":
            torch.cuda.empty_cache()
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

