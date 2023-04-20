import gc
import logging
from collections import Counter

import torch
import yaml
from torch.optim.lr_scheduler import StepLR

from method.fedala.ala import ALA

logger = logging.getLogger(__name__)


def create_clients(DataPartition):
    """Initialize each Client instance."""
    clients = []
    for k in range(DataPartition.num_clients):
        client = Client(client_id=k, DataPartition=DataPartition)
        clients.append(client)

    message = f"successfully created all {DataPartition.num_clients} clients!"
    print(message)
    return clients


class Client(object):

    def __init__(self, client_id, DataPartition):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.__model = None
        with open('./config.yaml', encoding="utf-8") as c:
            configs = yaml.load(c, Loader=yaml.FullLoader)

        self.dataloader = DataPartition.get_dataloader(cid=self.id, batch_size=configs["client_config"]["batch_size"],
                                                       type="train")
        self.dataset = DataPartition.get_dataset(cid=self.id, type="train")
        self.local_epoch = configs["client_config"]["num_local_epochs"]
        self.criterion = configs["client_config"]["criterion"]
        self.optimizer = configs["client_config"]["optimizer"]
        self.optim_config = configs["client_config"]["optim_config"]

        self.device = configs["client_config"]["device"]

        self.ALA = ALA(self.id, eval(self.criterion)(), self.dataset, batch_size=32,
                       rand_percent=100, layer_idx=1, eta=1, device=self.device)

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataloader) * self.dataloader.batch_size

    def data_distibute(self):
        # 统计每个标签的数量
        counter = Counter()
        for batch in self.dataloader:
            _, labels = batch
            counter.update(labels.tolist())
        return counter

    def weight_update(self, received_global_model):
        """Update local model using local dataset."""
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9934)
        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True);
        logging.info(message)
        del message;
        gc.collect()
        return test_loss, test_accuracy
