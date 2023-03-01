import copy
import gc
import sys

from multiprocessing import pool, cpu_count

import torchvision
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .models import *
from .client import Client


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """

    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={},
                 optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        # self.model = eval(model_config["name"])(**model_config)
        resnet18 = torchvision.models.resnet18()
        resnet18.conv1 = nn.Conv2d(3, 64, 3, 1, 1, 1, bias=False)
        resnet18.maxpool = nn.Sequential()
        resnet18.fc[0] = nn.Linear(512, 10)
        self.model = resnet18
        self.seed = global_config["seed"]
        self.device = global_config["device"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config

        """Set up all configuration for federated learning."""
        # initialize weights of the model
        torch.manual_seed(self.seed)
        self.init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message)

        # split local dataset for each client
        # 返回两个dataset，第一个是一个列表，代表本地数据，第二个是测试数据
        local_datasets, test_dataset = self.create_datasets(self.data_path,
                                                            self.dataset_name,
                                                            self.num_clients,
                                                            self.num_shards,
                                                            self.iid)

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets,
                                           batch_size=self.batch_size,
                                           criterion=self.criterion,
                                           num_local_epochs=self.local_epochs,
                                           optimizer=self.optimizer,
                                           optim_config=self.optim_config)

        # prepare hold-out dataset for evaluation
        self.test_data = test_dataset
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False)

        # send the model skeleton to all clients
        self.transmit_model()

    def init_net(self, model, init_type, init_gain):
        """Function for initializing network weights.

        Args:
            model: A torch.nn.Module to be initialized
            init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
            init_gain: Scaling factor for (normal | xavier | orthogonal).
            gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

        Returns:
            An initialized torch.nn.Module instance.
        """

        # if len(gpu_ids) > 0:
        #     assert(torch.cuda.is_available())
        #     model.to(gpu_ids[0])
        #     model = nn.DataParallel(model, gpu_ids)
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
        model.cuda()
        return model

    def create_datasets(self, data_path, dataset_name, num_clients, num_shards, iid):
        """Split the whole dataset in IID or non-IID manner for distributing to clients."""
        dataset_name = dataset_name.upper()
        # get dataset from torchvision.datasets if exists
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()

        else:
            error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
            raise AttributeError(error_message)

        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )

        # 获取所有标签名字
        num_categories = np.unique(training_dataset.targets).shape[0]

        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()

        # split dataset according to iid flag
        if iid:
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets
            ]
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards  # 300
            shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
            shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

            # sort the list to conveniently assign samples to each clients from at least two classes
            shard_inputs_sorted, shard_labels_sorted = [], []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                    shard_inputs_sorted.append(shard_inputs[i + j])
                    shard_labels_sorted.append(shard_labels[i + j])

            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients
            local_datasets = [
                CustomTensorDataset(
                    (
                        torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                    ),
                    transform=transform
                )
                for i in range(0, len(shard_inputs_sorted), shards_per_clients)
            ]
        return local_datasets, test_dataset

    def create_clients(self, local_datasets, **client_config):
        """Initialize each Client instance."""
        clients = []

        message = f"[Round: {str(self._round).zfill(4)}] ...Start created all {str(self.num_clients)} clients!"
        print(message)

        for k, dataset in tqdm(enumerate(local_datasets), file=sys.stdout):
            client = Client(client_id=k, local_data=dataset, device=self.device, **client_config)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message)

        return clients

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            message = f"[Round: {str(self._round).zfill(4)}] ...Start transmitted models to all {str(self.num_clients)} clients!"
            print(message)

            for client in tqdm(self.clients, file=sys.stdout):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message)
        else:
            # send the global model to selected clients
            message = f"[Round: {str(self._round).zfill(4)}] ...Start transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message)

            for idx in tqdm(sampled_client_indices, file=sys.stdout):
                self.clients[idx].model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message)

    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] ...Start updating selected {len(sampled_client_indices)} clients...!"
        print(message)

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, file=sys.stdout):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message)

        return selected_total_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] ...Start Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message)
        model_B = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), file=sys.stdout):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        model_B.load_state_dict(averaged_weights)
        for key in self.model.state_dict().keys():
            averaged_weights[key] = 0.7 * self.model.state_dict()[key] + 0.3 * model_B.state_dict()[key]

        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message)

    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] ...Start Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message)

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message)

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message)

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)],
                                                         size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.test_dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.test_dataloader)
        test_accuracy = correct / len(self.test_data)
        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()

            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {
                    f"[{self.dataset_name}]_{self.model.__class__.__name__} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}_0.1"
                    f"": test_loss},
                self._round
            )
            self.writer.add_scalars(

                'Accuracy',
                {
                    f"[{self.dataset_name}]_{self.model.__class__.__name__} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}_0.1": test_accuracy},
                self._round
            )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
            print(message)

        self.transmit_model()
