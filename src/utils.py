import copy
import os
import logging
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import yaml

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

#######################
# TensorBaord setting #
#######################
from tqdm import tqdm

from src.client import Client


def launch_tensor_board(log_path, port):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port}")
    return True


def create_datasets(data_path, dataset_name, num_clients, iid=True):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    local_datasets = []
    dataset_name = dataset_name.upper()
    transform = torchvision.transforms.ToTensor()

    # prepare raw training & test datasets
    training_dataset = torchvision.datasets.__dict__[dataset_name](
        root=data_path,
        train=True,
        download=True,
    )
    test_dataset = torchvision.datasets.__dict__[dataset_name](
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

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
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]
    else:
        # sort data by labels
        print()

    return local_datasets, test_dataset


def transmit_model(model, select_client):
    """Send the updated global model to selected/all clients."""
    client_id = []
    for client in tqdm(select_client, file=sys.stdout):
        client.model = copy.deepcopy(model)
        client_id.append(client.id)
    message = f"successfully transmitted models to clients{client_id}!"
    print(message)


def update_selected_clients(select_client):
    """Call "client_update" function of each selected client."""
    selected_total_size = 0
    for client in tqdm(select_client, file=sys.stdout):
        client.client_update()
        selected_total_size += len(client)
    message = f"clients are selected and updated (with total sample size: {str(selected_total_size)})!"
    print(message)

    return selected_total_size


def average_model(select_client, selected_total_size):
    """Average the updated and transmitted parameters from each selected client."""

    mixing_coefficients = [len(client) / selected_total_size for client in select_client]
    global_model = select_client[0].model
    averaged_weights = OrderedDict()
    for it, client in tqdm(enumerate(select_client), file=sys.stdout):
        local_weights = client.model.state_dict()
        for key in client.model.state_dict().keys():
            if it == 0:
                averaged_weights[key] = mixing_coefficients[it] * local_weights[key]
            else:
                averaged_weights[key] += mixing_coefficients[it] * local_weights[key]

    global_model.load_state_dict(averaged_weights)

    message = f"updated weights are successfully averaged!"
    print(message)
    return global_model


#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)