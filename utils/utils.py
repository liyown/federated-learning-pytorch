import copy
import logging
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.init as init
import yaml
from tqdm import tqdm


def launch_tensor_board(log_path, port):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the tensorboard is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port}")
    return True


def transmit_model(model, select_client):
    """Send the updated global model to selected/all clients."""
    client_id = []
    for client in select_client:
        client.model = copy.deepcopy(model)
        client_id.append(client.id)
    message = f"successfully transmitted models to clients{client_id}!"
    print(message)
    with open('../config.yaml', encoding="utf-8") as c:
        configs = yaml.load(c, Loader=yaml.FullLoader)
        if configs["global_config"]["record"]: logging.info(configs["global_config"]["record_id"] + message)


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


def init_net(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    Returns:
        An initialized torch.nn.Module instance.
    """

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


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# def draw(PartitionDataset):
#     PartitionDataset
#     csv_file = "./cifar10_hetero_dir_0.3_100clients.csv"
#
#     partition_report(trainset.targets, hetero_dir_part.client_dict,
#                      class_num=num_classes,
#                      verbose=False, file=csv_file)
#
#     hetero_dir_part_df = pd.read_csv(csv_file, header=1)
#     hetero_dir_part_df = hetero_dir_part_df.set_index('client')
#     col_names = [f"class{i}" for i in range(num_classes)]
#     for col in col_names:
#         hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
#
#     hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
#     plt.tight_layout()
#     plt.xlabel('sample num')
#     plt.savefig(f"./cifar10_hetero_dir_0.3_100clients.png", dpi=400)
