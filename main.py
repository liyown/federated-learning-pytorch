import os
import time
import datetime
import pickle
import yaml
import threading
import logging

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server

def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True


if __name__ == "__main__":
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=(log_config["log_path"], log_config["tb_port"], log_config["tb_host"])
    ).start()
    time.sleep(3.0)

    # initialize federated learning 
    central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)

    # do federate learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
