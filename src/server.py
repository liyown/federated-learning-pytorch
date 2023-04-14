import threading
import time

import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.client import create_clients
from src.models import MnistCNN, Cifar10CNN
from src.utils import create_datasets, transmit_model, update_selected_clients, average_model, launch_tensor_board

if __name__ == "__main__":
    with open('../config.yaml') as c:
        configs = yaml.load(c, Loader=yaml.FullLoader)

    global_round = 0
    # 日志记录
    logging.basicConfig(filename="../run.log", level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # tensorboard
    writer = SummaryWriter(log_dir=configs["log_config"]["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=(configs["log_config"]["log_path"], configs["log_config"]["tb_port"])
    ).start()
    time.sleep(3.0)

    # resnet18 = torchvision.models.resnet18()
    # resnet18.maxpool = nn.Sequential()
    # resnet18.fc[0] = nn.Linear(512, 10)
    models = Cifar10CNN()
    torch.manual_seed(configs["global_config"]["seed"])
    models = init_net(models, configs["init_config"]["init_type"], configs["init_config"]["init_gain"])

    device = configs["client_config"]["device"]

    # split local dataset for each client
    # 返回两个dataset，第一个是一个列表，代表本地数据，第二个是测试数据
    local_datasets, test_dataset = create_datasets(configs["data_config"]["data_path"],
                                                   configs["data_config"]["dataset_name"],
                                                   configs["fed_config"]["num_clients"],
                                                   configs["data_config"]["iid"])

    # assign dataset to each client
    clients = create_clients(local_datasets)
    # send the model skeleton to all clients
    transmit_model(models, clients)

    # prepare hold-out dataset for evaluation
    test_data = test_dataset
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    """Execute the whole process of the federated learning."""
    results = {"loss": [], "accuracy": []}
    for r in range(configs["fed_config"]["round"]):

        num_sampled_clients = max(int(configs["fed_config"]["fraction"] * configs["fed_config"]["num_clients"]), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(configs["fed_config"]["num_clients"])],
                                                         size=num_sampled_clients, replace=False).tolist())

        select_client = [clients[idx] for idx in sampled_client_indices]

        transmit_model(models, select_client)

        selected_total_size = update_selected_clients(select_client)

        models = average_model(select_client, selected_total_size)

        models.eval()
        models.to(device)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in test_dataloader:
                data, labels = data.float().to(device), labels.long().to(device)
                outputs = models(data)
                test_loss += eval(configs["client_config"]["criterion"])()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if device == "cuda": torch.cuda.empty_cache()
        models.to("cpu")
        test_loss = test_loss / len(test_dataloader)
        test_accuracy = correct / len(test_data)

        results['loss'].append(test_loss)
        results['accuracy'].append(test_accuracy)

        dataset_name = configs["data_config"]["dataset_name"]
        iid = configs["data_config"]["iid"]
        writer.add_scalars(
            'Loss',
            {
                f"[{dataset_name}]_{models.__class__.__name__} ,IID_{iid}": test_loss},
            r
        )
        writer.add_scalars(

            'Accuracy',
            {
                f"[{dataset_name}]_{models.__class__.__name__}, IID_{iid}": test_accuracy},
            r
        )

        if configs["global_config"]["record"]:
            writer.add_scalars(
                'Loss',
                {
                    f"[{dataset_name}]_{models.__class__.__name__} ,IID_{iid}": test_loss},
                r
            )
            writer.add_scalars(

                'Accuracy',
                {
                    f"[{dataset_name}]_{models.__class__.__name__}, IID_{iid}": test_accuracy},
                r
            )
        message = f"[Round: {str(r).zfill(4)}] Evaluate global model's performance...!\
            \n\t[Server] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
        print(message)
        if configs["global_config"]["record"]: logging.info(configs["global_config"]["record_id"] + f"Loss: {test_loss:.4f} Accuracy: {100. * test_accuracy:.2f}")
