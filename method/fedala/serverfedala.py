import json
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets.partitioned_cifar import PartitionCIFAR
from method.fedala.ala import update_selected_clients_weights
from method.fedala.clientfedala import create_clients
from model.models import MnistCNN, CNN2
from utils.utils import transmit_model, update_selected_clients, average_model, init_net, seed_torch

if __name__ == "__main__":
    with open('./config.yaml', encoding="utf-8") as c:
        configs = yaml.load(c, Loader=yaml.FullLoader)

    print(
        "\n\nid:{}dataset_name:{}--model:{}--optimizer:{}--lr:{}--num_clients:{}--fraction:{}--num_local_epochs:{}--batch_size:{}--record:{}\n\n".format(
            configs["global_config"]["record_id"],
            configs["data_config"]["dataset_name"],
            configs["client_config"]["model"],
            configs["client_config"]["optimizer"],
            configs["client_config"]["optim_config"]["lr"],
            configs["fed_config"]["num_clients"],
            configs["fed_config"]["fraction"],
            configs["client_config"]["num_local_epochs"],
            configs["client_config"]["batch_size"],
            configs["global_config"]["record"]
        ))
    # tensorboard
    writer = SummaryWriter(log_dir=configs["log_config"]["log_path"], filename_suffix="FL")
    time.sleep(2)

    # 修改模型的地方
    if configs["client_config"]["model"] == "Cifar10CNN":
        models = CNN2(in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10)
    elif configs["client_config"]["model"] == "MnistCNN":
        models = MnistCNN()

    seed_torch()
    models = init_net(models, configs["init_config"]["init_type"], configs["init_config"]["init_gain"])

    device = configs["client_config"]["device"]

    # 创建分割数据集及其类
    PartitionCifar10 = PartitionCIFAR("../data", "data", "cifar10",
                                      configs["fed_config"]["num_clients"],
                                      download=True, preprocess=True,
                                      balance=True, partition="iid",
                                      unbalance_sgm=0, num_shards=None,
                                      dir_alpha=None, transform=torchvision.transforms.ToTensor(),
                                      target_transform=None)

    # assign dataset to each client
    clients = create_clients(PartitionCifar10)

    # send the model skeleton to all clients
    transmit_model(models, clients)

    # prepare hold-out dataset for evaluation
    test_dataloader = PartitionCifar10.get_dataloader(cid=0, batch_size=64, type="test")

    """Execute the whole process of the federated learning."""
    results = {"loss": [], "accuracy": []}
    for r in range(configs["fed_config"]["round"]):
        # 随机采样，获取客户端id
        num_sampled_clients = max(int(configs["fed_config"]["fraction"] * configs["fed_config"]["num_clients"]), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(configs["fed_config"]["num_clients"])],
                                                         size=num_sampled_clients, replace=False).tolist())
        # 获取随机采样的客户端
        select_client = [clients[idx] for idx in sampled_client_indices]
        # 将全局模型传递给选中的客户端
        # transmit_model(models, select_client)
        update_selected_clients_weights(select_client, models)

        # 更新选中的客户端，并且返回选中客户端所有的数据量
        selected_total_size = update_selected_clients(select_client)
        # 进行模型聚合
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
        test_accuracy = correct / (len(test_dataloader) * test_dataloader.batch_size)

        results['loss'].append(test_loss)
        results['accuracy'].append(test_accuracy)

        dataset_name = configs["data_config"]["dataset_name"]
        iid = configs["data_config"]["iid"]
        record_id = configs["global_config"]["record_id"]
        if configs["global_config"]["record"]:
            writer.add_scalars(
                'Loss',
                {
                    record_id + f"{dataset_name}_{models.__class__.__name__} ,IID_{iid}": test_loss},
                r
            )
            writer.add_scalars(

                'Accuracy',
                {
                    record_id + f"{dataset_name}]_{models.__class__.__name__}, IID_{iid}": test_accuracy},
                r
            )
        message = record_id + f"[Round: {str(r).zfill(4)}] Evaluate global model's performance...!\
            \n\t[Server] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
        print(message)
    with open("./result/cifar10_fedala.json", encoding="utf-8", mode="w") as f:
        json.dump(results, f)
