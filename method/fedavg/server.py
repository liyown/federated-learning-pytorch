import json
import numpy as np
import torch
import torchvision.transforms
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets.partitioned_cifar import PartitionCIFAR
from method.fedavg.client import create_clients
from model.models import MnistCNN, Cifar10CNN
from utils.utils import seed_torch, init_net, transmit_model, update_selected_clients, \
    average_model, draw, update_LR

if __name__ == "__main__":
    with open('./config.yaml', encoding="utf-8") as c:
        configs = yaml.load(c, Loader=yaml.FullLoader)
    print(
        "\nid:{}dataset_name:{}--model:{}--optimizer:{}--lr:{}--num_clients:{}--fraction:{}--num_local_epochs:{}--batch_size:{}--record:{}\n".format(
            configs["global_config"]["record_id"],
            configs["data_config"]["dataset_name"],
            configs["client_config"]["model"],
            configs["client_config"]["optimizer"],
            configs["client_config"]["lr"],
            configs["fed_config"]["num_clients"],
            configs["fed_config"]["fraction"],
            configs["client_config"]["num_local_epochs"],
            configs["client_config"]["batch_size"],
            configs["global_config"]["record"]
        ))
    # tensorboard
    writer = SummaryWriter(log_dir=configs["log_config"]["log_path"], filename_suffix="FL")
    # -------------------------------------------有效代码开始———————————————————————————————————————————— #
    models = None
    # 修改模型的地方
    if configs["client_config"]["model"] == "Cifar10CNN":
        models = Cifar10CNN()
    elif configs["client_config"]["model"] == "MnistCNN":
        models = MnistCNN()

    # 设置随机种子
    seed_torch()
    # 初始化模型
    models = init_net(models, configs["init_config"]["init_type"], configs["init_config"]["init_gain"])
    device = configs["client_config"]["device"]

    # 创建分割数据集及其类
    PartitionCifar10 = PartitionCIFAR(configs["data_config"]["data_path"], "data", configs["data_config"]["dataset_name"],
                                      configs["fed_config"]["num_clients"],
                                      download=True, preprocess=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                      target_transform=None, **configs["data_config"]["partition_config"])

    draw(PartitionCifar10, "./result")
    # assign dataset to each client
    clients = create_clients(PartitionCifar10, models)

    # prepare hold-out dataset for evaluation
    test_dataloader = PartitionCifar10.get_dataloader(batch_size=64, type="test")

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
        transmit_model(models, select_client)
        # 更新选中的客户端，并且返回选中客户端所有的数据量
        selected_total_size = update_selected_clients(select_client)
        # 进行模型聚合
        models = average_model(select_client, selected_total_size)
        # 学习率衰减
        update_LR(clients, r, select_client[0].lr, 0.99, 5)

        # -----------------   测试模型   ---------------------- #
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
        record_id = configs["global_config"]["record_id"]
        if configs["global_config"]["record"]:
            writer.add_scalars(
                'Loss',
                {
                    record_id + f"{dataset_name}_{models.__class__.__name__}": test_loss},
                r
            )
            writer.add_scalars(

                'Accuracy',
                {
                    record_id + f"{dataset_name}]_{models.__class__.__name__}": test_accuracy},
                r
            )
        message = "\033[91m" + f"[Round: {str(r).zfill(4)}] " + "\033[0m" + f"Evaluate global model's performance...!\
            \n\t[Server] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
        print(message)
    with open("result/cifar10_fedavg_balance_{balance}_partition_{partition}_unbalance_sgm_{unbalance_sgm}_num_shards_{num_shards}_dir_alpha_{dir_alpha}.json".format(**configs["data_config"]["partition_config"]), encoding="utf-8", mode="w") as f:
        json.dump(results, f)
