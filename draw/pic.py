from unittest import TestCase

from draw.utills import MyPicture
from matplotlib import pyplot as plt


def pic_0_mnist_noniid_label_2():
    pic = MyPicture("../results/0_mnist_noniid-#label_2")
    pic.plot_compare(["FedAvgCNN_mnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[1chanel]_mnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[7chanel]_mnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[10chanel]_mnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[conv]_mnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[single]_mnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[multi]_mnist_noniid-#label_2_100_0.1"],
                     alias=["FedAvg", "MC-FFAF (1-chanel)", "MC-FFAF (7-chanel)", "MC-FFAF (10-chanel)",
                            "Multi", "Conv", "Single"],
                     columns="accuracy", isFillError=True)


def pic_0_fmnist_noniid_label_2():
    pic = MyPicture("../results/0_fmnist_noniid-#label_2")
    pic.plot_compare(["FedAvgCNN_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[1chanel]_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[7chanel]_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[10chanel]_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[conv]_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[single]_fmnist_noniid-#label_2_100_0.1",
                      "CnnWithFusion[multi]_fmnist_noniid-#label_2_100_0.1"],
                     alias=["FedAvg", "MC-FFAF (1-chanel)", "MC-FFAF (7-chanel)", "MC-FFAF (10-chanel)",
                            "Multi", "Conv", "Single"],
                     columns="accuracy", isFillError=True)


def pic_0_cifar10_noniid_label_2():
    pic = MyPicture("../results/0_cifar10_noniid-#label_2")
    pic.plot_compare(["FedAvgCNN_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[1chanel]_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[7chanel]_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithBatchFormer[10chanel]_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithFusion[conv]_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithFusion[single]_cifar10_noniid-#label_2_100_0.1",
                      "CnnWithFusion[multi]_cifar10_noniid-#label_2_100_0.1"],
                     alias=["FedAvg", "MC-FFAF (1-chanel)", "MC-FFAF (7-chanel)", "MC-FFAF (10-chanel)",
                            "Multi", "Conv", "Single"],
                     columns="accuracy", isFillError=True)


class TestPic(TestCase):
    def test_pic_0_mnist_noniid_label_2(self):
        pic_0_mnist_noniid_label_2()

    def test_pic_0_fmnist_noniid_label_2(self):
        pic_0_fmnist_noniid_label_2()

    def test_pic_0_cifar10_noniid_label_2(self):
        pic_0_cifar10_noniid_label_2()
