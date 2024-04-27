import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List


def avgResult(result1, result2):
    """
    Calculate the average of two results.
    :param result1: the first result
    :param result2: the second result
    :return: the average result
    """
    # 判断两个结果的长度是否相等
    assert len(result1) == len(result2)
    result = []
    for i in range(len(result1)):
        result.append((result1[i] + result2[i]) / 2)
    return result


def avgResults(results):
    """
    Calculate the average of multiple results.
    :param results: the results
    :return: the average result
    """
    result = results[0]
    for i in range(1, len(results)):
        result = avgResult(result, results[i])
    return result


# 画带有误差
import matplotlib.pyplot as plt
import numpy as np


def plotResults(results, labels, xlabel, ylabel, filename, hline_y=None):
    """
    Plot the results with custom markers.
    :param results: the results
        shape: (numResults, numEpochs)
    :param labels: the labels
    :param xlabel: the label of x-axis
    :param ylabel: the label of y-axis
    :param filename: the filename of the figure
    :param markers: a list of marker styles (optional)
    :return: None
    """
    assert len(results) == len(labels)
    plt.figure(figsize=(10, 5))
    markers = ['o', 's', '^', 'D']  # You can customize the marker styles here

    for i in range(len(results)):
        marker = markers[i % len(markers)]  # Cycle through the marker styles
        plt.plot(results[i], label=labels[i], marker=marker, markevery=10)

    if hline_y is not None:
        plt.axhline(y=hline_y, color='gray', linestyle='--')
        # plt.annotate(f'{hline_y}', xy=(-2, hline_y), xytext=(0, hline_y + 1),
        #              arrowprops=dict(arrowstyle='->'))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()


# 计算多个结果最大值的均值和标准差，输入的是results shape: (numResults, numEpochs)
def maxAvgStd(results):
    """
    Calculate the average and standard deviation of the maximum value of multiple results.
    :param results: the results
        shape: (numResults, numEpochs)
    :return: the average and standard deviation of the maximum value
    """
    maxValues = []
    for result in results:
        maxValues.append(max(result))
    return [sum(maxValues) / len(maxValues), np.std(maxValues)]


def movingAverage(data, window_size=10):
    """Calculate the moving average of the loss and accuracy."""
    if window_size == 0:
        return data
    else:
        return np.convolve(data, np.ones(window_size), 'valid') / window_size


# 指数加权平均
def exponentialMovingAverage(data, alpha=0.7):
    """Calculate the exponential moving average of the loss and accuracy."""
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * ema[i - 1] + (1 - alpha) * data[i])
    return ema


class MyPicture:
    def __init__(self, dir_path: str):
        """
        Initialize the MyPicture class.
        """
        self.data = None
        self.dir_path = dir_path
        # 读取路径下所有的csv文件，按照文件名分类
        self.save_path = os.path.join(dir_path, "pictures")
        self.init()

    def init(self):
        """
        Load the data from the csv files.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        data = {}
        for file_name in os.listdir(self.dir_path):
            key = file_name.split("_")[:-1]
            key = "_".join(key)
            if key not in data:
                data[key] = {}

        for file_name in os.listdir(self.dir_path):
            if not file_name.endswith(".csv"):
                continue
            key = file_name.split("_")[:-1]
            key = "_".join(key)
            dataFrame = pd.read_csv(f"{self.dir_path}/{file_name}")
            for column in dataFrame.columns:
                if column == "epoch":
                    continue
                if column not in data[key]:
                    data[key][column] = []
                data[key][column].append(dataFrame[column].values)
        self.data = data

    def plot_single(self, keys: List[str], isFillError: bool = True):
        """
        Plot a single figure.
        """
        for key in keys:
            results = self.data[key]
            for column in results.keys():
                if column == "epoch":
                    continue
                if column == "similarity":
                    self.plotResultsWithErrors(results[column], "epoch", column,
                                               '\\\\?\\' + os.path.abspath(
                                                   os.path.join(self.save_path, f"{key}_{column}.{isFillError}.png")),
                                               hline_y=1, isFillError=isFillError)
                self.plotResultsWithErrors(results[column], "epoch",
                                           column, '\\\\?\\' + os.path.abspath(
                        os.path.join(self.save_path, f"{key}_{column}.{isFillError}.png")), isFillError=isFillError)

    def plot_compare(self, keys: List[str], alias: List[str], columns: str, isFillError: bool = True):
        """
        Plot the comparison figure for a single column.
        """
        results = {}

        for key in keys:
            for column in self.data[key].keys():
                if column == "epoch" or column not in columns:
                    continue
                if column not in results:
                    results[column] = {key: self.data[key][column]}
                else:
                    results[column][key] = self.data[key][column]

        for column in results.keys():
            if column == "similarity":
                self.plotMutiResultsWithErrors(results[column], alias, "epoch", column,
                                               '\\\\?\\' + os.path.abspath(
                                                   os.path.join(self.save_path, f"{alias}_{column}.{isFillError}_compare.png")),
                                               hline_y=1, isFillError=isFillError)
            self.plotMutiResultsWithErrors(results[column], alias, "epoch", column,
                                           '\\\\?\\' + os.path.abspath(
                                               os.path.join(self.save_path, f"{alias}_{column}.{isFillError}_compare.png")),
                                           isFillError=isFillError)

    def plot_compare_all(self, keys: List[str], alias: List[str] = None, isFillError: bool = True):
        """
        Plot the comparison figure for all columns.
        """
        results = {}
        for key in keys:
            for column in self.data[key].keys():
                if column == "epoch":
                    continue
                if column not in results:
                    results[column] = {key: self.data[key][column]}
                else:
                    results[column][key] = self.data[key][column]

        for column in results.keys():
            if column == "similarity":
                self.plotMutiResultsWithErrors(results[column], alias, "epoch", column,
                                               '\\\\?\\' + os.path.abspath(
                                                   os.path.join(self.save_path, f"{alias}_{column}.{isFillError}_compare.png")),
                                               hline_y=1)
            self.plotMutiResultsWithErrors(results[column], alias, "epoch",
                                           column, '\\\\?\\' + os.path.abspath(
                    os.path.join(self.save_path, f"{alias}_{column}.{isFillError}_compare.png")))

    def plot_all(self):
        """
        plot all the figures.
        single figures and comparison figures.
        """
        self.plot_single(self.data.keys())
        self.plot_compare_all(self.data.keys())

    @staticmethod
    def plotResultsWithErrors(results: List[List[int]], xlabel: str, ylabel: str, filename: str, hline_y: int = None, isFillError: bool = True):
        """
        Plot the results with error lines and filled error bands.
        :param results: a column of results, represented as a list of multiple experiments
            Example1:
            [[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]]
        :param xlabel: the label of x-axis
        :param ylabel: the label of y-axis
        :param filename: the filename of the figure
        :param hline_y: the y-value for a horizontal line (optional)
        :return: None
        """
        # 设置分辨率
        plt.figure(figsize=(10, 5))

        num_epochs = len(results[0])

        # 计算每个数据点的平均值
        means = np.mean(np.array(results), axis=0)

        # 计算每个数据点的标准差
        stds = np.std(np.array(results), axis=0)

        # 绘制平均值的折线
        plt.plot(np.arange(num_epochs), means, label='Mean', marker='o')

        # 绘制误差线
        # for i in range(num_epochs):
        #     plt.plot([i, i], [means[i] - stds[i], means[i] + stds[i]], color='black')
        if isFillError:
            # 填充误差范围的颜色
            plt.fill_between(np.arange(num_epochs), means - stds, means + stds, alpha=0.3)

        if hline_y is not None:
            plt.axhline(y=hline_y, color='gray', linestyle='--')

        # 添加标题
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.show()

    @staticmethod
    def plotMutiResultsWithErrors(results: dict, alias: List[str], xlabel, ylabel, filename, hline_y=None,
                                  isFillError: bool = True):
        """
        Plot the results with error lines and filled error bands.
        :param results: a dict of results, each key-value pair represents an experiment
            Example1:
            {
                "FedAvg": [[1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5]],
                "FedAvg2": [[1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5]]
            }
        :param alias: the alias of the results
        :param xlabel: the label of x-axis
        :param ylabel: the label of y-axis
        :param filename: the filename of the figure
        :param hline_y: the y-value for a horizontal line (optional)
        :return: None
        """
        plt.figure(figsize=(10, 5))
        markers = ['o', 's', '^', 'D']  # You can customize the marker styles here
        line = ['-', '--', '-.', ':']

        style = [{"marker": "", "line": "-"},  # 实线无点
                 {"marker": "^", "line": "--"},  # 实线方块
                 {"marker": "^", "line": ":"},  # 实线三角
                 {"marker": "^", "line": "-."},  # 实线菱形
                 {"marker": "x", "line": "--"},
                 {"marker": "x", "line": ":"},
                 {"marker": "x", "line": "-."},
                 {"marker": "o", "line": "--"},
                 {"marker": "o", "line": ":"},
                 {"marker": "o", "line": "-."},
                 {"marker": "D", "line": "--"},
                 {"marker": "D", "line": ":"},
                 {"marker": "D", "line": "-."},
                 {"marker": "s", "line": "--"},
                 {"marker": "s", "line": ":"},
                 {"marker": "s", "line": "-."}]

        for index, (alia, (key, value)) in enumerate(zip(alias, results.items())):
            style_index = index % len(style)

            # 计算每个数据点的平均值
            means = np.mean(value, axis=0)

            # 计算每个数据点的标准差
            stds = np.std(value, axis=0)

            # 绘制平均值的折线
            plt.plot(np.arange(len(means)), means, label=alia, marker=style[style_index]["marker"],
                     linestyle=style[style_index]["line"])

            # 绘制误差线
            # for i in range(len(means)):
            #     plt.plot([i, i], [means[i] - stds[i], means[i] + stds[i]], color='black')
            if isFillError:
                # 填充误差范围的颜色
                plt.fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.3)

        if hline_y is not None:
            plt.axhline(y=hline_y, color='gray', linestyle='--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.show()
