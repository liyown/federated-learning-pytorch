import torch.nn as nn
import torch.nn.functional as F


# def conv3x3(in_planes, out_planes, stride=1):
#     # 3x3 convolution with padding
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         x += residual
#         x = self.relu(x)
#
#         return x
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         x += residual
#         x = self.relu(x)
#
#         return x
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=10):
#         self.inplanes = 16
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 32, layers[0])
#         self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(kernel_size=4)
#         self.fc = nn.Linear(256 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):  # 初始化函数，in_planes为输入通道数，planes为输出通道数，步长默认为1
        super(BasicBlock, self).__init__()
        # 定义第一个卷积，默认卷积前后图像大小不变但可修改stride使其变化，通道可能改变
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 定义第一个批归一化
        self.bn1 = nn.BatchNorm2d(planes)
        # 定义第二个卷积，卷积前后图像大小不变，通道数不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 定义第二个批归一化
        self.bn2 = nn.BatchNorm2d(planes)

        # 定义一条捷径，若两个卷积前后的图像尺寸有变化(stride不为1导致图像大小变化或通道数改变)，捷径通过1×1卷积用stride修改大小
        # 以及用expansion修改通道数，以便于捷径输出和两个卷积的输出尺寸匹配相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # 定义前向传播函数，输入图像为x，输出图像为out
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 第二个卷积和第二个批归一化后与捷径相加
        out = F.relu(out)  # 两个卷积路径输出与捷径输出相加后用ReLU激活
        return out


# 定义残差网络ResNet18
class ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = planes * block.expansion  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = F.avg_pool2d(out, 4)  # 经过一次4×4的平均池化
        out = out.view(out.size(0), -1)  # 将数据flatten平坦化
        out = self.linear(out)  # 全连接传播
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

#
# def resnet50(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#
#
# def resnet101(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#
#
# def resnet152(**kwargs):
#     return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
