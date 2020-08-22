import paddle
import paddle.fluid as fluid
import math
import numpy as np
from paddle.fluid.dygraph import Layer, Sequential 
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Pool2D, Linear
from .binarized_modules import *

__all__ = ['resnet20_binary']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias_attr=False)

def norm_layer(planes):
    return BatchNorm(planes)

def linear(in_planes, out_planes):
    return BinarizeLinear(in_planes, out_planes, bias_attr=False)

def act_layer(input):
    return fluid.layers.tanh(input)

class BasicBlock(Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = act_layer
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = act_layer
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu2(out)

        return out

class ResNet(Layer):

    def __init__(self, num_classes=10, in_dim=3):
        super(ResNet, self).__init__()
        self.inflate = 4
        self.inplanes = 16*self.inflate
        self.conv1 = conv3x3(in_dim, self.inplanes, 1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = act_layer
        self.layer1 = self._make_layer(BasicBlock, 16*self.inflate, 3)
        self.layer2 = self._make_layer(BasicBlock, 32*self.inflate, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64*self.inflate, 3, stride=2)
        self.avgpool = Pool2D(8, pool_type="avg")
        self.fc = linear(64*self.inflate, num_classes)

        # init_model(self)
        self.train_config = {
            'epochs': 200,
            'batch_size': 128,
            'opt_config': {
                'optimizer': 'Adam',
                'learning_rate': {
                    'bound': [80, 120, 160, 180],
                    'value': [1e-3, 1e-4, 1e-5, 5e-6, 1e-6]
                },
                'weight_decay': 1e-4,
                # 'momentum': 0.9,
                # 0: {, 'lr': 1e-1,
                #     'weight_decay': 1e-4, 'momentum': 0.9},
                # 81: {'lr': 1e-2},
                # 122: {'lr': 1e-3, 'weight_decay': 0},
                # 164: {'lr': 1e-4}
            },
            'transform': {
                'train': None,
                'eval': None,
                # 'train': transforms.Compose([
                #     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                #     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
                #                     # normalize
                # ]),
                # 'eval': transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #     # normalize
                # ])
            },    
        }

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2D(self.inplanes, planes * block.expansion, filter_size=1, stride=stride, bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x

def resnet20_binary(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return ResNet(num_classes=num_classes, in_dim=in_dim)
