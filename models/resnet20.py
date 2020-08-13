import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.dygraph import Layer, LayerList 
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Pool2D, Linear

__all__ = ['resnet20']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2D(in_planes, out_planes, 3, stride=stride, padding=1, bias_attr=False)

def norm_layer(planes):
    return BatchNorm(planes)

def act_layer(input):
    return fluid.layers.relu(input)

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
            for layer in self.downsample:
                identity = layer(identity)

        out += identity
        out = self.relu2(out)

        return out

class ResNet(Layer):

    def __init__(self, num_classes=10, in_dim=3):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(in_dim, self.inplanes, 1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = act_layer
        self.layer1 = self._make_layer(BasicBlock, 16, 3)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.avgpool = Pool2D(8, pool_type="avg")
        self.fc = Linear(64, num_classes)

        # init_model(self)
        # self.regime = {
        #     0: {'optimizer': 'SGD', 'lr': 1e-1,
        #         'weight_decay': 1e-4, 'momentum': 0.9},
        #     81: {'lr': 1e-2},
        #     122: {'lr': 1e-3, 'weight_decay': 0},
        #     164: {'lr': 1e-4}
        # }

        # self.input_transform = {
        #     'train': transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        #         transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
        #                         # normalize
        #     ]),
        #     'eval': transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         # normalize
        #     ])
        # }    

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LayerList([
                Conv2D(self.inplanes, planes * block.expansion, filter_size=1, stride=stride, bias_attr=False),
                BatchNorm(planes * block.expansion),
            ])

        layers = LayerList()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return ResNet(num_classes=num_classes, in_dim=in_dim)

# with fluid.dygraph.guard():

#     net = resnet20()

#     # for name,layer in net.named_sublayers():
#     #     print(name)
#     # print(net.state_dict())

#     fluid.save_dygraph(net.state_dict(), "resnet20")