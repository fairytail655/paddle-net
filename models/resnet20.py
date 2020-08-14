import paddle
import paddle.fluid as fluid
import math
import numpy as np
from paddle.fluid.dygraph import Layer, Sequential 
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
            identity = self.downsample(identity)

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
        self.train_param = {
            'batch_size': 128,
            'regime': {
                0: {'optimizer': 'SGD', 'lr': 1e-1,
                    'weight_decay': 1e-4, 'momentum': 0.9},
                81: {'lr': 1e-2},
                122: {'lr': 1e-3, 'weight_decay': 0},
                164: {'lr': 1e-4}
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
        x = fluid.layers.reshape(x, [x.shape[1], -1])
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

# x = np.random.randn(*[2,3,32,32])
# x = x.astype('float32')
# with fluid.dygraph.guard():
#     # 创建LeNet类的实例，指定模型名称和分类的类别数目
    # m = ResNet()
    # x = fluid.dygraph.to_variable(x)
#     # 通过调用LeNet从基类继承的sublayers()函数，
#     # 查看LeNet中所包含的子层
    # c = m(x)
    # print(c)
    # print(m.sublayers())
    # x = fluid.dygraph.to_variable(x)
    # for item in m.sublayers():
    #     # item是LeNet类中的一个子层
    #     # 查看经过子层之后的输出数据形状
    #     try:
    #         x = item(x)
    #     except:
    #         x = fluid.layers.reshape(x, [x.shape[0], -1])
    #         x = item(x)
    #     if len(item.parameters())==2:
    #         # 查看卷积和全连接层的数据和参数的形状，
    #         # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
    #         print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
    #     else:
    #         # 池化层没有参数
    #         print(item.full_name(), x.shape)