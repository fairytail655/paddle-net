import paddle
import paddle.fluid as fluid
import math
import numpy as np
from paddle.fluid.dygraph import Layer, Sequential, Conv2D, BatchNorm, Pool2D, Linear, Dropout, Pool2D

__all__ = ['vgg16']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2D(in_planes, out_planes, 3, stride=stride, padding=1, bias_attr=False)

def linear(in_planes, out_planes):
    return Linear(in_planes, out_planes, bias_attr=False)

def norm_layer(planes):
    return BatchNorm(planes)

def act_layer(input):
    return fluid.layers.relu(input)

cfg = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]

class BasicConv(Layer):

    def __init__(self, in_planes, out_planes):
        super(BasicConv, self).__init__()
        self.conv = conv3x3(in_planes, out_planes)
        self.bn = norm_layer(out_planes)
        self.act = act_layer

    def forward(self, input):
        conv_out = self.conv(input)
        bn_out = self.bn(conv_out)
        act_out = self.act(bn_out)
        return act_out

class BasicLiner(Layer):

    def __init__(self, in_planes):
        super(BasicLiner, self).__init__()
        self.fc = linear(in_planes, in_planes)
        self.bn = norm_layer(in_planes)
        self.act = act_layer
        self.dp = Dropout()

    def forward(self, input):
        fc_out = self.fc(input)
        bn_out = self.bn(fc_out)
        act_out = self.act(bn_out)
        dp_out = self.dp(act_out)
        return dp_out

class VGG16(Layer):

    def __init__(self, num_classes=10, in_dim=3):
        super(VGG16, self).__init__()
        self.features = self.__make_layer(in_dim, cfg)
        self.basic_linear1 = BasicLiner(512)
        self.basic_linear2 = BasicLiner(512)
        self.fc = Linear(512, num_classes)

        self.train_config = {
            'epochs': 150,
            'batch_size': 128,
            'opt_config': {
                'optimizer': 'Adam',
                'learning_rate': {
                    'bound': [40, 60, 100],
                    'value': [1e-3, 1e-4, 1e-5, 5e-6]
                },
                'weight_decay': 1e-4,
            },
            'transform': {
                'train': None,
                'eval': None,
            },    
        }

    def __make_layer(self, in_dim, cfg):
        in_planes = in_dim
        layer_list = []
        for layer in cfg:
            for out_planes in layer:
                layer_list.append(BasicConv(in_planes, out_planes))
                in_planes = out_planes
            layer_list.append(Pool2D(pool_size=2, pool_type='max', pool_stride=2))
        return Sequential(*layer_list)

    def forward(self, input):
        fs_out = self.features(input)
        _fs_out = fluid.layers.reshape(fs_out, [fs_out.shape[0], -1])
        l1_out = self.basic_linear1(_fs_out)
        l2_out = self.basic_linear2(l1_out)
        fc_out = self.fc(l2_out)
        return fc_out

def vgg16(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return VGG16(num_classes=num_classes, in_dim=in_dim)


# net = VGG16()
# image = fluid.layers.data(name='image', shape=[1, 3, 32, 32], dtype='float32', append_batch_size=False)
# predict = net(image)
# exe = fluid.Executor(fluid.CPUPlace())
# exe.run(fluid.default_startup_program())
# fluid.io.save_inference_model(dirname='./', feeded_var_names=['image'], 
#                               target_vars=[predict], executor=exe, params_filename='__params__')
