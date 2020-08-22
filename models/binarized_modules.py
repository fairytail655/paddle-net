import numpy as np
from paddle import fluid
from paddle.fluid.dygraph import Layer, Conv2D, Linear
from paddle.fluid.layers import conv2d, fc, sign

class BinarizeLinear(Layer):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__()
        self.linear = Linear(*kargs, **kwargs)

    def forward(self, input):
        input = fluid.layers.sign(input)
        w = self.linear.weight
        w_b = fluid.layers.sign(w)
        w_value = w.numpy()
        w.set_value(w_b.numpy())
        out = self.linear(input)
        w.set_value(w_value)

        return out

class BinarizeConv2d(Layer):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__()
        self.conv = Conv2D(*kargs, **kwargs)

    def forward(self, input):
        if input.shape[1] != 3:
            input = fluid.layers.sign(input)
        
        w = self.conv.weight
        w_b = fluid.layers.sign(w)
        w_value = w.numpy()
        w.set_value(w_b.numpy())
        out = self.conv(input)
        w.set_value(w_value)

        return out
