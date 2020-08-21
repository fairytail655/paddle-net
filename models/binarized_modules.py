import numpy as np
from paddle.fluid.dygraph import Layer, Conv2D, Linear
from paddle.fluid.layers import conv2d, fc, sign

class BinarizeLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        return out

class BinarizeConv2d(Conv2D):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        return out
