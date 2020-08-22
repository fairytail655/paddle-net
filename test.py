import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *
import numpy as np
import collections
import json
from utils import *
import time
import pandas as pd
from time import sleep
from visualdl import LogWriter
import os
import shutil
from vl_draw import *
import sys

def main():
    with fluid.dygraph.guard():
        # net = vgg16_binary()
        # a = np.random.rand(1, 3, 32, 32).astype(np.float32)
        # input = fluid.dygraph.to_variable(a)
        # out = net(input)
        a = np.array([[-1.2]], dtype="float32")
        input = fluid.dygraph.to_variable(a)
        # net = BinarizeConv2d(num_channels=1, num_filters=1, filter_size=1)
        net = BinarizeLinear(1, 1)
        out = net(input)
        out.backward()
        g = net.linear.weight.gradient()
        print(g)

if __name__ == "__main__":
    main()
