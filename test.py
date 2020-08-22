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
        net = resnet20_binary()
        for i in net.named_parameters():
            print(i[0])
        # net = BinarizeConv2d(1, 1, 1)
        # a = np.array([[[[-1.2]]]], dtype='float32')
        # input = fluid.dygraph.to_variable(a)
        # w = net.conv.weight
        # out = net(input)
        # out.backward()
        # print(w)
        # print(out)
        # print(w.gradient())

if __name__ == "__main__":
    main()
