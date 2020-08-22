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
        net = BinarizeLinear(1, 1)
        w = net.linear.weight
        a = np.ones([1], dtype='float32')
        input = fluid.dygraph.to_variable(a)
        label = fluid.dygraph.to_variable(a)
        out = net(input)
        # net = Linear(1, 1, bias_attr=False)
        # w = net.weight
        # a = np.ones([1], dtype='float32')
        # input = fluid.dygraph.to_variable(a)
        # label = fluid.dygraph.to_variable(a*3)
        # out = net(input)
        loss = fluid.layers.square_error_cost(out, label)
        loss.backward()
        print(w)
        # print(out)
        # print(loss)
        # print(w.gradient())
        optimizer = fluid.optimizer.AdamOptimizer(0.1, parameter_list=net.parameters())
        for i in range(100):
            optimizer.minimize(loss)
            net.clear_gradients()
        print(w)



if __name__ == "__main__":
    main()
