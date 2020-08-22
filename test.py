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
        net = vgg16_binary()
        a = np.random.rand(1, 3, 32, 32).astype(np.float32)
        input = fluid.dygraph.to_variable(a)
        out = net(input)
        print(out)

if __name__ == "__main__":
    main()
