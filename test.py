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
    # with fluid.dygraph.guard():
        # net = vgg16_binary()
        # fluid.layers.
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.tanh(data)

if __name__ == "__main__":
    main()
