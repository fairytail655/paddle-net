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

thread = DrawHistogram("./vl_log/histogram/vgg16", "vgg16")
thread.start()

def main():
    with fluid.dygraph.guard():
        net = vgg16()
        layers = net.named_parameters()
        for layer in layers:
            print(layer[0])
            # a = layer[1].numpy().reshape(-1)
            # thread.set_value(epoch=0, value=a)
            # sleep(1)
            # break
#     # value = np.arange(100)
#     # for i in range(5):    
#         # thread.set_value(epoch=i, value=value+i)
#         # sleep(1)
#     net = vgg16()
#     net.su

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("main KeyboardInterrupt...")
    thread.stop()
    thread.join()
    sys.exit(1)

# with fluid.dygraph.guard():

#     net = vgg16()
#     layers = net.named_parameters()
#     # a = net.features.named_parameters()
#     for layer in layers:
#         a = layer[1].numpy().reshape(-1)
#         print(a)
#         break