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

class model(Layer):

    def __init__(self):
        super(model, self).__init__()
        self.conv = BinarizeConv2d(1, 1, 3, bias_attr=False)

    def forward(self, input):
        return self.conv(input)

def main():
    with fluid.dygraph.guard():
        net = vgg16()
        for layer in net.named_parameters():
            print(layer[0])
            # return


if __name__ == "__main__":
    main()
