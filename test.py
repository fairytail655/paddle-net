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
from visualDL import *
import sys

# if __name__ == '__main__':
#     try:
#         shutil.rmtree("./vl_log/scalar_test/train")
#     except:
#         print("haha")
#     value = [i/1000.0 for i in range(1000)]
#     # for i in range(1000):
#     # 初始化一个记录器
#     with LogWriter(logdir="./vl_log/scalar_test/train") as writer:
#         for step in range(500):
#             # 向记录器添加一个tag为`acc`的数据
#             writer.add_scalar(tag="resnet20/train/acc", step=step, value=value[step])
#             # 向记录器添加一个tag为`loss`的数据
#             writer.add_scalar(tag="resnet20/train/loss", step=step, value=1/(value[step] + 1))
#             sleep(1)

train_thread = DrawScalar("./vl_log/scalar_test/resnet20/train", "resnet20")
val_thread = DrawScalar("./vl_log/scalar_test/resnet20/val", "resnet20")
train_thread.start()
val_thread.start()

train_values = [i/1000 for i in range(1,1000)]
val_values = [1000/i for i in range(1,1000)]

try:
    for i in range(500):
        train_thread.set_value(i, {'loss': i, 'acc': train_values[i]})
        val_thread.set_value(i, {'loss': i, 'acc': val_values[i]})
        train_thread.event.set()
        val_thread.event.set()
        sleep(0.5)
except KeyboardInterrupt:
    print("Main KeyboardInterrupt....")
    train_thread.flag = True
    train_thread.event.set()
    train_thread.join()
    val_thread.flag = True
    val_thread.event.set()
    val_thread.join()
    sys.exit(1)
