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

# with fluid.dygraph.guard():
#     bo = [2, 3, 5]
#     val = [1e-1, 1e-2, 1e-3, 1]
#     a = fluid.dygraph.PiecewiseDecay(bo, val, 0)
#     linear = fluid.dygraph.Linear(3, 1)
#     adam = fluid.optimizer.Adam(a, parameter_list=linear.parameters())
#     inputs = np.random.rand(3).astype("float32")
#     inputs = fluid.dygraph.to_variable(inputs)
#     output = linear(inputs)
#     loss = fluid.layers.reduce_mean(output)

#     for i in range(7):
#         adam.minimize(loss)
#         lr = adam.current_step_lr()
#         print("{} : {}".format(i, lr))

# with fluid.dygraph.guard():

a = "C:\\Users\\26235\\Desktop\\code\\python\\paddle-net\\results\\resnet20\\results.csv"
#     b = os.path.splitext(a)[0]
#     model_dict, opt_dict = fluid.dygraph.load_dygraph(b)
#     net = resnet20()
#     net.load_dict(model_dict)
b = pd.read_csv(a)
c = pd.DataFrame()
print(b)
