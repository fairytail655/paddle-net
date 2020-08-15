import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *
import numpy as np
import collections
import json
from utils import *
import time

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

a = {'0': 0, '1': 1, '2': 2}
b = {**a, '3': 3}
# b = a.pop('0')
print(b)