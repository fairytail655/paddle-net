import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *
import numpy as np
import collections

with fluid.dygraph.guard():

    model = resnet20()
    state_dict = model.state_dict()
    param = state_dict.copy()
    param['epoch'] = 1
    param['aaa'] = {"a": 1}
    # param = collections.OrderedDict()
    # param.values()
    # param['haha'] = 'a'
    # param.update()
    # param['model_state_dict'] = state_dict
    # print(state_dict)
    print(type(param))
    # fluid.dygraph.save_dygraph(param, 'haha')