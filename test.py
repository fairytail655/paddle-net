import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *
import numpy as np
import collections

class net(fluid.dygraph.Layer):
    def __init__(self, n={"a": 0}):
        super(net, self).__init__()
        self.conv1 = Conv2D(3, 3, 3)

    def forward(self, input):
        return self.conv1(input)

with fluid.dygraph.guard():

    model = net()
    state_dict = model.state_dict()
    state_dict['epoch'] = 1
    # param = collections.OrderedDict()
    # param.values()
    # param['haha'] = 'a'
    # param.update()
    # param['model_state_dict'] = state_dict
    print(state_dict)
    # print(type(param))
    fluid.dygraph.save_dygraph(state_dict, 'haha')