import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *
import numpy as np
import collections
import json
from utils import *
import time

my_loggging = MyLogging()

# my_loggging.info("haha %s", "haha")
# my_loggging.debug("aaaa")
