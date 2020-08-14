import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D
from models import *

net = resnet20()

image = fluid.layers.data(name='img', shape=[3, 32, 32], dtype='float32')
# label = fluid.layers.data(name='label', shape=[10], dtype='int64')
# feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
predict = net(image)

# loss = fluid.layers.softmax_with_cross_entropy(input=predict, label=label)
# avg_loss = fluid.layers.mean(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

# 数据输入及训练过程

# 保存预测模型。注意，用于预测的模型网络结构不需要保存标签和损失。
fluid.io.save_inference_model(dirname="./pre_model", feeded_var_names=['img'], target_vars=[predict], executor=exe, params_filename='__params__')
