import os
import paddle
import shutil
import pandas as pd
import paddle.fluid as fluid
import json
import time
import numpy as np
from bokeh.io import output_file, save, show

class MyLogging(object):

    def __init__(self, log_file='./log.txt'):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write("")

    def info(self, message):
        date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
        message = "[{}] - INFO: {}\n".format(date, message)
        print(message, end='')
        with open(self.log_file, 'a') as f:
            f.write(message)

    def debug(self, message):
        date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
        message = "[{}] - DEBUG: {}\n".format(date, message)
        with open(self.log_file, 'a') as f:
            f.write(message)

class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.results = pd.DataFrame()

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

def save_checkpoint(model_dict, train_dict, is_best, path='.', filename='checkpoint', save_all=False):
    state_dict = model_dict.copy()
    # save model dict
    filename = os.path.join(path, filename)
    fluid.dygraph.save_dygraph(state_dict, filename)
    # save train dict
    temp = json.dumps(train_dict, sort_keys=True, indent=4)
    with open(filename+'.json', 'w') as f:
        f.write(temp)
    if is_best:
        shutil.copyfile(filename+'.pdparams', os.path.join(path, 'best_model.pdparams'))
        shutil.copyfile(filename+'.json', os.path.join(path, 'best_model.json'))
    if save_all:
        shutil.copyfile(filename+'.pdparams', os.path.join(
            path, 'checkpoint_epoch_%s.pdparams' % train_dict['epoch']))
        shutil.copyfile(filename+'.json', os.path.join(
            path, 'checkpoint_epoch_%s.json' % train_dict['epoch']))

def calculate_params(model, show_fc=print):
    train_param = 0
    n_train_param = 0
    show_fc("===================================================================")
    for p in model.parameters():
        if p.trainable:
            train_param += np.prod(p.shape)
        else:
            n_train_param += np.prod(p.shape)
    total_param = train_param + n_train_param
    show_fc("total parameters: {} -> {:.2f}MB".format(total_param, total_param*4/1024/1024))
    show_fc("trainable parameters: {} -> {:.2f}MB".format(train_param, train_param*4/1024/1024))
    show_fc("===================================================================")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': paddle.fluid.optimizer.SGDOptimizer,
    'Momentum': paddle.fluid.optimizer.MomentumOptimizer,
    'Adam': paddle.fluid.optimizer.AdamOptimizer,
}


def get_optimizer(start_epoch, factor, opt_config, model):
    """Reconfigures the optimizer according to config dict"""
    optimizer = __optimizers[opt_config['optimizer']]
    opt_params = {'parameter_list': model.parameters()}
    for key in opt_config.keys():
        if key == 'learning_rate':
            lr = opt_config[key]
            bo = [val*factor for val in lr['bound']]
            decay_lr = fluid.dygraph.PiecewiseDecay(bo, lr['value'], start_epoch*factor)
            opt_params['learning_rate'] = decay_lr
        elif key == 'weight_decay':
            regularization = fluid.regularizer.L2Decay(regularization_coeff=opt_config[key])
            opt_params['regularization'] = regularization
        elif key == 'momentum':
            opt_params['momentum'] = opt_config[key]

    return optimizer(**opt_params)
