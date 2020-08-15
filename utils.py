import os
# import logging
import paddle
import shutil
import pandas as pd
import paddle.fluid as fluid
import json
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
import time
#from bokeh.charts import Line, defaults
#
#defaults.width = 800
#defaults.height = 400
#defaults.tools = 'pan,box_zoom,wheel_zoom,box_select,hover,resize,reset,save'

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
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = pd.DataFrame()

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        # if self.results is None:
        #     self.results = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


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


# def accuracy(output, target):
#     """Computes the precision@k for the specified values of k"""
#     batch_size = target.size(0)

#     _, pred = output.float().topk(1, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     correct = correct[:1].view(-1).float().sum(0)
#     res = correct.mul_(100.0 / batch_size)

#     return res

#     # kernel_img = model.features[0][0].kernel.data.clone()
#     # kernel_img.add_(-kernel_img.min())
#     # kernel_img.mul_(255 / kernel_img.max())
#     # save_image(kernel_img, 'kernel%s.jpg' % epoch)
