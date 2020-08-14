import os
import logging
import paddle
import shutil
import pandas as pd
import paddle.fluid as fluid
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
#from bokeh.charts import Line, defaults
#
#defaults.width = 800
#defaults.height = 400
#defaults.tools = 'pan,box_zoom,wheel_zoom,box_select,hover,resize,reset,save'


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create hander
    hander = logging.FileHandler(log_file, mode='w')
    hander.setLevel(logging.INFO)
    # define output format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    hander.setFormatter(formatter)
    # add hander
    logger.addHandler(hander)

class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
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
        if os.path.isfile(path):
            self.results.read_csv(path)

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


def save_checkpoint(model_dict, state, is_best, path='.', filename='checkpoint', save_all=False):
    state_dict = model_dict.copy()
    state_dict['epoch'] = state['epoch']
    state_dict['model'] = state['model']
    state_dict['config'] = state['config']
    state_dict['best_prec'] = state['best_prec']
    state_dict['regime'] = state['regime']
    filename = os.path.join(path, filename)
    fluid.dygraph.save_dygraph(state_dict, filename)
    if is_best:
        shutil.copyfile(filename+'.pdparams', os.path.join(path, 'best_model.pdparams'))
    if save_all:
        shutil.copyfile(filename+'.pdparams', os.path.join(
            path, 'checkpoint_epoch_%s.pdparams' % state['epoch']))


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


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
        return optimizer

        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


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
