import argparse
import platform
import os
import sys
import time
import models
import math
# from preprocess import get_transform
import paddle
import paddle.fluid as fluid
import numpy as np
from data import get_dataset
from utils import *
from datetime import datetime
from ast import literal_eval
from draw import *
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from vl_draw import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Paddle ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='resnet20',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: lenet)')
# parser.add_argument('--input_size', type=int, default='32',
#                     help='image input size')
# parser.add_argument('--model_config', default='',
#                     help='additional architecture configuration')
# parser.add_argument('--type', default='torch.cuda.FloatTensor',
#                     help='type of tensor - e.g torch.cuda.HalfTensor')
# parser.add_argument('--gpu', default='0',
#                     help='gpus used for training - e.g 0(none),1')
# parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                     help='number of data loading workers (default: 8)')
parser.add_argument('--device', default='CPU', type=str, metavar='OPT',
                    help='choose to use CPU or GPU')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')

def main():
    global my_logging
    global args, best_prec
    global progress, task2, task3
    global input_size, in_dim, target_size
    global thread_train, thread_val

    best_prec = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = './tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    my_logging = MyLogging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    my_logging.info("saving to {}".format(save_path))
    my_logging.info("run arguments: {}".format(args))

    # create model
    my_logging.info("creating model {}".format(args.model))
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset}

    # if args.device.upper() == 'GPU': 
    #     device = fluid.CUDAPlace(0)
    # else:
    #     device = fluid.CPUPlace()
    if platform.system() == 'Linux':
        device = fluid.CUDAPlace(0)
    else:
        device = fluid.CPUPlace()

    # save net struct: __model__
    # command: visualdl --model save_path/struct/__model__ --port 8080
    if args.dataset == 'mnist':
        input_size = 28
        in_dim = 1
        target_size = 1
    elif args.dataset == 'cifar10':
        input_size = 32
        in_dim = 3
        target_size = 1
    net = model(**model_config)
    params = getattr(net, 'train_config', {'batch_size': args.batch_size})
    image = fluid.layers.data(name='img', shape=[params['batch_size'], in_dim, input_size, input_size], 
                              dtype='float32', append_batch_size=False)
    predict = net(image)
    exe = fluid.Executor(device)
    exe.run(fluid.default_startup_program())
    fluid.io.save_inference_model(dirname=os.path.join(save_path,'struct'), feeded_var_names=['img'], 
                                  target_vars=[predict], executor=exe, params_filename='__params__')

    with fluid.dygraph.guard(device):

        model = model(**model_config)
        my_logging.info("created model with configuration: {}".format(model_config))

        # optionally resume from a checkpoint
        if args.evaluate:
            if not os.path.isfile(args.evaluate):
                parser.error('invalid checkpoint: {}'.format(args.evaluate))
            filename = os.path.splitext(args.evaluate)[0]
            train_file = filename + '.json'
            if not os.path.isfile(train_file):
                parser.error('missing file: {}'.format(train_file))
            with open(train_file, 'r') as f:
                train_conf = json.load(f)
            model_dict, opt_dict = fluid.dygraph.load_dygraph(filename)
            model.load_dict(model_dict)
            my_logging.info("loaded checkpoint '{}' (epoch {})"
                            .format( args.evaluate, train_conf['epoch']))

        elif args.resume:
            checkpoint_dir = args.resume
            if os.path.isdir(checkpoint_dir):
                results.load(os.path.join(checkpoint_dir, 'results.csv'))
                checkpoint = os.path.join(
                    checkpoint_dir, 'checkpoint')
            else:
                parser.error('invalid checkpoint dir: {}'.format(checkpoint_dir))
            if os.path.isfile(checkpoint+'.pdparams'):
                my_logging.info("loading checkpoint '{}'".format(checkpoint+'.pdparams'))
                model_dict, opt_dict = fluid.dygraph.load_dygraph(checkpoint)
                model.load_dict(model_dict)
                train_file = checkpoint + '.json'
                if not os.path.isfile(train_file):
                    parser.error('missing file: {}'.format(train_file))
                with open(train_file, 'r') as f:
                    train_conf = json.load(f)
                args.start_epoch = train_conf['epoch']
                best_prec = train_conf['best_prec']
                my_logging.info("loaded checkpoint '{}' (epoch {})"
                                .format(checkpoint+'.params', train_conf['epoch']))
            else:
                parser.error('no checkpoint found: {}'.format(checkpoint+'.pdparams'))

        # Train Config Load
        train_config = getattr(model, 'train_config', {})
        if 'epochs' in train_config:
            epochs = train_config['epochs']
        else:
            epochs = args.epochs

        if 'batch_size' in train_config:
            batch_size = train_config['batch_size']
        else:
            batch_size = args.batch_size

        if 'opt_config' in train_config:
            opt_config = train_config['opt_config']
        else:
            opt_config = { 'optimizer': 'SGD',
                           'learning_rate': {
                           'bound': [],
                           'value': [args.lr]
                           }
            }

        if 'transform' in train_config:
            transform = train_config['transform']
        else:
            transform = {'train': None, 'eval': None}

        if 'criterion' in train_config:
            criterion = train_config['criterion']
        else:
            criterion = fluid.layers.softmax_with_cross_entropy

        my_logging.info("\n----------------------------------------------\n"
                     "epochs: {}\tbatch_size: {}\n"
                     "opt_config: {}\n"
                     "transform: {}\n"
                     "criterion: {}\n"
                     "----------------------------------------------"
                     .format(epochs, batch_size, opt_config, transform, criterion.__name__)
        )

        val_data = get_dataset(args.dataset, 'eval', transform['eval'])
        val_len = len(list(val_data()))
        val_loader = paddle.batch(paddle.reader.shuffle(val_data, val_len),
                                    batch_size=batch_size)
        my_logging.info('val dataset size: {}'.format(val_len))

        if args.evaluate:
            with Progress("[progress.description]{task.description}{task.completed}/{task.total}",
                        BarColumn(),
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TimeRemainingColumn(),
                        auto_refresh=False) as progress:
                task3 = progress.add_task("[yellow]validating:", total=math.ceil(val_len/batch_size))
                val_loss, val_prec1 = validate(val_loader, model, criterion, 0)
                my_logging.info('Evaluate {0}\t'
                            'Validation Loss {val_loss:.4f} \t'
                            'Validation Prec@1 {val_prec1:.3f} \t'
                            .format(args.evaluate, val_loss=val_loss, val_prec1=val_prec1))
            return

        train_data = get_dataset(args.dataset, 'train', transform['train'])
        train_len = len(list(train_data()))
        train_loader = paddle.batch(paddle.reader.shuffle(train_data, train_len),
                                    batch_size=batch_size)
        my_logging.info('train dataset size: {}'.format(train_len))

        optimizer = get_optimizer(args.start_epoch, math.ceil(train_len/batch_size), opt_config, model)

        log_path = os.path.join("./vl_log/scalar", args.model)
        my_logging.info('save visualDL scalar log in: {}'.format(log_path))
        thread_train = DrawScalar(os.path.join(log_path, "train"), args.model)
        thread_val = DrawScalar(os.path.join(log_path, "val"), args.model)
        thread_train.start()
        thread_val.start()

        # print progressor
        with Progress("[progress.description]{task.description}{task.completed}/{task.total}",
                      BarColumn(),
                      "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeRemainingColumn(),
                      auto_refresh=False) as progress:
            task1 = progress.add_task("[red]epoch:", total=epochs)
            task2 = progress.add_task("[blue]train:", total=math.ceil(train_len/batch_size))
            task3 = progress.add_task("[yellow]validate:", total=math.ceil(val_len/batch_size))

            for i in range(args.start_epoch):
                progress.update(task1, advance=1, refresh=True)

            begin = time.time()
            for epoch in range(args.start_epoch, epochs):
                start = time.time()

                # train for one epoch
                train_loss, train_prec = train(
                    train_loader, model, criterion, epoch, optimizer)

                # train_loss_list.append(train_loss)
                # train_prec_list.append(train_prec)

                # evaluate on validation set
                val_loss, val_prec = validate(
                    val_loader, model, criterion, epoch)

                # val_loss_list.append(val_loss)
                # val_prec_list.append(val_prec)

                # remember best prec@1 and save checkpoint
                # is_best = True
                is_best = val_prec > best_prec
                best_prec = max(val_prec, best_prec)

                model_dict = model.state_dict()
                train_dict = {'epoch': epoch + 1, 
                              'model': args.model,
                              'config': model_config,
                              'best_prec': best_prec,
                }
                save_checkpoint(model_dict, train_dict, is_best, path=save_path)
                my_logging.debug('\n----------------------------------------------\n'
                            'Epoch: [{0}/{1}] Cost_Time: {2:.2f}s\t'
                            'current_lr: {3:.7f}\t'
                            'Training Loss {train_loss:.4f} \t'
                            'Training Prec {train_prec:.3f} \t'
                            'Validation Loss {val_loss:.4f} \t'
                            'Validation Prec {val_prec:.3f} \n'
                            '----------------------------------------------'
                            .format(epoch + 1, epochs, time.time()-start, optimizer.current_step_lr(),
                                    train_loss=train_loss, val_loss=val_loss, 
                                    train_prec=train_prec, val_prec=val_prec))

                results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                            train_prec=train_prec, val_prec=val_prec)
                results.save()

                # draw visualDL scalar
                thread_train.set_value(epoch, {'loss': train_loss, 'acc': train_prec})
                thread_val.set_value(epoch, {'loss': val_loss, 'acc': val_prec})

                # update progressor
                progress.update(task1, advance=1, refresh=True)

        my_logging.info('\n----------------------------------------------------------------\n'
                    'Whole Cost Time: {0:.2f}s      Best Validation Prec {1:.3f}\n'
                    '-----------------------------------------------------------------'.format(time.time()-begin, best_prec))
        
    #     epochs = list(range(args.epochs))
    #     draw2(epochs, train_loss_list, val_loss_list, train_prec_list, val_prec_list)

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()

    start = time.time()

    for i, data in enumerate(data_loader()):
        # measure data loading time
        data_time.update(time.time() - start)
        # load inputs and target
        x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, in_dim, input_size, input_size)
        y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, target_size)
        # convert numpy.ndarry to Tensor
        image = fluid.dygraph.to_variable(x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        # get model output
        output = model(image)

        loss = criterion(output, label)
        avg_loss = fluid.layers.mean(loss)
        losses.update(avg_loss.numpy()[0], loss.shape[0])

        if training:
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

        pred = fluid.layers.softmax(output)
        acc = fluid.layers.accuracy(pred, label)
        precisions.update(acc.numpy()[0], pred.shape[0])
        # measure elapsed time
        batch_time.update(time.time() - start)
        # update progressor
        if training:
            progress.update(task2, advance=1, refresh=True)
        else:
            progress.update(task3, advance=1, refresh=True)

    if not training:
        progress.update(task3, completed=0)
    else:
        progress.update(task2, completed=0)

    return losses.avg, precisions.avg

def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                    training=True, optimizer=optimizer)

def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                    training=False, optimizer=None)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Main KeyboardInterrupt....")
    thread_train.stop()
    thread_train.join()
    thread_val.stop()
    thread_val.join()
    sys.exit(1)
