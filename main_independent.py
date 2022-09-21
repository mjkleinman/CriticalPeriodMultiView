#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import torch.nn.parallel
import torch.optim
import torch.utils.data

from logger import Logger
from models import *

parser = argparse.ArgumentParser(description='Critical period experiments')
parser.add_argument('--arch', default='sresnet', type=str,
                    help='architecture to use. Choose from sresnet or sresnetmulti')
parser.add_argument('--view-size', default=14, type=int,
                    metavar='N', help='mini-batch size (default: 14)')
parser.add_argument('--diff-aug', action='store_true',
                    help='different augmentation for cifar ind loader')
parser.add_argument('--log-name', default=None, type=str,
                    help='index for the log file')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--schedule', nargs='+', default=[160], type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--beta', default=1.0, type=float,
                    help='value of beta (default: 1.0)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-f', '--filters', default=.25, type=float,
                    help='percentage of filters to use')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0.)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--name', default='model', type=str,
                    help='name of the run')
parser.add_argument('--decay', default=0.97, type=float,
                    help='Learning rate exponential decay')
parser.add_argument('--slow', dest='slow', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save', dest='save', action='store_true',
                    help='save the model every x epochs')
parser.add_argument('--save-final', action='store_true',
                    help='save last version of the model')
parser.add_argument('--save-every', default=10, type=int,
                    help='interval fro saving')
parser.add_argument('-o', '--optimizer', default='sgd', type=str,
                    help='Optimizer to use')
parser.add_argument('--augment', action='store_true',
                    help='augment dataset')
parser.add_argument('--dropout', action='store_true',
                    help='use dropout')
parser.add_argument('--no-bn', dest='batch_norm', action='store_false',
                    help='use dropout')
parser.add_argument('--show', dest='show', action='store_true',
                    help='show images')
parser.add_argument('--show-filters', action='store_true',
                    help='show first layer filters')
parser.add_argument('--deficit', '-d', nargs='+', default=[], type=str,
                    help='Deficit to introduce')
parser.add_argument('--deficit-start', default=None, type=int,
                    help='start epoch for deficit')
parser.add_argument('--deficit-end', default=None, type=int,
                    help='end epoch for deficit')
parser.add_argument('-k', default=2, type=int,
                    help='layer multiplier')
parser.add_argument('--n-blocks', default=2, type=int,
                    help='number of layers in the network')
args = parser.parse_args()

n_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unison_shuffled_copies(a, b, target):
    """
    Return a permutation of the input views (a,b) and target. Helper function for randomly sampling from each view
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], target[p]


def train(data_loader, model, criterion, optimizer, epoch, train=True):
    """
    Main training script allowing for sampling target corresponding to independent views during a deficit
    """
    kls = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    losses_a = AverageMeter()
    losses_b = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    for i, (input_a, input_b, target) in enumerate(data_loader):
        input_a = input_a.to(device)
        input_b = input_b.to(device)
        target = target.to(device)

        # -- Ensure target should be either the target from input_a or input_b during deficit with prob 0.5
        # TODO: Maybe better to do directly in a dataloader? should probably be done cleaner
        if args.deficit_start is not None and args.deficit_end is not None:
            if train and (epoch > args.deficit_start and epoch < args.deficit_end):
                _, input_b_shuffled, target_shuffled = unison_shuffled_copies(input_a, input_b, target) # Make sure input_b is independent from input_a by performing a permutation
                random_indices = [i for i in range(len(target)) if np.random.rand(1) < 0.5]
                target[random_indices] = target_shuffled[random_indices] # target either corresponds to input_a or input_b with prob 0.5
                input_b = input_b_shuffled

        output_both, output_a, output_b = model(input_a, input_b) # recall sresnetmulti has three heads
        if (args.deficit_start is not None and args.deficit_end is not None) and train and (epoch > args.deficit_start and epoch < args.deficit_end):
            output = output_a if np.random.rand(1) < 0.5 else output_b # sample output from multi-head
            # loss_a = criterion(output_a, target)
            # loss_b = criterion(output_b, target_shuffled)
            # loss = 0.5 * (loss_a + loss_b)
        else:
            output = output_both

        loss = criterion(output, target)
        acc, = get_error(output, target)
        losses.update(loss.item(), target.size(0))
        accuracies.update(acc.item(), target.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # -- Logging
    print('[{}] Epoch: [{epoch}] lr: {lr:.4f} Loss {loss.avg:.3f} Lz: {lz.avg:.3f} Error: {acc.avg:.2f}'.format(
        'train' if train else 'valid', epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss=losses, lz=kls,
        acc=accuracies))

    logger.append('train' if train else 'valid', epoch=epoch, loss=losses.avg, lz=kls.avg, error=accuracies.avg, lr=optimizer.param_groups[0]['lr'], loss_a=losses_a.avg, loss_b=losses_b.avg)


def validate(val_loader, model, criterion, optimizer, epoch, label=''):
    """Calls train(), but sets train=False"""
    train(val_loader, model, criterion, optimizer, epoch, train=False)


def dry_run(train_loader, model, train=False):
    if train:
        model.train()
    else:
        model.eval()
    set_batchnorm_mode(model, train=True)
    for i, (input_a, input_b, target) in enumerate(train_loader):
        target = target.cuda()
        input_var_a = torch.autograd.Variable(input_a).cuda()
        input_var_b = torch.autograd.Variable(input_b).cuda()
        output_both, _, _ = model(input_var_a, input_var_b)


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    return torch.mean(torch.sum(- softmax(soft_targets) * logsoftmax(pred), 1))


def get_targets(input, model):
    model.eval()
    output, _ = model(input)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().detach()
    return (pred)


def save_checkpoint(state, step=True):
    if step:
        epoch = state['epoch']
        target_file = logger['checkpoint_step'].format(epoch)
    else:
        target_file = logger['checkpoint']
    print ("Saving {}".format(target_file))
    torch.save(state, target_file)


def adjust_learning_rate(optimizer, epoch, schedule):
    """
    args.slow allows exponentially decaying learning rate
    """
    if not args.slow:
        lr = args.lr * (0.1 ** np.less(schedule, epoch).sum())
    else:
        lr = args.lr * args.decay**epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    logger = Logger(index=args.log_name)
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index + '.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index + '_{}.pth')

    print ("[Logging in {}]".format(logger.index))

    n_channels=3
    if args.arch == 'resnet':
        model = ResNet18(n_channels=n_channels).cuda()
    elif args.arch == 'sresnet':
        model = SResNet18(n_channels=n_channels).cuda()
    elif args.arch == 'sresnetmulti':
        model = SResNet18MultiHead(n_channels=n_channels).cuda()
    else:
        raise ValueError("Architecture {} not valid.".format(args.arch))

    if args.show:
        print (sum([p.nelement() for p in model.parameters()]))
        print ([p.size() for p in model.parameters()])

    # define loss function (criterion) and optimizer
    from cifar_data import get_cifar_loaders_ind #
    get_dataset_loaders = get_cifar_loaders_ind
    datasets = {
        'deficit': get_dataset_loaders(batch_size=args.batch_size, workers=args.workers,
                                       augment=args.augment, deficit=args.deficit, view_size=args.view_size, is_diff_aug=args.diff_aug),
        'normal': get_dataset_loaders(batch_size=args.batch_size, workers=args.workers,
                                      augment=args.augment, deficit=[], view_size=args.view_size, is_diff_aug=args.diff_aug)
    }
    criterion = nn.CrossEntropyLoss().cuda()
    parameters = model.parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer {} not valid.".format(args.optimizer))

    if args.resume:
        checkpoint_file = args.resume
        time.sleep(10)
        if not os.path.isfile(checkpoint_file):
            print ("=== waiting for checkpoint to exist ===")
            try:
                while not os.path.isfile(checkpoint_file):
                    time.sleep(10)
            except KeyboardInterrupt:
                print ("=== waiting stopped by user ===")
                import sys
                sys.exit()
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

    cudnn.benchmark = True

    try:
        for epoch in range(args.start_epoch, args.schedule[-1]):
            adjust_learning_rate(optimizer, epoch, args.schedule[:-1])
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if args.save and epoch % args.save_every == 0:
                save_checkpoint(state, step=True)

            train_loader, val_loader = datasets['normal']
            loss = train(train_loader, model, criterion, optimizer, epoch)
            validate(val_loader, model, criterion, optimizer, epoch)


        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        if args.save_final:
            save_checkpoint(state, step=False)

        logger['finished'] = True
        # dry_run(train_loader, model)
        # validate(val_loader, model, criterion, epoch)
    except KeyboardInterrupt:
        print ("Run interrupted")
        logger.append('interrupt', epoch=epoch)
    print ("[Logs in {}]".format(logger.index))
