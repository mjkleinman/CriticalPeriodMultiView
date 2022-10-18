#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from logger import Logger
from models import SResNet18, SResNet18MultiHead, DoubleNLayerDisciminator
from utils import get_dominance, get_error, AverageMeter, set_batchnorm_mode

parser = argparse.ArgumentParser(description='Critical period experiments')
parser.add_argument('--arch', default='sresnet', type=str,
                    help='architecture to use')
parser.add_argument('--view-size', default=14, type=int,
                    metavar='N', help='mini-batch size (default: 14)')
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
parser.add_argument('--dominance', action='store_true',
                    help='compute dominance')
parser.add_argument('--is-rand-zero-input', action='store_true',
                    help='compute dominance')
parser.add_argument('--dominance_save_name', default='histogram.pdf', type=str,
                    help='dominance save name')
parser.add_argument('--dominance_file_name', default=None, type=str,
                    help='dominance file name')
parser.add_argument('--dropout', action='store_true',
                    help='use dropout')
parser.add_argument('--no-bn', dest='batch_norm', action='store_false',
                    help='use dropout')
parser.add_argument('--show', dest='show', action='store_true',
                    help='show images')
parser.add_argument('--estimate-var', action='store_true',
                    help='estimate gradient variance')
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

def train(data_loader, model, criterion, optimizer, epoch, train=True):
    """
    Main train function
    Args:
        train: specify whether in train or eval mode (by setting train=True)
    """
    kls = AverageMeter()
    losses = AverageMeter()
    losses_a = AverageMeter()
    losses_b = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    for i, (input_a, input_b, target) in enumerate(data_loader):
        input_a = input_a.to(device)
        input_b = input_b.to(device)
        target = target.to(device)

        # compute output
        output, _ = model(input_a, input_b)
        loss = criterion(output, target)
        acc, = get_error(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))
        accuracies.update(acc.item(), target.size(0))

        # input to each pathways to store logging for each view decoding (useful only when zeroing out)
        # epoch % 10 == 1 is useful when resuming
        if (epoch % 10 == 0 or epoch % 10 == 1) and args.is_rand_zero_input:
            with torch.no_grad():
                output_b, _ = model(torch.zeros_like(input_a).to(device), input_b)
                loss_b = criterion(output_b, target)
                losses_b.update(loss_b.item(), target.size(0))
                output_a, _ = model(input_a, torch.zeros_like(input_a).to(device))
                loss_a = criterion(output_a, target)
                losses_a.update(loss_a.item(), target.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(losses.avg)
    if epoch % 10 == 0:
        print(f'losses a avg: {losses_a.avg}')
        print(f'losses b avg: {losses_b.avg}')
    print('[{}] Epoch: [{epoch}] lr: {lr:.4f} Loss {loss.avg:.3f} Lz: {lz.avg:.3f} Error: {acc.avg:.2f}'
          .format('train' if train else 'valid', epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss=losses, lz=kls, acc=accuracies))

    logger.append('train' if train else 'valid', epoch=epoch, loss=losses.avg, lz=kls.avg, error=accuracies.avg, lr=optimizer.param_groups[0]['lr'], loss_a=losses_a.avg, loss_b=losses_b.avg)


def validate(val_loader, model, criterion, optimizer, epoch):
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
        outputs = model(input_var_a, input_var_b)


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

    # -- 0. Logging
    logger = Logger(index=args.log_name)
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index + '.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index + '_{}.pth')

    print ("[Logging in {}]".format(logger.index))

    # -- 1. Architecture: sresnetmulti has multiple heads to read out outputs
    n_channels = 3
    if args.arch == 'sresnet':
        model = SResNet18(n_channels=n_channels).cuda()
    elif args.arch == 'sresnetmulti':
        model = SResNet18MultiHead(n_channels=n_channels).cuda()
    elif args.arch == 'nlayer':
        model = DoubleNLayerDisciminator(n_channels=n_channels).cuda()
    else:
        raise ValueError("Architecture {} not valid.".format(args.arch))

    if args.show:
        print (sum([p.nelement() for p in model.parameters()]))
        print ([p.size() for p in model.parameters()])

    # -- 2. Dataloader with deficit.
    from cifar_data import get_cifar_loaders_s
    get_dataset_loaders = get_cifar_loaders_s

    datasets = {
        'deficit': get_dataset_loaders(batch_size=args.batch_size, workers=args.workers,
                                       augment=args.augment, deficit=args.deficit, is_rand_zero_input=args.is_rand_zero_input,
                                       view_size=args.view_size),
        'normal': get_dataset_loaders(batch_size=args.batch_size, workers=args.workers,
                                      augment=args.augment, deficit=[], is_rand_zero_input=args.is_rand_zero_input,
                                      view_size=args.view_size)
    }

    # -- 3. Optimizer and Loss
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

    # Optional Resume. Helpful for experiments where we resume from saved checkpoint for with/without deficits
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

    # Get dominance plots used in figures
    if args.dominance:
        train_loader, val_loader = datasets['normal'] # Use normal dataset (without deficit)
        dry_run(train_loader, model)
        get_dominance(val_loader, model, criterion, args.start_epoch, args.dominance_save_name, device=device,
                      filename=args.dominance_file_name, title_prepend='Blur')
        sys.exit()

    # -- 5. Main training loop
    try:
        for epoch in range(args.start_epoch, args.schedule[-1]):
            # Save at beginning so deficit isn't applied for one epoch and resumed model is random
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if args.save and epoch % args.save_every == 0:
                save_checkpoint(state, step=True)

            adjust_learning_rate(optimizer, epoch, args.schedule[:-1])
            if args.deficit_start is None or args.deficit_end is None:
                train_loader, val_loader = datasets['deficit']
            elif epoch > args.deficit_start and epoch < args.deficit_end:
                train_loader, val_loader = datasets['deficit']
            else:
                train_loader, val_loader = datasets['normal']

            loss = train(train_loader, model, criterion, optimizer, epoch)

            dry_run(train_loader, model)
            # validate(val_loader, model, criterion, epoch)
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
