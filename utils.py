# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import numpy as np
import os
from pathlib import Path


def get_parameter(model, parameter):
    result = []
    if hasattr(model, parameter):
        result.append(getattr(model, parameter))
    for l in model.children():
        result += get_parameter(l, parameter)
    return result


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100. - correct_k.mul_(100.0 / batch_size))
    return res


def set_norm(model, train=True):
    if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_norm(l, train=train)


def set_batchnorm_mode(model, train=True):
    if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_norm(l, train=train)


def call_on_model(model, name, *args, **kwargs):
    results = []
    if hasattr(model, name):
        results += [getattr(model, name)(*args, **kwargs)]
    for l in model.children():
        results += call_on_model(l, name, *args, **kwargs)
    return results


def set_parameter(model, parameter, value):
    if hasattr(model, parameter):
        setattr(model, parameter, value)
    for l in model.children():
        set_parameter(l, parameter, value)

# --- Start of getdominance() code
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2
    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)


def get_dominance(val_loader, model, criterion, print_epoch, save_dir, device, normalize_indep=True, zero_input=False,
                  plot_diff=True, filename=None, save_raw_fsv=True, title_prepend=None):
    """
    Computes the variance on units in a representation, when one view is varied, and the other is kept fixed.
    Mirrors the occular dominance diagrams found in neuroscience papers
    i.e, see: https://www.nobelprize.org/uploads/2018/06/wiesel-lecture.pdf.
    We use Welford's online algorithm to compute a running variance in an online manner
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    model.eval()

    # Statistics varying input_a and input_b sampled from p(a)p(b) [for normalization is normalize_indep=False]
    existingAggregate = (0, 0, 0)
    for i, (input_a, input_b, target) in enumerate(val_loader):
        inv_idx = torch.arange(input_b.size(0) - 1, -1, -1).long()
        input_a = input_a.to(device)
        input_b = input_b[inv_idx].to(device)
        outputs = model(input_a, input_b)
        value = model.layer4_out.data
        existingAggregate = update(existingAggregate, value)
    mean, var, _ = finalize(existingAggregate)

    # Statistics fixing input a
    existingAggregate = (0, 0, 0)
    for i, (_, input_b, target) in enumerate(val_loader):
        if zero_input:
            input_a = torch.zeros_like(input_a)
        input_a = input_a.to(device)
        input_b = input_b.to(device)
        outputs = model(input_a, input_b)
        value = model.layer4_out.data
        existingAggregate = update(existingAggregate, value)
    mean_a, var_a, _ = finalize(existingAggregate)

    # Statistics fixing input b
    existingAggregate = (0, 0, 0)
    for i, (input_a, _, target) in enumerate(val_loader):
        if zero_input:
            input_b = torch.zeros_like(input_b)
        input_a = input_a.to(device)
        input_b = input_b.to(device)
        outputs = model(input_a, input_b)
        value = model.layer4_out.data
        existingAggregate = update(existingAggregate, value)
    mean_b, var_b, _ = finalize(existingAggregate)

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(3.2, 2.7))

    # either normalize by sum of variance to test if this is what causes the four modes
    if normalize_indep:
        a = (var_a / (var_a + var_b)).reshape(-1).cpu()
        b = (var_b / (var_a + var_b)).reshape(-1).cpu()
    else:
        a = (var_a / var).sqrt().reshape(-1).cpu()
        b = (var_b / var).sqrt().reshape(-1).cpu()

    bins = np.linspace(-1.15, 1.15, 40)

    if plot_diff:
        plt.hist(a - b, bins, label='a', color='r')
    else:
        plt.hist(b, bins, label='a', color='r')

    # plt.title(f'{print_epoch} epoch {save_name[:-4]}')
    # Hacky before deadline but needed since plotting during training as well (at epochs 1, 21, 41 etcs).
    # Note we save the values for plot modification as well
    change_title = (title_prepend == 'Blur' and print_epoch % 10 == 0)
    title_save_epoch = print_epoch - 180 if change_title else print_epoch
    plt.title(f'{title_prepend} {title_save_epoch} epoch' if change_title else f'{print_epoch} epoch')
    plt.xlabel('Relative Source Variance')
    plt.ylabel('Number of Units')
    plt.yticks([])
    ax = plt.gca()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    output_dir = os.path.join('plots', save_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_filename = f'{print_epoch}_{filename}.pdf' if filename is not None else f'{print_epoch}_histogram.pdf'
    savepath = os.path.join(output_dir, save_filename)
    plt.savefig(savepath, format='pdf', dpi=None, bbox_inches='tight')

    # Saving FSV to help with future plotting
    if save_raw_fsv:
        fsv_log_output_dir = os.path.join(output_dir, 'fsv_logs')
        Path(fsv_log_output_dir).mkdir(parents=True, exist_ok=True)
        fsv_log_filename = f"{save_filename[:-4]}.npz" # changing .pdf to .npz
        savepath_fsv = os.path.join(fsv_log_output_dir, fsv_log_filename)
        np.savez(savepath_fsv, a=a, b=b)
