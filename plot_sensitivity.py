# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sh
import os, random, time, sys
from operator import itemgetter
import json
import itertools
from logger import Logger
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


def get_sensitivity(experiment_id, return_loss=False):
    with open('experiments/{}.json'.format(experiment_id)) as f:
        exlog = json.load(f)

    x, y = [], []
    for run_id in exlog['runs']:
        try:
            logger = Logger.load(run_id)
        except ValueError:
            continue
        start_epoch = logger['args'].start_epoch
        if start_epoch == 0:
            start_epoch = logger.get('train')[0]['epoch']
        if return_loss:
            test_errors = [v['loss'] for v in logger.get('valid')]
        else:
            test_errors = [v['error'] for v in logger.get('valid')]

        m = min(1000,len(test_errors))
        test_error = np.median(test_errors[m-10:m])
        x += [start_epoch-1]
        y += [test_error]
    y = np.array(y)
    x = np.array(x)
    window_size = logger['args'].deficit_end - logger['args'].deficit_start

    logger = Logger.load(exlog['main'])
    xm = [v['epoch'] for v in logger.get('valid')]
    ym = [v['error'] for v in logger.get('valid')]
    base = np.median(ym[-10:])
    ym = savgol_filter(ym, 11, 1)

    return x, y, base, xm, ym, window_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('experiment', type=str,
                        help='experiment to plot')
    parser.add_argument('--format', '-f', default='pdf', type=str,
                        help='output a format')
    parser.add_argument('--name', '-n', default='plots/sensitivity.pdf', type=str)
    parser.add_argument('--title', '-t', default=None, type=str,
                        help='title of the plot')
    parser.add_argument('--max-y', '-y', default=None, type=float,
                        help='minimum value of y')
    parser.add_argument('--save', '-s', action='store_true',
                        help='save the plot')
    parser.add_argument('--hide-y', '-i', action='store_true',
                        help='hide y axis')
    args = parser.parse_args()

    x, y, base, xm, ym, _ = get_sensitivity(args.experiment)
    print(y)
    print(base)
    print(xm)

    plt.figure(figsize=(3.2,2.7))

    plt.plot(x, y-base, color='#C25054', marker='o',markersize=5, label='Decrease in Test Accuracy')
    # plt.plot(xm, ym-base, '--',color='#58A76B', label='Learning curve w/o deficit')

    if args.title is not None:
        plt.title(args.title)

    plt.xlabel('Window onset (epoch)')
    plt.ylabel('Decrease in Test Accuracy')
    if args.max_y is not None:
        plt.ylim(-1, args.max_y)
    else:
        plt.ylim(-1, max(y)+1.-base)
    # plt.xlim(0, max(x))

    # plt.axes().xaxis.set_major_locator(tck.MultipleLocator(base=20))
    # vals = plt.axes().get_yticks()
    # plt.axes().set_yticklabels(['{:.1f}%'.format(x) for x in vals])

    plt.tight_layout()
    plt.savefig(args.name, format=args.format, dpi=None, bbox_inches='tight' )

