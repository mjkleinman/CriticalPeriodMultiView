# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import sys
# import torch
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
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('experiment', nargs='+', type=str,
                    help='experiment to plot')
parser.add_argument('--start-epoch', default=9, type=int)
parser.add_argument('--save_suffix', default='', type=str)
parser.add_argument('--format', '-f', default='pdf', type=str,
                    help='output a format')
parser.add_argument('--title', '-t', default='Information Through Training', type=str,
                    help='title of the plot')
parser.add_argument('--min-y', '-y', default=None, type=float,
                    help='minimum value of y')
parser.add_argument('--max-x', '-x', default=None, type=float,
                    help='maximum value of y')
parser.add_argument('--save', '-s', action='store_true',
                    help='save the plot')
parser.add_argument('--hide-y', '-i', action='store_true',
                    help='hide y axis')
parser.add_argument('--normalize', '-n', action='store_true',
                    help='normalize information by number of connections')
parser.add_argument('--skip', '-k', action='store_true',
                    help='skip plotting the first layer')
args = parser.parse_args()


dataframes, deficit_ends = [], []

fig = plt.figure(figsize=(4., 3.))
ax = fig.add_subplot(111)

for experiment in args.experiment:
    logger = Logger.load(experiment)

    df = pd.DataFrame.from_dict({
        'epoch': [v['epoch'] for v in logger.get('valid')],
        'loss': [v['loss'] for v in logger.get('valid')],
        'loss_a': [v['loss_a'] for v in logger.get('valid')],
        'loss_b': [v['loss_b'] for v in logger.get('valid')], })
    # df_t = pd.DataFrame.from_dict({
    #     'epoch': [v['epoch'] for v in logger.get('trace')],
    #     'trace': [v['trace'] for v in logger.get('trace')],})
    dataframes += [df]
    deficit_ends += [logger['args'].deficit_end if logger['args'].deficit_end is not None else logger['args'].schedule[-1]]


lines = []

cmap = plt.get_cmap('Blues')
start_epoch = args.start_epoch
end_epoch = start_epoch + 171 # 180 is the number of epochs to continue
for i, (deficit_end, df) in enumerate(zip(deficit_ends, dataframes)):
    color = cmap(float(i + 3) / (len(dataframes) + 3))
    y_a = 3.32 - (df['loss_a'] * 1.44)
    y_b = 3.32 - (df['loss_b'] * 1.44)
    y = 3.32 - (df['loss'] * 1.44)
    lines += ax.plot(df['epoch'][start_epoch:end_epoch:10], y[start_epoch:end_epoch:10], color='black', label='both inputs')
    lines += ax.plot(df['epoch'][start_epoch:end_epoch:10], y_a[start_epoch:end_epoch:10], color='red', label='input_a')
    lines += ax.plot(df['epoch'][start_epoch:end_epoch:10], y_b[start_epoch:end_epoch:10], color='orange', label='input_b')

    ax.axvline(x=deficit_end, linestyle=':', color=color)

plt.xlim(0, df['epoch'].max())


plt.title(args.title)
plt.xlabel('Epoch')
ax.set_ylabel('Usable info (bits)')

if args.title is not None:
    plt.title(args.title)
if args.hide_y:
    plt.gca().yaxis.label.set_visible(False)


labs = [l.get_label() for l in lines]
plt.legend(lines, labs, loc=1, prop={'size': 9})

plt.tight_layout()
output=f'plots/information_all_{args.save_suffix}.pdf'
plt.savefig(output, format=args.format, dpi=None, bbox_inches='tight')

