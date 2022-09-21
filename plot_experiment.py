# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('experiment', type=str,
                    help='experiment to plot')
parser.add_argument('--format', '-f', default='png', type=str,
                    help='output a format')
parser.add_argument('--name', '-n', default='sresnet.pdf', type=str)
parser.add_argument('--save_dir', default=None, type=str)
parser.add_argument('--title', '-t', default=None, type=str,
                    help='title of the plot')
parser.add_argument('--min-y', '-y', default=None, type=float,
                    help='minimum value of y')
parser.add_argument('--max-x', '-x', default=None, type=float,
                    help='maximum value of y')
parser.add_argument('--save', '-s', action='store_true',
                    help='save the plot')
parser.add_argument('--hide-y', '-i', action='store_true',
                    help='hide y axis')
args = parser.parse_args()

with open('experiments/{}.json'.format(args.experiment)) as f:
    exlog = json.load(f)

print(exlog['runs'])
plt.figure(figsize=(3.2, 2.7))

x, y = [], []
for run_id in exlog['runs']:
    try:
        logger = Logger.load(run_id)
    except ValueError:
        continue
    start_epoch = logger['args'].start_epoch
    # print(start_epoch)
    if start_epoch == 0:
        start_epoch = logger.get('train')[0]['epoch']
    test_errors = [v['error'] for v in logger.get('valid')]
    test_error = np.mean(test_errors[-5:])
    x += [start_epoch - 1]
    y += [test_error]

# print x, y
print("Y")
print(y)

if x[-1] < 200:
    plt.axes().xaxis.set_major_locator(tck.MultipleLocator(base=20))
else:
    plt.axes().xaxis.set_major_locator(tck.MultipleLocator(base=80))

y = 100 - np.array(y)
print(y)
plt.plot(x, y, marker='o', markersize=5, label='Final test accuracy w/ deficit')

# plt.plot(x,y, '--',color='g', label='Final test accuracy w/ deficit')
if args.min_y is not None:
    plt.ylim(args.min_y, None)
else:
    plt.ylim(min(y) - 1., None)
# plt.ylim(98,None)
if args.max_x is not None:
    plt.xlim(0, args.max_x)
else:
    plt.xlim(0, max(x))
# plt.xlim(0,180)

try:
    logger = Logger.load(exlog['runs'][0])
    x = [v['epoch'] for v in logger.get('valid')]
    y = [v['error'] for v in logger.get('valid')]
    y = 100. - savgol_filter(y, 11, 1)
    plt.plot(x, y, '--', label='Test accuracy during learning w/o deficit')
    # plt.plot(x,y,label='Test accuracy during learning w/o deficit')
except ValueError:
    pass

# vals = plt.axes().get_yticks()
# plt.axes().set_yticklabels(['{:.1f}%'.format(x) for x in vals])

plt.xlabel('Deficit removal (epoch)')
plt.ylabel('Test accuracy')
if args.title is not None:
    plt.title(args.title)
if args.hide_y:
    plt.gca().yaxis.label.set_visible(False)

plt.tight_layout()

output_dir = os.path.join('plots', args.save_dir) if args.save_dir is not None else 'plots'
print(output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)
save_filename = args.name
savepath = os.path.join(output_dir, save_filename)
plt.savefig(savepath, format=args.format, dpi=None, bbox_inches='tight')
