# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import argparse
import json

parser = argparse.ArgumentParser(description='Plotting dominance for experiment')
parser.add_argument('experiment', type=str,
                    help='experiment to plot')
parser.add_argument('--dominance_save_name', default='indep.pdf', type=str)
parser.add_argument('--view_size', default=14, type=int)
parser.add_argument('--arch', default='sresnet', type=str)
parser.add_argument('--format', '-f', default='png', type=str,
                    help='output a format')
parser.add_argument('--name', '-n', default='sresnet.pdf', type=str)
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
parser.add_argument('--plot_dominances', action='store_true',
                    help='hide y axis')
args = parser.parse_args()


with open('experiments/{}.json'.format(args.experiment)) as f:
    exlog = json.load(f)

print(exlog['runs'])


# Plot the checkpoints during training for the independent view
rc = subprocess.call(f"python plot_experiment.py {args.experiment} --save_dir {args.dominance_save_name} -f pdf -t Independent\ Pathway", shell=True)

if args.plot_dominances:
    main_log = exlog['main']
    for epoch in range(0, 181, 20):
        rc = subprocess.call(f"./main_independent.py --dominance --dominance_save_name {args.dominance_save_name} --dominance_file_name indep \
                         --arch {args.arch} --view-size {args.view_size} --resume /home/ubuntu/CriticalPeriodMultiView/models/{main_log}_{epoch}.pth",
                             shell=True)


    # Plot after resuming training without the deficit
    for log in exlog['runs']:
        rc = subprocess.call(f"./main_independent.py --dominance --dominance_save_name {args.dominance_save_name} --dominance_file_name resume \
                         --arch {args.arch} --view-size {args.view_size} --resume /home/ubuntu/CriticalPeriodMultiView/models/{log}.pth", shell=True)

