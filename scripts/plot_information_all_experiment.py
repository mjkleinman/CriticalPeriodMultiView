# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import argparse
import json

parser = argparse.ArgumentParser(description='Plotting dominance for experiment')
parser.add_argument('experiment', type=str,
                    help='experiment to plot')
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
args = parser.parse_args()




with open('experiments/{}.json'.format(args.experiment)) as f:
    exlog = json.load(f)

print(exlog['runs'])

for i, log in enumerate(exlog['runs']): # doesn't include the main run
    rc = subprocess.call(f"python plot_information_all.py {log} -t \"Resume {i * 20}\" --save_suffix {i * 20}", shell=True)
