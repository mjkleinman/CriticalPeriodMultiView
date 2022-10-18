# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import random
import sys
import time

import sh
import torch
from tqdm import tqdm

from logger import Logger

parser = argparse.ArgumentParser(description='Critical period experiments')
parser.add_argument('-j', '--procs', default=1, type=int, metavar='N',
                    help='processes per GPU')
parser.add_argument('-l', '--lr', default=0.075, type=float,
                    help='processes per GPU')
parser.add_argument('--wd', default=0.0005, type=float,
                    help='processes per GPU')
parser.add_argument('-v', '--view-size', default=14, type=int,
                    help='processes per GPU')
parser.add_argument('--n-blocks', default=1, type=int,
                    help='number of blocks (depth)')
parser.add_argument('--no-augment', action='store_true',
                    help='augment dataset')
parser.add_argument('--diff-aug', action='store_true',
                    help='different augmention for cifar ind')
parser.add_argument('-c', '--cont', default=None, type=str,
                    help='resume given experiment')
parser.add_argument('--add', default=None, type=str,
                    help='add new data to experiment')
args = parser.parse_args()

gpu_procs = {i: [] for i in range(torch.cuda.device_count())}
command_queue = []
procs = []
logs = []


def l2s(l):
    return ' '.join(str(x) for x in l)


def get_random_name():
    return '{:06x}'.format(random.getrandbits(6 * 4))


def get_cuda_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def kill_procs():
    for p in procs:
        try:
            p.kill_group()
        except:
            pass
    sys.exit()


def wait_progressbar():
    pbar = [tqdm(total=100, position=i) for i in range(len(logs))]
    try:
        while not all(p.process.exit_code != None for p in procs):
            for i, l in enumerate(logs):
                try:
                    logger = Logger.load(l)
                    epoch = logger.get('train')[-1]['epoch']
                    start = logger['args'].start_epoch
                    end = logger['args'].schedule[-1]
                    pbar[i].total = end - start
                    pbar[i].update(epoch + 1 - pbar[i].n - start)
                except:
                    pass
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    for p in pbar:
        p.close()


def wait_procs():
    try:
        [p.wait() for p in procs]
    except KeyboardInterrupt:
        pass


def _gpu_done(p, success, exit_code):
    for k, v in gpu_procs.items():
        v[:] = [a for a in v if a.pid != p.pid]
    run_next()


def run_gpu(command, gpu):
    p = command(_env=get_cuda_env(gpu), _done=_gpu_done)
    gpu_procs[gpu].append(p)
    procs.append(p)
    return p


def run_next():
    free_gpu, l = min(gpu_procs.items(), key=lambda x: len(x[1]))
    if len(l) >= args.procs:
        return None
    if command_queue:
        command = command_queue.pop(0)
        # print "Running new on {}".format(free_gpu)
        return run_gpu(command, free_gpu)


def queue(command):
    log_name = get_random_name()
    command = command.bake(log_name=log_name, _out="outputs/{}".format(log_name))
    print("=> {}".format(command))
    command_queue.append(command)
    run_next()
    logs.append(log_name)
    return log_name


def continue_experiment(logs):
    raise NotImplemented('continue_experiment not implemented')


def add_to_experiment(experiment_logs):
    main_log = experiment_logs['main']
    pass


# def start_experiment():
#     critical = sh.Command("./main_independent.py")
#     critical = critical.bake(_bg=True, _bg_exc=False)
#     critical = critical.bake(lr=args.lr, slow=True, augment=not args.no_augment, weight_decay=args.wd,
#                              arch='sresnetmulti',
#                              view_size=args.view_size, diff_aug=args.diff_aug)
#     main_log = queue(critical.bake(schedule=181, save=True, deficit_start=-1, deficit_end=181))
#     for k in range(0, 181, 20):
#         queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
#                             resume='models/{}_{}.pth'.format(main_log, k)))

# def start_experiment():
#     critical = sh.Command("./main_independent.py")
#     critical = critical.bake(_bg=True, _bg_exc=False)
#     critical = critical.bake(lr=0.05, slow=True, augment=True, weight_decay=0.00025,
#                              arch='sresnetmulti',
#                              view_size=16, diff_aug=True)
#     main_log = queue(critical.bake(schedule=181, save=True, deficit_start=-1, deficit_end=181))
#     for k in range(0, 181, 20):
#         queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
#                             resume='models/{}_{}.pth'.format(main_log, k)))

# def start_experiment():
#     critical = sh.Command("./main_independent1.py")
#     critical = critical.bake(_bg=True, _bg_exc=False)
#     critical = critical.bake(lr=args.lr, slow=True, augment=True, weight_decay=0.0005,
#                              arch='sresnetmulti',
#                              view_size=16)
#     main_log = queue(critical.bake(schedule=181, save=True, deficit_start=-1, deficit_end=181))
#     for k in range(0, 181, 20):
#         queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
#                             resume='models/{}_{}.pth'.format(main_log, k)))
def start_experiment():
    critical = sh.Command("./main_independent_rhf.py")
    critical = critical.bake(_bg=True, _bg_exc=False)
    critical = critical.bake(lr=0.05, slow=True, augment=False, weight_decay=0,
                             arch='sresnetmulti',
                             view_size=16, diff_aug=False)
    main_log = queue(critical.bake(schedule=181, save=True, deficit_start=-1, deficit_end=181))
    for k in range(0, 181, 20):
        queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
                            resume='models/{}_{}.pth'.format(main_log, k)))

# def start_experiment():
#     critical = sh.Command("./main.py")
#     critical = critical.bake(_bg=True, _bg_exc=False)
#     critical = critical.bake(lr=0.075, slow=True, augment=True, weight_decay=0.0005,
#                              arch='sresnet', view_size=16)
#     main_log = queue(critical.bake(schedule=181, save=True, deficit='downsample', deficit_start=-1, deficit_end=181))
#     for k in range(0, 181, 20):
#         queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
#                             resume='models/{}_{}.pth'.format(main_log, k)))



# Nlayer
# Running for n_blocks=1,2,3
# def start_experiment():
#     critical = sh.Command("./main.py")
#     critical = critical.bake(_bg=True,_bg_exc=False)
#     critical = critical.bake(lr=0.001, slow=False, filters=1., augment=True, arch='nlayer', k=2, n_blocks=args.n_blocks,
#                              weight_decay=0.001, view_size=16)
#     main_log = queue(critical.bake(deficit='downsample', schedule=281, save=True))
#     for k in range(0,281,40):
#         queue(critical.bake(start_epoch=k+1,schedule=k+160, resume='models/{}_{}.pth'.format(main_log,k)))

# Rand-ZERO-Input for INFORMATION
# def start_experiment():
#     critical = sh.Command("./main.py")
#     critical = critical.bake(_bg=True, _bg_exc=False)
#     critical = critical.bake(lr=0.075, slow=True, augment=True, weight_decay=0.0005,
#                              arch='sresnet', view_size=16, is_rand_zero_input=True)
#     main_log = queue(critical.bake(schedule=181, save=True, deficit='downsample', deficit_start=-1, deficit_end=181))
#     for k in range(0, 181, 20):
#         queue(critical.bake(start_epoch=k + 1, schedule=k + 180, save_final=True,
#                             resume='models/{}_{}.pth'.format(main_log, k)))

if args.cont is not None:
    name = args.cont
    with open('experiments/{}.json'.format(name), 'w') as f:
        logs = json.load(f)['all']
    continue_experiment(logs)
if args.add is not None:
    name = args.add
    with open('experiments/{}.json'.format(name), 'w') as f:
        experiment_logs = json.load(f)
        logs = experiment_logs['all']
    add_to_experiment()
else:
    name = get_random_name()
    print("Starting experiment {}".format(name))
    start_experiment()
    code = []
    try:
        with open(__file__) as f:
            code = [l.rstrip('\n') for l in f]
    except:
        pass

with open('experiments/{}.json'.format(name), 'w') as f:
    json.dump({
        'all': logs,
        'main': logs[0],
        'runs': logs[1:],
        'code': code,
    }, f, indent=4, separators=(',', ': '))

print("")
print("====== {} ======".format(name))
print("Main:\t{}".format(logs[0]))
for i, l in enumerate(logs[1:]):
    print("{}\t{}".format('Logs:' if i == 0 else '', l))
print("====================")
print("")
wait_progressbar()

kill_procs()
