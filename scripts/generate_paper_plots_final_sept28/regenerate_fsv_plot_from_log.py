# load saved logs
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# should pass these in:
save_dirs = ['view16_lr0.05_wd0_sresnetmulti_ind_RHF_noaug_dcd6cd',
             'view16_lr0.05_sresnetmulti_ind_RHF_diffaug_7b58a9',
             'view16_lr0.05_sresnetmulti_ind_RHF_noaug_b2830e',
             'view16_lr0.05_wd0.00025_sresnetmulti_ind_RHF_diffaug_e5eb36']

for save_dir in save_dirs:
    data_dir = os.path.join('plots', save_dir, 'fsv_logs')
    # print_epoch = 180
    for print_epoch in range(180, 361, 20):
        filename = 'resume'
        fsv_filename = os.path.join(data_dir, f'{print_epoch}_{filename}.npz')
        data_fsv = np.load(fsv_filename)
        a = data_fsv['a']
        b = data_fsv['b']

        fig = plt.figure(figsize=(3.2, 2.7))
        bins = np.linspace(-1.15, 1.15, 40)

        plot_diff = True
        if plot_diff:
            plt.hist(a - b, bins, label='a', color='r')
        else:
            plt.hist(b, bins, label='a', color='r')

        # plt.title(f'{print_epoch} epoch {save_name[:-4]}')
        plt.title(f'Indep. Path: {print_epoch - 180} epochs')
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
        plt.close()