# FSV distribution
# import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_matrix(matrix_to_plot, ax_plot, title, plot_cbar=False, fig=None):
    im = ax_plot.imshow(matrix_to_plot, cmap='bwr', vmin=-np.max(matrix_to_plot), vmax=np.max(matrix_to_plot)) #used to be symmertric with cmap 'bwr'
    ax_plot.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Major ticks
    ax_plot.set_xticks(np.arange(0, matrix_to_plot.shape[1], 1))
    ax_plot.set_yticks(np.arange(0, matrix_to_plot.shape[0], 1))

    # Labels for major ticks
    ax_plot.set_xticklabels(np.arange(1, matrix_to_plot.shape[1] + 1, 1))
    ax_plot.set_yticklabels(np.arange(1, matrix_to_plot.shape[0] + 1, 1))

    # Minor ticks
    ax_plot.set_xticks(np.arange(-.5, matrix_to_plot.shape[1], 1), minor=True)
    ax_plot.set_yticks(np.arange(-.5, matrix_to_plot.shape[0], 1), minor=True)
    ax_plot.set_title(title, fontsize=20)

    if plot_cbar:
        divider = make_axes_locatable(ax_plot)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


