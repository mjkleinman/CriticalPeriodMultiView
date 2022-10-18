import numpy as np
import matplotlib.pyplot as plt
from utils import plot_matrix

from mpl_toolkits.axes_grid1 import ImageGrid

"""
From Fig 3 of Saxe et al.
"""


def extract_data(data):
    wrow0 = data['wrow0']
    wrow2 = data['wrow2']
    wrow6 = data['wrow6']
    return wrow0, wrow2, wrow6

use_shallow = True
filename_suffix = '_shallow' if use_shallow else '_deep'
data_perturbed = np.load(f'logs/weight_mnorms_perturbed{filename_suffix}.npz')
data_original = np.load(f'logs/weight_mnorms_original{filename_suffix}.npz')

w_orig_0, w_orig_2, w_orig_6 = extract_data(data_original)
w_pet_0, w_pet_2, w_pet_6 = extract_data(data_perturbed)

plt.title(f'Norm of properties weights ({filename_suffix[1:]})')
plt.plot(w_orig_0, color='blue', label='grow (orig)')
plt.plot(w_orig_2, color='green', label='roots (orig)')
plt.plot(w_orig_6, color='red', label='petals (orig)')
plt.plot(w_pet_0, color='blue', linestyle='dashed', label='grow (new)')
plt.plot(w_pet_2, color='green', linestyle='dashed', label='roots (new)')
plt.plot(w_pet_6, color='red', linestyle='dashed', label='petals (new)')
plt.legend()
plt.savefig(f'plots/property_norm_diff{filename_suffix}.pdf', format='pdf', dpi=None, bbox_inches='tight')


