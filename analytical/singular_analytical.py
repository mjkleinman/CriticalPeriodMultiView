import numpy as np
import matplotlib.pyplot as plt
from utils import plot_matrix

from mpl_toolkits.axes_grid1 import ImageGrid

"""
From Fig 3 of Saxe et al. A mathematical theory of semantic development in deep neural networks.
"""

use_perturbed = False
use_shallow = False

suffix_shallow = '_shallow' if use_shallow else '_deep'
suffix_perturbed = '_perturbed' if use_perturbed else '_original'

# Perturbed
Sigma_YX_Perturbed = 0.7 * np.array([
                [1, 1, 1, 1],
                [1, 1, -1, -1],
                [-1, -1, 1, 1],
                [1, 0, 0, -1],
                [0, 1, -1, 0],
                [0, -1, 1, 0],
                [-1, 0, 0, 1]
                ])
# Original
Sigma_YX_Original = 0.7 * np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

Sigma_YX = Sigma_YX_Perturbed if use_perturbed else Sigma_YX_Original
Sigma_YX = Sigma_YX/np.linalg.norm(Sigma_YX, axis=1, keepdims=True)
ncols = Sigma_YX.shape[1]
u, s, vh = np.linalg.svd(Sigma_YX)
fig = plt.figure(figsize=(20, 10))
axes = fig.subplots(nrows=1, ncols=4)
plot_matrix(Sigma_YX, axes[0], title='Sigma_YX')
plot_matrix(np.diag(s), axes[2], title='A(t)')
plot_matrix(u[:, :Sigma_YX.shape[1]], axes[1], title='U')
plot_matrix(vh, axes[3], title='V^T', plot_cbar=True, fig=fig)
plt.savefig(f'plots/svd{suffix_perturbed}{suffix_shallow}.pdf', format='pdf', dpi=None, bbox_inches='tight')
plt.clf()
print(f'Singular values are: {s}')

# Plotting effective singular values during trainig, s taken from SVD above
tau = 0.1
a0 = 0.0001
num_pts = 1000
a = np.zeros((num_pts, len(s)))
for t in range(0, num_pts, 1):
    t_scaled = (t/num_pts) / tau
    if use_shallow:
        a[t, :] = s * (1 - np.exp(- t_scaled)) + a0 * np.exp(-t_scaled)
    else:
        a[t, :] = s * np.exp(2 * s * t_scaled) / (np.exp(2 * s * t_scaled) - 1 + s/a0)
plt.title('Singular Values')
plt.plot(a)
plt.savefig(f'plots/singular_vals{suffix_perturbed}{suffix_shallow}.pdf', format='pdf', dpi=None, bbox_inches='tight')
plt.clf()

# Plot SVD at particular epoch during trainig
epoch = 400
print(f'Effective singular values are: {a[epoch, :]}')
fig = plt.figure(figsize=(20, 10))
axes = fig.subplots(nrows=1, ncols=4)
s = a[epoch, :]
emp_sigma = u[:, :ncols] @ np.diag(s) @ vh
plot_matrix(emp_sigma, axes[0], title='Hat_Sigma_YX')
plot_matrix(np.diag(s), axes[2], title='A(t)')
plot_matrix(u[:, :ncols], axes[1], title='U')
plot_matrix(vh, axes[3], title='V^T', plot_cbar=True, fig=fig)
plt.savefig(f'plots/eff_svd_ep{epoch}{suffix_perturbed}{suffix_shallow}.pdf', format='pdf', dpi=None, bbox_inches='tight')
plt.clf()
print(np.linalg.norm(emp_sigma[0]))

# Store the norm for rows 0, 2, 6
wrow0 = np.zeros((num_pts,))
wrow2 = np.zeros((num_pts,))
wrow6 = np.zeros((num_pts,))
for epoch in range(num_pts):
    s = a[epoch, :]
    emp_sigma = u[:, :ncols] @ np.diag(s) @ vh
    wrow0[epoch] = np.linalg.norm(emp_sigma[0])
    wrow6[epoch] = np.linalg.norm(emp_sigma[6])
    wrow2[epoch] = np.linalg.norm(emp_sigma[2])

if use_perturbed:
    np.savez(f'logs/weight_mnorms{suffix_perturbed}{suffix_shallow}.npz', wrow0=wrow0, wrow2=wrow2, wrow6=wrow6)
else:
    np.savez(f'logs/weight_mnorms{suffix_perturbed}{suffix_shallow}.npz', wrow0=wrow0, wrow2=wrow2, wrow6=wrow6)


