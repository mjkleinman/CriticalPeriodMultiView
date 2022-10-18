import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import plot_matrix
import matplotlib.cm as cm


Sigma_YX = np.array([
    [1, 0, 3, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 3, 0, 0, 0, 0, 1]
    ])

fig = plt.figure(figsize=(3.2, 2.7))
fig, ax = plt.subplots()
plot_matrix(Sigma_YX, ax, title=r'$\Sigma^{YX}$', plot_cbar=True, fig=fig)
ax.set_xlabel('Sensor', fontsize=16)
ax.set_ylabel('Task', fontsize=16)
x, y, w, h = 1.5, -0.5, 1, 5
ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=cm.tab10(2), lw=3, clip_on=False))
plt.savefig('plots/matrix_perturbed.pdf', format='pdf', dpi=None, bbox_inches='tight')
plt.show()