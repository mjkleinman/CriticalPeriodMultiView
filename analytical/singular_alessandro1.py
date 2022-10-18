import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_empirical_sigma_yx(sigma_yx, num_samples, y_noise=0.1, seed=None):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(num_samples, sigma_yx.shape[0]))
    y = np.matmul(sigma_yx.T, x.T).T + y_noise * rng.normal(size=(num_samples, sigma_yx.shape[1]))
    return np.matmul(x.T, y) / x.shape[0]


w = []
w_shallow = []
remove_col = 2
use_empirical = False
for use_perturbed in [False, True]:
    # Sigma_YX = np.array([
    #     [1, 0, 1, 0, 0, 0],
    #     [1, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1]
    # ])
    Sigma_YX = np.array([
        [1, 0, 3, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 3, 0, 0, 0, 0, 1]
        ])
    nproperties, nsensors = Sigma_YX.shape
    if use_empirical:
        Sigma_YX = get_empirical_sigma_yx(Sigma_YX, 20, seed=1)
    if use_perturbed:
        Sigma_YX = np.delete(Sigma_YX, remove_col, axis=1)



    # Sigma_YX = Sigma_YX/np.linalg.norm(Sigma_YX, axis=1, keepdims=True)
    ncols = Sigma_YX.shape[1]
    u, s, vh = np.linalg.svd(Sigma_YX)
    tau = 0.1
    a0 = 0.0001
    num_pts = 1000
    t = np.linspace(0, 1 / tau, num_pts)
    t, s = t.reshape(-1, 1), s.reshape(1, -1)
    a = s * np.exp(2 * s * t) / (np.exp(2 * s * t) - 1 + s / a0)
    b = s * (1 - np.exp(-t)) + a0 * np.exp(-t)
    res = (u[None, :, :s.shape[1]] * a[:, None, :]) @ vh[:s.shape[1]]
    res_shallow = (u[None, :, :s.shape[1]] * b[:, None, :]) @ vh[:s.shape[1]]

    if use_perturbed:
        res = np.insert(res, remove_col, 0, axis=2)
        res_shallow = np.insert(res_shallow, remove_col, 0, axis=2)
    w.append(res_shallow)

fig, axes = plt.subplots(1, nproperties, figsize=(nproperties * 4, 4))
for task, ax in enumerate(axes):
    for i in range(len(w)):
        for j in range(nsensors):
            ax.plot(w[i][:, task, j], linestyle='-' if i == 0 else '--', color=cm.tab10(j), label=str(j))
            ax.set_title(f'Property {task}')
            if task == 0:
                ax.set_ylabel('Weight during training', fontsize=13)
# plt.legend()
plt.savefig(f'plots/weight_training_shallow.pdf', format='pdf', dpi=None, bbox_inches='tight')