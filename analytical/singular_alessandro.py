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
remove_row = 2
use_empirical = False
for use_perturbed in [False, True]:
    Sigma_YX = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    # Sigma_YX = np.array([
    #         [1, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 0],
    #         [3, 0, 0, 0, 3],
    #         [1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #         [0, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1],
    #         ])
    nfeatures, nclasses = Sigma_YX.shape
    if use_empirical:
        Sigma_YX = get_empirical_sigma_yx(Sigma_YX, 20, seed=1)
    if use_perturbed:
        Sigma_YX = np.delete(Sigma_YX, remove_row, axis=0)

    # Sigma_YX = Sigma_YX/np.linalg.norm(Sigma_YX, axis=1, keepdims=True)
    u, s, vh = np.linalg.svd(Sigma_YX)
    tau = 0.1
    a0 = 0.0001
    num_pts = 1000
    t = np.linspace(0, 1 / tau, num_pts)
    t, s = t.reshape(-1, 1), s.reshape(1, -1)
    a = s * np.exp(2 * s * t) / (np.exp(2 * s * t) - 1 + s / a0)
    b = s * (1 - np.exp(-t)) + a0 * np.exp(-t)
    res = (u[None, :, :s.shape[1]] * a[:, None, :]) @ vh
    res_shallow = (u[None, :, :s.shape[1]] * b[:, None, :]) @ vh

    if use_perturbed:
        res = np.insert(res, remove_row, 0, axis=1)
        res_shallow = np.insert(res_shallow, remove_row, 0, axis=1)
    w.append(res)

fig, axes = plt.subplots(1, nclasses, figsize=(nclasses * 4, 4))
for task, ax in enumerate(axes):
    for i in range(len(w)):
        for j in range(nfeatures):
            ax.plot(w[i][:, j, task], linestyle='-' if i == 0 else '--', color=cm.tab10(j), label=str(j))
plt.legend()
plt.show()