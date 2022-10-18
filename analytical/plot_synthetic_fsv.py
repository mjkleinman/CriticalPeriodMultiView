import numpy as np
import matplotlib.pyplot as plt

"""
x_a = x_0 + n_a
x_b = x_0 + n_b
z_i = alpha_i x_a + (1-alpha_i) x_b
alpha_i ~ Beta()
"""

num_samples = 20000
beta = 20
vx0 = 1
vna = vnb = 1
vxa = vx0 + vna
vxb = vx0 + vnb
for alpha in [1, 20, 30]:  # 10 if just varying with one Modality
    # Want half the units to be "dominant" for x_a, other half to be dominant for x_b, hence the (1 - w) term
    wa = np.concatenate((np.random.beta(a=alpha, b=beta, size=num_samples // 2),
                        1 - np.random.beta(a=alpha, b=beta, size=num_samples // 2)))

    # Computing variance of Z given X_A
    vz = vx0 + wa ** 2 * vna + (1 - wa) ** 2 * vnb
    var_zxa = vz * (1 - ((vx0 + wa * vna) ** 2) / (vxa) * vz)

    # Computing variance of Z given X_B
    wb = 1 - wa  # Want the weightings reversed for other half of units so that the sampled weight corresponds to x_b
    vz = vx0 + (1 - wb) ** 2 * vna + wb ** 2 * vnb # note the w and 1 - w are swapped
    var_zxb = vz * (1 - ((vx0 + wb * vnb) ** 2) / (vxb) * vz)

    fsv = (var_zxa - var_zxb) / (var_zxa + var_zxb)
    bins = np.linspace(-1.1, 1.1, 40)
    # plt.hist(fsv[:num_samples // 2], bins, color='r')
    fig = plt.figure(figsize=(3.2, 2.7))
    plt.hist(fsv, bins, color='r')
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.yticks([])
    plt.xlabel('Relative Source Variance')
    plt.ylabel('Number of units')
    plt.savefig(f'plots/distbeta_alpha{alpha}_beta{beta}_updated.pdf', format='pdf', dpi=None, bbox_inches='tight')
    plt.close()
