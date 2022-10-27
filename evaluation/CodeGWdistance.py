"""
2CET-GAN
Copyright (c) 2022-present, [author].
This work is licensed under the MIT License.
"""
import numpy as np
import ot
import scipy as sp
# plt.show()
n_clusters = 25


def get_codes():
    codes = []
    labels = []
    for c in range(2, 2 + n_clusters):
        c_codes = np.load(f"code_data/code_{c:02}.data.npy")
        labels.extend([c - 2] * len(c_codes))
        codes.extend(c_codes)
    return codes, labels

def calculate_GWdistance(x1, x2):
    C1 = sp.spatial.distance.cdist(x1, x1)
    C2 = sp.spatial.distance.cdist(x2, x2)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(len(C1))  # Return a uniform histogram of length `n` (simplex).
    q = ot.unif(len(C2))
    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=True, log=True)

    return log0['gw_dist']

