"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import numpy as np
import torch
from munch import Munch

# plt.show()
n_clusters = 25


def get_codes():
    codes = []
    labels = []
    for c in range(2, 2 + n_clusters):
        c_codes = np.load(f"code_data/code_{c:02}.data.npy")
        # codes.append(c_codes)
        c_mean = c_codes.mean(axis=0, keepdims=False)
        codes.append(c_mean)
        labels.append(c)
    return codes, labels


codes, labels = get_codes()

# M = sp.spatial.distance.cdist(codes, codes)
# M /= M.max()

map = Munch()
distances = []
n = len(codes)
for i in range(n):
    for j in range(i + 1, n):
        dis = dis2 = torch.dist(torch.from_numpy(codes[i]), torch.from_numpy(codes[j]), p=2)
        distances.append(dis)
        map[dis] = (labels[i], labels[j])

# print(f"M.shape:{M.shape}")
distances = sorted(distances)
for i in range(5):
    print(f"most close: {map[distances[i]]}, distance:{distances[i]:.3f}")

distances.reverse()
for i in range(5):
    print(f"most far away: {map[distances[i]]}, distance:{distances[i]:.3f}")


# print(M)
# plt.imshow(M)
# plt.show()

# most close: (12, 13), distance:0.071 Angrily disgusted, Appalled
# most close: (5, 11), distance:0.096 Angry, Sadly angry
# most close: (8, 21), distance:0.148 Disgustedly surprised, Fearfully surprised
# most close: (11, 14), distance:0.155 Sadly angry, Hatred
# most close: (7, 13), distance:0.158 Disgusted, Appalled

# most far away: (5, 9), distance:1.020 Angry, Happily surprised
# most far away: (6, 12), distance:0.999
# most far away: (6, 13), distance:0.982
# most far away: (6, 10), distance:0.978
# most far away: (9, 11), distance:0.976
