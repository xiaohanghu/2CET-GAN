import torch
import Demo
from DataLoader import create_transform_test
import numpy as np

# from torchmetrics.functional import pairwise_euclidean_distance

config = Demo.get_config_(eval_model_step=65000)
transform = create_transform_test(config)
ids = ["122", "174", "185", "187", "188", "224", "265", "272"]
rs = ["227_02", "176_02", "133_05", "269_05", "113_16", "207_16", "200_26", "303_26"]

ns = []
for id in ids:
    ns.append(id + "_01")
x_n = Demo.get_images(config.test_dir, ns, transform, True)


def pairwise_euclidean_distance(x):
    ds = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = torch.dist(x[i], x[j], p=2)
            # print(f"{i}-{j}:", d)
            ds.append(d)
    return np.mean(ds)


def avg_euclidean_distance(x1, x2):
    ds = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            d = torch.dist(x1[i], x2[j], p=2)
            d=d.item()
            print(f"{i}-{j}:", d)
            ds.append(d)

    return np.mean(ds)


e_ds = []
for id in ids:
    es = []
    for c in range(2, 27):
        es.append(f"{id}_{c:02}")
    x_n_0 = Demo.get_images(config.test_dir, [id + "_01"], transform, True)
    x_e = Demo.get_images(config.test_dir, es, transform, False)
    e_ds.append(avg_euclidean_distance(x_n_0, x_e))

ed = np.mean(e_ds)

nd = pairwise_euclidean_distance(x_n)
print(ed / nd)
