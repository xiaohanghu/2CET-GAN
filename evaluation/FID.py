"""
2CET-GAN
Copyright (c) 2022-present, [author].
This work is licensed under the MIT License.
"""

import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import argparse
from munch import Munch

from DataLoader import SingleFolderDataset
from DataLoader import create_data_loader_eval


# frechet inception distance (FID)
class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


@torch.no_grad()
def calculate_act(path, img_paths, inception, batch_size, config):
    loader = create_data_loader_eval(path, img_paths, config.img_size, batch_size)
    actvs = []
    for (x, id, cls, _) in tqdm(loader, total=len(loader)):
        actv = inception(x.to(config.device))
        actvs.append(actv)
    actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
    return actvs


def calculate_fid(mu1, sigma1, mu2, sigma2):
    # calculate mean and covariance statistics

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid_by_data(dir1, dir2, img_paths1, img_paths2, config, batch_size=50):
    inception = InceptionV3().eval().to(config.device)
    act1 = calculate_act(dir1, img_paths1, inception, batch_size, config)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    del act1
    act2 = calculate_act(dir2, img_paths2, inception, batch_size, config)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    del act2
    return calculate_fid(mu1, sigma1, mu2, sigma2)


def calculate_fid_by_dir(dir1, dir2, config, batch_size=50):
    print(f"calculate_fid_by_dir. dir1:{dir1}, dir2:{dir2}")
    return calculate_fid_by_data(dir1, dir2, None, None, config, batch_size=batch_size)


def calculate_fid_by_cls(dir1, dir2, config, batch_size=50):
    print(f"calculate_fid_by_cls. dir1:{dir1}, dir2:{dir2}")
    e_dataset_1 = SingleFolderDataset(dir1, None, None)
    e_dataset_2 = SingleFolderDataset(dir2, None, None)
    cls_index_map_1 = e_dataset_1.get_cls_index_map()
    cls_index_map_2 = e_dataset_2.get_cls_index_map()

    result = Munch()
    clses = cls_index_map_1.keys() | cls_index_map_2.keys()
    clses = sorted(clses)
    fids = []
    for cls in clses:
        print(f"Calculate class [{cls}].")
        if cls not in cls_index_map_1:
            print(f"Class [{cls}] no in [{dir1}]")
            continue
        if cls not in cls_index_map_2:
            print(f"Class [{cls}] no in [{dir2}]")
            continue
        img_paths_1 = [e_dataset_1.get_filepath(i) for i in cls_index_map_1[cls]]
        img_paths_2 = [e_dataset_2.get_filepath(i) for i in cls_index_map_2[cls]]
        fid = calculate_fid_by_data(None, None, img_paths_1, img_paths_2, config, batch_size=batch_size)
        fids.append(fid)
        result[cls] = fid

    result["avg"] = np.array(fids).mean()

    return result


def calculate_all_fid(data_dir, eval_dir, config, batch_size=50):
    data_n_dir = data_dir + "/a_n"
    data_e_dir = data_dir + "/b_e"
    eval_dir_n = eval_dir + "/a_n"
    eval_dir_e_z = eval_dir + "/b_e_z"
    eval_dir_e_r = eval_dir + "/b_e_r"

    fid_n = calculate_fid_by_dir(data_n_dir, eval_dir_n, config, batch_size)
    fid_e_z = calculate_fid_by_dir(data_e_dir, eval_dir_e_z, config, batch_size)
    fid_e_r = calculate_fid_by_dir(data_e_dir, eval_dir_e_r, config, batch_size)
    return fid_n, fid_e_z, fid_e_r


def test():
    data_dir = "/Users/$USER/Documents/Repositories/datasets/expression_CFEE_128/test"
    eval_dir = "/Users/$USER/Documents/Repositories/expression-GAN/V2.6.6/test/eval/test"
    config = Munch()
    config.img_size = 128
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calculate_all_fid(data_dir, eval_dir, config)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir1', type=str, nargs=2, help='')
    # parser.add_argument('--dir2', type=str, nargs=2, help='')
    # parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    # config = parser.parse_args()
    # fid_value = calculate_fid_by_dir(config.dir1, config.dir2, config, config.batch_size)
    # print('FID: ', fid_value)

    # test()
    None
