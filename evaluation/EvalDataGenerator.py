"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import math

import cv2
from munch import Munch
import shutil
from tqdm import tqdm
import random

import Utils
from DataLoader import create_transform_test, SingleFolderDataset, img_files
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from Utils import load_model, denormalize, show_imgs, get_neutral_code


@torch.no_grad()
def generate_fake(img, c, models_s, transform):
    # img = transform(img)
    imgs = torch.unsqueeze(img, dim=0)
    imgs_new = models_s.generator(imgs, c)
    imgs_new = denormalize(imgs_new)
    img_new = imgs_new[0]
    # img_new = np.transpose(img_new, (1, 2, 0))
    # print(img_new.shape)
    img_new = img_new.mul(255).add_(0.5).clamp_(0, 255) \
        .permute(1, 2, 0) \
        .to('cpu', torch.uint8).numpy()
    return img_new


def save_image(file_name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, img)


def generate_eval_n(root_dir, eval_dir, models_s, transform, config):
    e_dir = root_dir + "/b_e"
    eval_dir_n = eval_dir + "/a_n"
    Utils.recreate_dir(eval_dir_n)
    e_dataset = SingleFolderDataset(e_dir, None, transform)

    n_e = len(e_dataset)
    c_n_real = torch.FloatTensor(torch.zeros((1, config.code_dim))).to(config.device)

    # E to N
    for index, (img_e, id, cls) in enumerate(e_dataset):
        img_n_fake = generate_fake(img_e, c_n_real, models_s, transform)
        file_name = f"{index}_n.png"
        file_name = eval_dir_n + "/" + file_name
        print(f"E to N [{index + 1}/{n_e}], save file {file_name}.")
        save_image(file_name, img_n_fake)


def generate_eval_e_z(source_data_dir, eval_dir, models_s, transform, config):
    n_dir = source_data_dir + "/a_n"
    e_dir = source_data_dir + "/b_e"
    n_dataset = SingleFolderDataset(n_dir, None, transform)
    e_dataset = SingleFolderDataset(e_dir, None, transform)

    n_n = len(n_dataset)
    n_e = len(e_dataset)

    # N to E by z
    eval_dir_e_z = eval_dir + "/b_e_z"
    Utils.recreate_dir(eval_dir_e_z)
    mul = math.ceil(n_e / n_n)
    for m in range(mul):
        for index, (img_n, id, c) in enumerate(n_dataset):
            c = Utils.generate_rand_code(1, config)
            img_e_fake = generate_fake(img_n, c, models_s, transform)
            index_e = m * n_n + index
            file_name = f"{index_e}_e.png"
            file_name = eval_dir_e_z + "/" + file_name
            print(f"N to E by z [{m * n_n + index}/{mul * n_n}], save file {file_name}.")
            save_image(file_name, img_e_fake)

    print(f"generate_eval() all done!")


def generate_eval_e_r(source_data_dir, eval_dir, num_each_cls, models_s, transform, config):
    n_dir = source_data_dir + "/a_n"
    e_dir = source_data_dir + "/b_e"
    n_dataset = SingleFolderDataset(n_dir, None, transform)
    e_dataset = SingleFolderDataset(e_dir, None, transform)

    n_n = len(n_dataset)
    # n_e = len(e_dataset)

    # N to E by z
    cls_index_map = e_dataset.get_cls_index_map()

    num_cls = len(cls_index_map)
    print(f"Number of emotional classes: {num_cls}")

    eval_dir_e_r = eval_dir + "/b_e_r"
    Utils.recreate_dir(eval_dir_e_r)
    num_each_cls = num_each_cls
    count = 0
    for index, (img_n, id, cls_n) in enumerate(n_dataset):
        for cls, img_indexes in cls_index_map.items():
            num_each = min(num_each_cls, len(img_indexes))
            total = n_n * num_cls * num_each
            img_indexes = random.sample(img_indexes, num_each)
            for img_index in img_indexes:
                # print(f"imgs_index:{imgs_index}")
                img_e, id_r, cls_r = e_dataset[img_index]
                c_e = models_s.encoder(torch.unsqueeze(img_e, dim=0))
                img_e_fake = generate_fake(img_n, c_e, models_s, transform)
                file_name = f"{id}_{id_r}_{cls_r}.png"
                file_name = eval_dir_e_r + "/" + file_name
                count += 1
                print(f"N to E by reference [{count}/{total}], save file {file_name}.")
                save_image(file_name, img_e_fake)

    assert count == len(img_files(eval_dir_e_r))


def generate_eval(root_dir, eval_dir,num_each_cls, models_s, transform, config):
    generate_eval_n(root_dir, eval_dir, models_s, transform, config)
    generate_eval_e_z(root_dir, eval_dir, models_s, transform, config)
    generate_eval_e_r(root_dir, eval_dir,num_each_cls, models_s, transform, config)
    print(f"generate_eval() all done!")


def generate_eval_data(num_each_cls, models_s, transform, config):
    Utils.recreate_dir(config.eval_dir)
    generate_eval(config.test_dir, config.eval_dir + "/test",num_each_cls, models_s, transform, config)
    generate_eval(config.train_dir, config.eval_dir + "/train",num_each_cls, models_s, transform, config)
