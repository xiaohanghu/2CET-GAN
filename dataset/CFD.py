"""
2CET-GAN
Copyright (c) 2022-present, [author].
This work is licensed under the MIT License.
"""

import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil

from torchvision.datasets import ImageFolder

from FaceUtils import extract_face, extract_face_fixed
from dataset.DataUtils import recreate_dir

root_dir = "/Users/$USER/Documents/Repositories/datasets"

i_w = 256
i_h = 256

# output_dir = "."
output_dir = root_dir + f"/expression_CFD_{i_w}"


# create_dir(output_dir)


def clear_dir():
    for c in ["a_n", "b_e"]:
        class_dir = output_dir + "/" + c
        recreate_dir(class_dir)


def parse_file_name(filename):
    filename = filename.replace(".jpg", "")
    # print(filename)
    ss = filename.split("-")
    c = ss[-1]
    id = ""
    for t in ss[0:-2]:
        id = id + "-" + t
    id = id[1:]

    c = c.lower()
    # print(id, c)
    return id, c


def process_img(img):
    img = extract_face_fixed(img)
    img = cv2.resize(img, (i_w, i_h))
    return img


def process_and_save_img(file, path, output_dir):
    img_file = path + "/" + file
    img = cv2.imread(img_file)
    # print(f"Read image [{img_file}]")
    img = process_img(img)
    id, class_name = parse_file_name(file)
    # cv2.imwrite(output_dir+file, img)
    # break
    class_dir = 'a_n'
    if class_name != 'n':
        class_dir = 'b_e'
    file_full = output_dir + "/" + class_dir + "/" + id + "_" + class_name + ".png"
    print(file_full)
    # if os.path.exists(file_full):
    #     raise Exception(f'File [{file_full}] exists!')
    cv2.imwrite(file_full, img)


def extract_images(source_dir):
    # input 256*256
    # clear_dir()
    dirs = os.listdir(source_dir)
    identity_count = 0
    img_count = 0
    for dir in dirs:
        if dir.startswith(".") or dir.startswith("Icon"):
            continue
        if dir.endswith(".jpg"):
            process_and_save_img(dir, source_dir, output_dir)
            img_count += 1
            identity_count += 1
            continue
        path = source_dir + "/" + dir
        files = os.listdir(path)
        if len(files) == 0:
            raise Exception(f"Empty dir [{path}]!")
        files = [f for f in files if not (f.startswith(".") or f.startswith("Icon"))]
        # if len(files) == 1:
        #     continue

        for file in files:
            process_and_save_img(file, path, output_dir)
            img_count += 1
        identity_count += 1
    return identity_count, img_count


clear_dir()
identity_count_total = 0
img_count_total = 0

identity_count, img_count = extract_images(root_dir + "/CFD Version 3.0/Images/CFD")
identity_count_total += identity_count
img_count_total += img_count
identity_count, img_count = extract_images(root_dir + "/CFD Version 3.0/Images/CFD-INDIA")
identity_count_total += identity_count
img_count_total += img_count
identity_count, img_count = extract_images(root_dir + "/CFD Version 3.0/Images/CFD-MR")
identity_count_total += identity_count
img_count_total += img_count
print(f"Done! people:{identity_count_total}, img_count:{img_count_total}")
# CFD: people:158, img_count:768
# CFD full: people:597, img_count:1207
# CFD-INDIA: people:142, img_count:146
# CFD-MR: people:88, img_count:88
