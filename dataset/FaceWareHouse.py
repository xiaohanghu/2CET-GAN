"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil

from torchvision.datasets import ImageFolder

from dataset.DataUtils import recreate_dir

root_dir = "/Users/xiaohanghu/Documents/Repositories/dataset"
source_dir = root_dir + "/FaceWarehouse/FaceWarehouse_Data"
output_dir = root_dir + "/expression_1.0"


# create_dir(output_dir)

dirs = os.listdir(source_dir)
# input 256*256
i_w = 256
i_h = 256


def clear_dir():
    for c in ["a_n", "b_e"]:
        class_dir = output_dir + "/" + c
        recreate_dir(class_dir)


# clear_dir()

for dir in dirs:
    if not dir.startswith("Tester_"):
        continue
    id = dir.replace("Tester_", "")
    # print(id)
    img_dir = dir + "/TrainingPose"
    # print(img_dir)
    for pose_i in range(20):
        file = "pose_" + str(pose_i) + ".png"
        # file = "pose_5.png"
        file_full = source_dir + "/" + img_dir + "/" + file
        # print(file_full)
        img = cv2.imread(file_full)
        h = img.shape[0]
        w = img.shape[1]
        s_h = int(h / 2 - i_h / 2) + 20
        s_w = int(w / 2 - i_w / 2)
        img = img[s_h:s_h + i_h, s_w:s_w + i_w]
        class_name = str(pose_i)
        if class_name == "0":
            class_name = "n"
        class_dir = "a_n"
        if class_name != "n":
            class_dir = "b_e"
        file_name = id + "_" + class_name + ".png"
        file_full = output_dir + "/" + class_dir + "/" + file_name
        print(file_full)
        if os.path.exists(file_full):
            raise Exception(f'File [{file_full}] exists!')
        cv2.imwrite(file_full, img)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        # break

print("Done!")
