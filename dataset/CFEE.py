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

import FaceUtils
import dataset.DataUtils as DataUtils

i_w = 256
i_h = 256


# create_dir(output_dir)


def clear_dir(output_dir):
    for c in ["a_n", "b_e"]:
        class_dir = output_dir + "/" + c
        DataUtils.recreate_dir(class_dir)


def parse_file_name(filename):
    filename = filename.replace(".jpg", "")
    # print(filename)
    ss = filename.split("_")
    id = ss[1]
    c = ss[0]
    return id, c


def process_img(img):
    img = FaceUtils.extract_face(img)
    img = cv2.resize(img, (i_w, i_h))
    return img


def process_and_save_img(file, path, output_dir):
    img_file = path + "/" + file
    img = cv2.imread(img_file)
    img = process_img(img)
    id, class_name = parse_file_name(file)
    clas = int(class_name)
    # cv2.imwrite(output_dir+file, img)
    # break
    class_dirs = ['b_e']  # all images will put in b_e. We think a neutral face is also an expression
    if clas == 1:
        class_dirs.append('a_n')

    for class_dir in class_dirs:
        dir = output_dir + "/" + class_dir
        DataUtils.create_dir(dir, overwrite=False)
        file_full = dir + "/" + id + "_" + class_name + ".png"
        # print(f"Save image [{file_full}]")
        if os.path.exists(file_full):
            # raise Exception(f"File [{file_full}] exists!")
            print(f"File [{file_full}] exists!")
        cv2.imwrite(file_full, img)
    return id, class_name


def extract_images(source_dir, output_dir):
    # input 256*256
    # clear_dir()
    dir_0_s = DataUtils.get_subdirs(source_dir)
    identities = set()
    img_count = 0
    for dir_0 in dir_0_s:
        dir_0 = source_dir + "/" + dir_0
        dir_1_s = DataUtils.get_subdirs(dir_0)
        for dir_1 in dir_1_s:
            if not dir_1.startswith("Images"):
                continue
            dir_2 = dir_0 + "/" + dir_1
            image_fs = DataUtils.get_subdirs(dir_2)
            for image_f in image_fs:
                if not image_f.endswith(".jpg"):
                    continue
                print(f'Process image [{dir_2 + "/" + image_f}]')
                id, class_name = process_and_save_img(image_f, dir_2, output_dir)
                img_count += 1
                identities.add(id)

    return len(identities), img_count


def compare():
    img_file = root_dir + "/CFEE_Database_230/01-04/Images1/02_339_3470.jpg"
    img_file = root_dir + "/CFEE_Database_230/01-04/Images1/01_126_3963.jpg"

    img = cv2.imread(img_file)
    print(img.shape)
    img1 = cv2.imread(root_dir + "/expression_2.0/a_n/CFD-AM-242_n.png")

    plt.rcParams['figure.figsize'] = [10, 6]
    f, axarr = plt.subplots(nrows=1, ncols=2)

    # axarr[0].set_title("I")
    # axarr[1].set_title("O")
    img = process_img(img)
    img1 = process_img(img1)
    # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axarr[1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()


def to_neutral_expression_file(file_e):
    prefix = file_e.split(".")[0]
    return prefix + "_01.png"


def check_pairs(source_dir):
    group_dirs = DataUtils.get_subdirs(source_dir)
    for group_dir in group_dirs:  # train, test
        source_b_e_dir = source_dir + "/" + group_dir + "/b_e"
        source_a_n_dir = source_dir + "/" + group_dir + "/a_n"
        source_image_files = DataUtils.get_image_files(source_b_e_dir)
        for source_image_file in source_image_files:
            n_file = to_neutral_expression_file(source_image_file)
            n_file_full = source_a_n_dir + "/" + n_file
            if not os.path.exists(n_file_full):
                print(n_file_full)
                os.remove(source_b_e_dir + "/" + source_image_file)

    print("Done!")


root_dir = "/Users/$USER/Documents/Repositories/datasets"
source_dir = root_dir + "/CFEE_Database_230"
output_dir = root_dir + "/expression_CFEE"

# compare()
# clear_dir()
# identity_count, img_count = extract_images()
# print(f"Done! people:{identity_count}, img_count:{img_count}")
# people:230, img_count:5954

# DataUtils.split_data(output_dir, root_dir + "/expression_CFEE_1.0")

# DataUtils.rszie_images(root_dir + "/expression_CFEE_p_1.0", root_dir + "/expression_CFEE_p_128", (128, 128))
# DataUtils.rszie_images(root_dir + "/expression_CFEE_1.0", root_dir + "/expression_CFEE_128", (128, 128))


# check_pairs(root_dir + "/expression_CFEE_p_128")

###################################### by each class
ids_test_V2 = ["122", "174", "185", "224", "272", "188", "187", "265", "378", "365", "369", "268", "235", "284", "304",
               "118"]
# DataUtils.split_by_class(root_dir + "/expression_CFEE", root_dir + "/expression_CFEE_cls")
# DataUtils.split_data(root_dir + "/expression_CFEE", root_dir + "/expression_CFEE_id_256", ids_test=["122", "174", "185", "224", "272", "188", "187", "265"])
DataUtils.split_data(root_dir + "/expression_CFEE", root_dir + "/expression_CFEE_id_256_V2", ids_test=ids_test_V2)
# DataUtils.split_data(root_dir + "/expression_CFEE_cls", root_dir + "/expression_CFEE_id_cls_256", ids_test=["122", "174", "185", "224", "272", "188", "187", "265"])
# DataUtils.split_data(root_dir + "/expression_CFEE_cls", root_dir + "/expression_CFEE_cls_256")
DataUtils.rszie_images(root_dir + "/expression_CFEE_id_256_V2", root_dir + "/expression_CFEE_id_128_V2", (128, 128))
# DataUtils.rszie_images(root_dir + "/expression_CFEE_cls_1.0", root_dir + "/expression_CFEE_cls_128", (128, 128))
# DataUtils.rszie_images(root_dir + "/expression_CFEE_id_cls_256", root_dir + "/expression_CFEE_id_cls_128", (128, 128))
# identity_count, img_count = extract_images(source_dir, output_dir, by_each_class=True)
# print(f"Done! people:{identity_count}, img_count:{img_count}")
# people:230, img_count:5954

# DataUtils.delete_by_identity(root_dir + "/expression_CFEE_id_diff_128/train")
