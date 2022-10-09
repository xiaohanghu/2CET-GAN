"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import os
import random
import shutil
from munch import Munch
import cv2


def create_dir(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
        return

    if overwrite and os.path.isdir(path):
        shutil.rmtree(path)
        os.makedirs(path)


def recreate_dir(path):
    create_dir(path, True)


def get_subdirs(dir):
    dirs = os.listdir(dir)
    result = [d for d in dirs if not (d.startswith(".") or d.startswith("Icon") or d.endswith(".xls"))]
    return sorted(result)


def is_image_file(file):
    return file.endswith(('.png', '.jpg', '.jpeg', '.JPG'))


def get_image_files(dir):
    files = os.listdir(dir)
    files = [f for f in files if is_image_file(f)]
    return files


def parse_file_name(filename):
    filename = filename.rsplit('/', 1)[-1]
    filename = filename.replace(".png", "")
    filename = filename.replace(".jpg", "")
    # print(filename)
    ss = filename.split("_")
    id = ss[0]
    cls = ss[-1]
    return id, cls


def split_data(source_dir, target_dir, ids_test=None, test_proportion=0.05):
    """
    Split dataset into train and test folder
    """
    class_dirs = get_subdirs(source_dir)
    target_train_dir = target_dir + "/train"
    target_test_dir = target_dir + "/test"

    recreate_dir(target_train_dir)
    recreate_dir(target_test_dir)

    id_map = Munch()
    for cls_dir in class_dirs:
        source_cls_dir = source_dir + "/" + cls_dir

        target_train_cls_dir = target_train_dir + "/" + cls_dir
        target_test_cls_dir = target_test_dir + "/" + cls_dir
        create_dir(target_train_cls_dir)
        create_dir(target_test_cls_dir)

        source_image_files = get_image_files(source_cls_dir)
        for image_file_name in source_image_files:
            is_train_set = True
            if ids_test is not None:
                image_file_name.split()
                id, cls = parse_file_name(image_file_name)
                if (id not in id_map) or (id_map[id] is None):
                    id_map[id] = 1
                else:
                    id_map[id] += 1
                if id in ids_test:
                    is_train_set = False
            else:
                is_train_set = (random.random() >= test_proportion)
            if is_train_set:
                target_cls_dir = target_train_cls_dir
            else:
                target_cls_dir = target_test_cls_dir

            target_image_file_full = target_cls_dir + "/" + image_file_name
            source_image_file_full = source_cls_dir + "/" + image_file_name
            print(f"Copy [{source_image_file_full}] to [{target_image_file_full}]")
            shutil.copy2(source_image_file_full, target_image_file_full)

    for id, count in id_map.items():
        if count != 26:
            print(f"{id}:{count}")
    print(f"Split done!")


def parse_file_name(filename):
    filename = filename.rsplit('/', 1)[-1]
    filename = filename.replace(".png", "")
    filename = filename.replace(".jpg", "")
    # print(filename)
    ss = filename.split("_")
    id = ss[0]
    c = ss[-1]
    return id, c


def split_by_class(source_dir, target_dir):
    """
    Split dataset into folder by class
    """
    class_dirs = get_subdirs(source_dir)

    for cls_dir in class_dirs:
        source_cls_dir = source_dir + "/" + cls_dir

        source_image_files = get_image_files(source_cls_dir)
        for image_file_name in source_image_files:
            id, cls = parse_file_name(image_file_name)

            target_cls_dir = target_dir + "/" + cls
            create_dir(target_cls_dir, overwrite=False)

            target_image_file_full = target_cls_dir + "/" + image_file_name
            source_image_file_full = source_cls_dir + "/" + image_file_name
            print(f"Copy [{source_image_file_full}] to [{target_image_file_full}]")
            shutil.copy2(source_image_file_full, target_image_file_full)

    print(f"Split done!")


def delete_by_identity(dir):
    """
    Let a_n and b_e contain different identities.
    """

    dir_a_n = dir + "/a_n"
    a_n_files = get_image_files(dir_a_n)
    n_ids = []
    random.seed(10)
    for a_n_file in a_n_files:
        id, cls = parse_file_name(a_n_file)
        # print(f"{id},{cls}")
        keep_in_n = random.randint(0, 1)
        if keep_in_n:
            n_ids.append(id)
        else:
            os.remove(dir_a_n + "/" + a_n_file)
    print(n_ids)

    dir_b_e = dir + "/b_e"
    for b_e_file in get_image_files(dir_b_e):
        id, cls = parse_file_name(b_e_file)
        if id in n_ids:
            os.remove(dir_b_e + "/" + b_e_file)
    print(n_ids)


def rszie_images(source_dir, target_dir, size):
    source_subdirs = get_subdirs(source_dir)
    for source_subdir in source_subdirs:
        if is_image_file(source_subdir):
            source_img_file = source_subdir
            source_img_file_full = source_dir + "/" + source_img_file
            target_img_file_full = target_dir + "/" + source_img_file
            img = cv2.imread(source_img_file_full)
            img = cv2.resize(img, size)
            print(f"Resize [{source_img_file_full}] to [{target_img_file_full}]")
            cv2.imwrite(target_img_file_full, img)
        else:
            source_subdir_full = source_dir + "/" + source_subdir
            if os.path.isdir(source_subdir_full):
                target_subdir_full = target_dir + "/" + source_subdir
                print(f"Recreate dir [{target_subdir_full}]")
                recreate_dir(target_subdir_full)
                rszie_images(source_subdir_full, target_subdir_full, size)

    print(f"Reszie done!")
