"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import matplotlib.pyplot as plt
import numpy as np
import DataLoader
import torch
import Utils
from PIL import Image

from Model import create_model
from Utils import load_model
from main import get_config
from torch import FloatTensor


def get_models(config, step):
    models, models_s = create_model(config)
    # load_models(config, models, "models", True, step)
    load_model(config, models_s, "model_s", True, step)
    del models
    return models_s


def show(happy, sad, angry):
    x = np.arange(0, 32, 1)
    CODE_MIN = -0.5
    CODE_MAX = 0.5
    random = np.random.uniform(CODE_MIN, CODE_MAX, (len(angry), 32))

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 4))

    # alpha = 0.03
    alpha = 0.1

    def show_bar(ax, ys, name):
        xmin = np.arange(-0.5, -0.5 + 32, 1)
        xmax = np.arange(0.5, 0.5 + 32, 1)
        for y1 in ys:
            # ax1.fill_between(x, y1, color='b', alpha=alpha)
            # ax.bar(x, y1, color='#42946B', width=1.0, alpha=alpha)
            ax.hlines(y1, xmin, xmax, color='#42946B', alpha=alpha)
        # ax.bar(x, ys_mean, color='r', width=1.0, alpha=1)

        ys_mean = ys.mean(axis=0, keepdims=False)
        ax.hlines(ys_mean, xmin, xmax, color='black')

        # ax.step(x, ys_mean, 'k', where="mid", linewidth=1)
        ax.set_title(name)
        ax.set_ylim([-0.51, 0.51])

    def show_line(ax, ys, name):
        x = np.arange(0, 32, 1)
        for y in ys:
            ax.plot(x, y, color='#42946B', alpha=alpha)
        ys_mean = ys.mean(axis=0, keepdims=False)
        ax.plot(x, ys_mean, color='black', alpha=0.5)
        ax.set_title(name)
        ax.set_ylim([-0.51, 0.51])

    show_fun = show_line
    show_fun = show_bar
    show_fun(axs[0][0], random, '(Random)')
    show_fun(axs[0][1], happy, 'Happy')
    show_fun(axs[1][0], sad, 'Sad')
    show_fun(axs[1][1], angry, 'Angry')

    # ax2.fill_between(x, y1, 1)
    # ax2.set_title('fill between y1 and 1')
    #
    # ax3.fill_between(x, y1, y2)
    # ax3.set_title('fill between y1 and y2')
    # ax3.set_xlabel('x')
    fig.tight_layout()
    plt.savefig('output/code_chart.png')
    plt.show()


DATASETS_ROOT = "/Users/xiaohanghu/Documents/Repositories/datasets"


def get_codes(image_files, transform, models_s):
    result = []
    for file in image_files:
        img = Image.open(file).convert('RGB')
        img = transform(img)
        print(img.shape)
        c = models_s.encoder(img)
        result.append(c)
    return c


def generate_data(config, models_s, transform):
    dir = config.train_dir + "/b_e"

    e_dataset = DataLoader.SingleFolderDataset(dir, None, transform)
    cls_index_map = e_dataset.get_cls_index_map()

    def get_codes(img_indexes):
        c_e_s = []
        i = 0
        for img_index in img_indexes:
            # print(f"{i}")
            i += 1
            img_e, _, _, _ = e_dataset[img_index]
            c_e = models_s.encoder(torch.unsqueeze(img_e, dim=0))
            # print(c_e)
            c_e_s.append(c_e.cpu().detach().numpy()[0])
        return c_e_s

    for c in range(1, 27):
        print(f"Generate {c:02}...")
        codes = get_codes(cls_index_map[f"{c:02}"])
        np.save(f"code_data/code_{c:02}.data", codes)


def generate_data_():
    config = get_config_()

    transform = DataLoader.create_transform_test(config)
    models_s = get_models(config, config.eval_model_step)
    generate_data(config, models_s, transform)


def code_face(transform, config, models_s):
    # n_dir = config.test_dir + "/a_n"
    n_dir = config.train_dir + "/a_n"
    # n_dataset = DataLoader.SingleFolderDataset(n_dir, None, transform)
    # img_n, _, _ = n_dataset[2]

    img_f_0 = n_dir + "/115_01.png"
    # img_f_0 = config.test_dir + "/a_n/174_01.png"
    img_0 = transform(Image.open(img_f_0).convert('RGB'))
    x_0 = torch.unsqueeze(img_0, dim=0)
    xs = [x_0, x_0, x_0, x_0]
    add_org = False
    output_f = "ouput/code_face_CFEE.png"

    # img_f_0 = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFD_256/a_n/CFD-MF-346_n.png"
    # img_f_1 = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFD_256/a_n/CFD-AF-218_n.png"
    # img_f_2 = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFD_256/a_n/CFD-LM-212_n.png"
    # img_f_3 = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFD_256/a_n/CFD-MM-306_n.png"
    # img_0 = transform(Image.open(img_f_0).convert('RGB'))
    # img_1 = transform(Image.open(img_f_1).convert('RGB'))
    # img_2 = transform(Image.open(img_f_2).convert('RGB'))
    # img_3 = transform(Image.open(img_f_3).convert('RGB'))
    # x_0= torch.unsqueeze(img_0, dim=0)
    # x_1 = torch.unsqueeze(img_1, dim=0)
    # x_2 = torch.unsqueeze(img_2, dim=0)
    # x_3 = torch.unsqueeze(img_3, dim=0)
    # xs = [x_0,x_1,x_2,x_3]
    # add_org = True
    # output_f = "code_face_CFD.png"

    def propotion_code(code, ratio):
        # if ratio == 0:
        #     return Utils.get_neutral_code(1, x_n.dtype, config)
        c = code * ratio
        c = FloatTensor(c).to(config.device)
        c = torch.unsqueeze(c, dim=0)
        return c

    happy = np.load("code_data/code_02.data.npy")
    sad = np.load("code_data/code_03.data.npy")
    # angry = np.load("code_data/code_05.data.npy")
    surprised = np.load("code_data/code_06.data.npy")

    happy_mean = happy.mean(axis=0, keepdims=False)
    sad_mean = sad.mean(axis=0, keepdims=False)
    # angry_mean = angry.mean(axis=0, keepdims=False)
    surprised_mean = surprised.mean(axis=0, keepdims=False)

    cols = []
    col0 = []
    n = 5
    gap = 2.5

    if add_org:
        for x in xs:
            col0.append(Utils.denormalize_RGB(x)[0])
        col0 = torch.stack(col0, dim=0)
        cols.append(col0)

    for i in np.arange(gap, 1 + gap * n, gap):
        ratio = (i / 10)
        print(f"ratio:{ratio}")

        col = []
        c_happy = propotion_code(happy_mean, ratio)
        print(c_happy)
        col.append(Utils.denormalize_RGB(models_s.generator(xs[0], c_happy))[0])

        c_sad = propotion_code(sad_mean, ratio)
        col.append(Utils.denormalize_RGB(models_s.generator(xs[1], c_sad))[0])

        # c_angry = propotion_code(angry_mean, ratio)
        # col.append(Utils.denormalize_RGB(models_s.generator(xs[2], c_angry))[0])

        c_surprised = propotion_code(surprised_mean, ratio)
        col.append(Utils.denormalize_RGB(models_s.generator(xs[3], c_surprised))[0])

        col = torch.stack(col, dim=0)
        print(col.shape)
        cols.append(col)

    x_concat = torch.stack(cols, dim=0)
    ncol = x_concat.size(0)
    x_concat = torch.swapaxes(x_concat, 0, 1)
    x_concat = x_concat.flatten(start_dim=0, end_dim=1)
    Utils.save_image_RGB(x_concat, ncol, output_f)


def get_config_():
    # 2.8.7.2
    config = get_config(None)
    dataset = "expression_CFEE_id_128_V2"
    config.train_dir = DATASETS_ROOT + f"/{dataset}/train"
    config.test_dir = DATASETS_ROOT + f"/{dataset}/test"
    config.models_dir = "../test/models"
    config.output_dir = "../test/output"
    config.encoder_grey = True
    config.code_dim = 32
    config.img_size = 128

    config.models_dir = "../test/models/CFEE/V2.9.0.0"
    config.eval_model_step = 85000

    return config


def draw_chart():
    config = get_config_()

    transform = DataLoader.create_transform_test(config)
    models_s = get_models(config, config.eval_model_step)
    # generate_data(config,step,models_s,transform)

    # happy = np.load("code_data/code_02.data.npy")
    # sad = np.load("code_data/code_03.data.npy")
    # angry = np.load("code_data/code_05.data.npy")
    # surprised = np.load("code_data/code_06.data.npy")
    #
    # happy_mean = happy.mean(axis=0, keepdims=False)
    # sad_mean = sad.mean(axis=0, keepdims=False)
    # angry_mean = angry.mean(axis=0, keepdims=False)
    # surprised_mean = surprised.mean(axis=0, keepdims=False)

    # show(happy, sad, angry)

    happy = np.load("code_data/code_11.data.npy")
    print("type(happy):", happy.shape)
    all = None
    for i in range(1, 27):
        d = np.load(f"code_data/code_{i:02d}.data.npy")
        if all is None:
            all = d
        else:
            all = np.concatenate((all, d), axis=0)
    happy = all
    sad = np.load("code_data/code_11.data.npy")
    angry = np.load("code_data/code_10.data.npy")

    sad_mean = sad.mean(axis=0, keepdims=False)
    angry_mean = angry.mean(axis=0, keepdims=False)

    dis1 = np.linalg.norm(sad_mean - angry_mean)
    dis2 = torch.dist(torch.from_numpy(sad_mean), torch.from_numpy(angry_mean), p=2)
    print(f"dis:{dis1}, {dis2}")

    show(happy, sad, angry)


def generate_code_face():
    config = get_config_()
    config.models_dir = "../test/models/CFEE"
    transform = DataLoader.create_transform_test(config)
    step = 15000
    models_s = get_models(config, step)

    code_face(transform, config, models_s)


def cut_code_face_CFEE():
    img = Image.open("output/code_face_CFEE_20.png")
    print(img.size)
    img1 = np.asarray(img.convert('RGB'))
    l = 0
    t = 0
    h = 544
    h = h - 128
    w = 567
    w = 567 + 128
    result = img1[t:t + h, l:l + w, :]
    im = Image.fromarray(result)
    im.save("code_face_CFEE_21.png", format=None)


if __name__ == '__main__':
    # generate_data_()
    draw_chart()
    # generate_code_face()
    # cut_code_face_CFEE()
