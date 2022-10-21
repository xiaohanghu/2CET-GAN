"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

from DataLoader import SingleFolderDataset
from evaluation import IS
from evaluation.EvalDataGenerator import generate_eval_data
from evaluation.FID import calculate_all_fid, calculate_fid_by_cls
from Model import create_model
from Utils import load_model, generate_output_and_save_
import Utils
from DataLoader import create_transform_test, img_files
import numpy as np
import torch
from munch import Munch
from main import get_config
from DataLoader import create_sample_getter
from PIL import Image
from DataLoader import parse_file_name
import cv2


def get_model(config, step):
    model, model_s = create_model(config)
    # load_models(config, models, "models", True, step)
    load_model(config, model_s, "model_s", True, step)
    del model
    return model_s


def generate_output_and_save(models_s, config, x_n, x_e, y_e, step, number, name):
    n_group = 8
    need_add_y = False
    n_col = 3
    display_by_col = False
    generate_output_and_save_(models_s, config, x_n, x_e, y_e, step, number, name, n_group, need_add_y, n_col,
                              display_by_col)


def generate_output_and_save_ENE(models_s, config, x_e, x_e_r, y_e_r, step, number, name):
    n_group = 8
    need_add_y = False
    n_col = 3
    display_by_col = False
    Utils.generate_output_and_save_ENE_(models_s, config, x_e, x_e_r, y_e_r, step, number, name, n_group, need_add_y,
                                        n_col,
                                        display_by_col)


def demo(config, sample_getter, step, n):
    models_s = get_model(config, config.eval_model_step)
    for i in range(n):
        sample = sample_getter.next_sample()
        generate_output_and_save(models_s, config, sample.x_n, sample.x_e, sample.y_e, step, i, f"demo")


@torch.no_grad()
def generate_fake(img_n, img_r, models_s, transform):
    # img = transform(img)
    c_e = models_s.encoder(torch.unsqueeze(img_r, dim=0))
    imgs = torch.unsqueeze(img_n, dim=0)
    imgs_new = models_s.generator(imgs, c)
    imgs_new = Utils.denormalize(imgs_new)
    img_new = imgs_new[0]
    # img_new = np.transpose(img_new, (1, 2, 0))
    # print(img_new.shape)
    img_new = img_new.mul(255).add_(0.5).clamp_(0, 255) \
        .permute(1, 2, 0) \
        .to('cpu', torch.uint8).numpy()
    return img_new


def get_image(root, file, transform, is_n=True):
    if is_n:
        path = root + "/a_n"
    else:
        path = root + "/b_e"
    path = path + "/" + file + ".png"
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img


def get_images(root, files, transform, is_n=True):
    imgs = []
    for file in files:
        imgs.append(get_image(root, file, transform, is_n))
    return torch.stack(imgs, dim=0)


DATASETS_ROOT = "/Users/xiaohanghu/Documents/Repositories/datasets"


def get_config_(eval_model_step):
    # 2.8.8
    config = get_config(None)
    config.train_dir = DATASETS_ROOT + "/expression_CFEE_id_128/train"
    config.test_dir = DATASETS_ROOT + "/expression_CFEE_id_128/test"
    config.models_dir = "../test/models"
    config.output_dir = "../test/output"
    config.encoder_grey = True
    config.code_dim = 32
    config.img_size = 128
    config.eval_model_step = eval_model_step
    return config


def generate_demo():
    # 2.8.8
    config = get_config_(eval_model_step=65000)
    _, sample_getter_test = create_sample_getter(config)

    transform = create_transform_test(config)

    ns = ["288_01", "288_01", "294_01", "294_01", "357_01", "357_01", "263_01", "263_01"]
    rs = ["227_02", "176_02", "133_05", "269_05", "113_16", "207_16", "200_26", "303_26"]
    x_n = get_images(config.train_dir, ns, transform, True)
    x_e = get_images(config.train_dir, rs, transform, False)
    print(x_n.shape)
    y_e = []
    for r in rs:
        _, y = parse_file_name(r)
        y_e.append(int(y))
    y_e = torch.LongTensor(y_e)

    # demo(config, sample_getter_test, config.eval_model_step, 10)

    models_s = get_model(config, config.eval_model_step)
    generate_output_and_save(models_s, config, x_n, x_e, y_e, config.eval_model_step, 0, f"demo")


def generate_demo_ENE():
    # 2.8.8
    config = get_config_(eval_model_step=65000)
    _, sample_getter_test = create_sample_getter(config)

    transform = create_transform_test(config)

    es = ["288_03", "288_05", "294_02", "294_06", "357_07", "357_08", "263_11", "263_20"]
    rs = ["227_02", "176_02", "133_05", "269_05", "113_16", "207_16", "200_26", "303_26"]
    x_e = get_images(config.train_dir, es, transform, False)
    x_e_r = get_images(config.train_dir, rs, transform, False)
    print(x_e.shape)
    y_e_r = []
    for r in rs:
        _, y = parse_file_name(r)
        y_e_r.append(int(y))
    y_e_r = torch.LongTensor(y_e_r)

    # demo(config, sample_getter_test, config.eval_model_step, 10)

    models_s = get_model(config, config.eval_model_step)
    generate_output_and_save_ENE(models_s, config, x_e, x_e_r, y_e_r, config.eval_model_step, 0, f"demo_ENE")


def generate_demo_test():
    # 2.8.8
    config = get_config_(eval_model_step=65000)
    _, sample_getter_test = create_sample_getter(config)

    transform = create_transform_test(config)
    is_test = True

    # ["122", "174", "185", "224", "272", "188", "187", "265"]
    ns = ["122_01", "188_01", "174_01", "187_01", "174_01", "122_01", "188_01", "187_01"]
    rs = ["265_02", "224_16", "272_05", "185_16", "174_26", "188_05", "187_02", "122_26"]
    x_n = get_images(config.test_dir, ns, transform, True)
    x_e = get_images(config.test_dir, rs, transform, False)
    print(x_n.shape)
    y_e = []
    for r in rs:
        _, y = parse_file_name(r)
        y_e.append(int(y))
    y_e = torch.LongTensor(y_e)

    # demo(config, sample_getter_test, config.eval_model_step, 10)

    models_s = get_model(config, config.eval_model_step)
    generate_output_and_save(models_s, config, x_n, x_e, y_e, config.eval_model_step, 0, f"demo_test")


def generate_demo_CFE():
    # 2.8.8
    config = get_config_(eval_model_step=65000)
    _, sample_getter_test = create_sample_getter(config)

    transform = create_transform_test(config)

    root = DATASETS_ROOT + "/expression_CFD_256"

    # ["122", "174", "185", "224", "272", "188", "187", "265"]
    ns = ["CFD-MF-346_n", "CFD-AF-218_n", "CFD-LM-212_n", "CFD-MM-306_n",
          "CFD-BM-249_n", "CFD-AM-208_n", "CFD-BF-013_n", "CFD-AF-210_n"]
    rs = ["CFD-WM-031_f", "CFD-WF-027_a", "CFD-WF-011_ho", "CFD-WM-029_hc",
          "CFD-BF-002_hc", "CFD-WF-012_f", "CFD-WM-003_ho", "CFD-BM-019_a"]
    x_n = get_images(root, ns, transform, True)
    x_e = get_images(root, rs, transform, False)
    print(x_n.shape)
    y_e = []
    for r in rs:
        _, y = parse_file_name(r)
        y_e.append(y)
        # y_e.append(int(y))
    # y_e = torch.LongTensor(y_e)

    # demo(config, sample_getter_test, config.eval_model_step, 10)

    models_s = get_model(config, config.eval_model_step)
    generate_output_and_save(models_s, config, x_n, x_e, y_e, config.eval_model_step, 0, f"demo_test")


def merge_image():
    file1 = "../test/output/030000_demo_0.jpg"
    file2 = "../test/output/065000_demo_0.jpg"
    img1 = np.asarray(Image.open(file1).convert('RGB'))
    img2 = np.asarray(Image.open(file2).convert('RGB'))
    print(img1.shape)
    result = np.zeros((256 * 4, 256 * 8, 3), dtype=img2.dtype)
    result[:256 * 3, :, :] = img2[:, :, :]
    result[256 * 3:, :, :] = img1[256 * 2:, :, :]
    im = Image.fromarray(result)
    im.save("result.jpg", format=None)
    # img1[128*3:]


def generate_demo_matrix(config, es, rs, name):
    _, sample_getter_test = create_sample_getter(config)

    transform = create_transform_test(config)

    rs = rs[:4]
    print(f"len:{len(rs)}")
    x_es = get_images(config.train_dir, es, transform, False)
    x_e_rs = get_images(config.train_dir, rs, transform, False)

    y_e_rs = []
    for r in rs:
        _, y = parse_file_name(r)
        y_e_rs.append(int(y))
    y_e_rs = torch.LongTensor(y_e_rs)

    models_s = get_model(config, config.eval_model_step)
    cs = models_s.encoder(x_e_rs)
    c_n_real = Utils.get_neutral_code(x_es.shape[0], x_es.dtype, config)
    print(f"x_es.shape:{x_es.shape}")
    x_ns = models_s.generator(x_es, c_n_real)

    out_img_size = 256
    result = np.full((out_img_size * (len(rs) + 1), out_img_size * (len(es) + 1), 3), 255, dtype=np.uint8)

    x_es_RGB = Utils.denormalize_RGB(x_es)
    x_es_RGB = Utils.resize_imgs(x_es_RGB, 2)
    Utils.save_image_RGB(x_es_RGB, len(x_es_RGB), f"output/demo_matrix_{name}_target.png")
    for c, x_n in enumerate(x_es_RGB.numpy()):
        row = 0
        col = c + 1
        result[(row) * out_img_size:(row + 1) * out_img_size, out_img_size * col:out_img_size * (col + 1),
        :] = x_n

    x_e_rs_RGB = Utils.denormalize_RGB(x_e_rs)
    x_e_rs_RGB = Utils.resize_imgs(x_e_rs_RGB, 2)
    Utils.save_image_RGB(x_e_rs_RGB, 1, f"output/demo_matrix_{name}_reference.png")
    for r, x_e in enumerate(x_e_rs_RGB.numpy()):
        row = r + 1
        col = 0
        result[(row) * out_img_size:(row + 1) * out_img_size, out_img_size * col:out_img_size * (col + 1),
        :] = x_e

    output = []
    for r, c in enumerate(cs):
        row = r + 1
        c_row = torch.unsqueeze(c, dim=0)
        c_row = c_row.expand(len(x_ns), -1)
        x_fake = models_s.generator(x_ns, c_row)
        x_fake = Utils.denormalize_RGB(x_fake)
        x_fake = Utils.resize_imgs(x_fake, 2)
        output.append(x_fake)
        # x_fake = torch.squeeze(x_fake, dim=1)
        # print(x_fake.shape)
        x_fake = x_fake.numpy()
        for c, x_fake_1 in enumerate(x_fake):
            col = c + 1
            result[(row) * out_img_size:(row + 1) * out_img_size, out_img_size * (col):out_img_size * (col + 1),
            :] = x_fake_1
    output = torch.stack(output, dim=0)
    output = output.flatten(start_dim=0, end_dim=1)
    # output = Utils.resize_imgs(output, 2)
    print(output.shape)
    Utils.save_image_RGB(output, len(es), f"output/demo_matrix_{name}_output.png")

    # for
    # models_s.generator(x_ns)
    im = Image.fromarray(result)
    im.save(f"output/demo_matrix_{name}.png", format=None)
    print(f"Save to : output/demo_matrix_{name}.png")


def generate_demo_matrix_RafD():
    # 2.8.8.2
    es = ["01_135_2_08", "64_90_1_09", "70_45_1_07",
          ]
    rs = ["23_135_0_02", "14_90_2_06", "63_45_1_05"]
    config = get_config_(eval_model_step=100000)
    config.train_dir = DATASETS_ROOT + "/expression_RafD_gaze_id_128/train"
    config.test_dir = DATASETS_ROOT + "/expression_RafD_gaze_id_128/test"
    config.models_dir = config.models_dir + "/RafD"
    generate_demo_matrix(config, es, rs, "RafD")


def generate_demo_matrix_CFEE():
    # 2.8.8.2
    # es = ["392_07", "114_09", "134_03",
    #       "277_10",
    #       # "209_10",
    #       # "266_10",
    #       # "361_10",
    #       # "376_10",
    #       # "183_10",
    #       "363_21",
    #       "181_11",
    #       ]
    es = ["130_11", "134_19", "363_09",
          # "181_11",
          ]
    rs = ["392_02", "110_16", "209_05"]
    config = get_config_(eval_model_step=60000)
    config.models_dir = config.models_dir
    generate_demo_matrix(config, es, rs, "CFEE")


def cut_demo_matrix():
    img = Image.open("output/demo_matrix_0.png")
    print(img.size)
    img1 = np.asarray(img)
    print(img1.min())
    print(img1.max())
    img = cv2.imread("demo_matrix_0.png")
    print(img.shape)
    w = 256 * 7 + 8 + 8
    h = 256 * 4 + 8 + 8
    t = 0
    l = 0
    print(w)
    print(h)
    result_image = img[t:t + h, l:l + w, :]
    cv2.imwrite("demo_matrix_1.png", result_image)


def cut_structure():
    img = Image.open("output/structure_0.png")
    print(img.size)
    img1 = np.asarray(img.convert('RGB'))
    l = 238
    result = img1[100:1080, l:3460 - l, :]
    im = Image.fromarray(result)
    im.save("structure_1.jpg", format=None)


def cut_demo_compare():
    img = Image.open("output/demo_compare_0.png")
    print(img.size)
    img1 = np.asarray(img.convert('RGB'))
    l = 355
    t = 289
    h = 1095
    w = 2175
    result = img1[t:t + h, l:l + w, :]
    im = Image.fromarray(result)
    im.save("demo_compare_1.png", format=None)


def cut_demo_compare_00():
    img = Image.open("output/demo_compare_00.png")
    print(img.size)
    img1 = np.asarray(img.convert('RGB'))
    l = 355
    t = 360
    h = 1080
    w = 2171
    result = img1[t:t + h, l:l + w, :]
    im = Image.fromarray(result)
    im.save("demo_compare_11.png", format=None)


def cut_demo_compare_test():
    img = Image.open("output/demo_compare_test_0.png")
    print(img.size)
    img1 = np.asarray(img.convert('RGB'))
    l = 210
    t = 158
    h = 828
    w = 2146
    result = img1[t:t + h, l:l + w, :]
    im = Image.fromarray(result)
    im.save("output/demo_compare_test_1.png", format=None)


if __name__ == '__main__':
    # generate_demo()
    # generate_demo_ENE()
    # generate_demo_test()
    # generate_demo_CFE()
    # merge_image()
    # generate_demo_matrix_RafD()
    # generate_demo_matrix_CFEE()
    # cut_demo_matrix()
    # cut_structure()
    # cut_demo_compare()
    # cut_demo_compare_00()
    cut_demo_compare_test()
