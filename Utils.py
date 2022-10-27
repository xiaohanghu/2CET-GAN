"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import os
from os import path
import shutil
from torch import FloatTensor
from PIL import Image
import cv2

sec_min = 60
sec_hour = sec_min * 60
sec_day = sec_hour * 24


def format_sec(sec):
    sec = int(sec)
    days = int(sec // sec_day)
    hours = int((sec % sec_day) // sec_hour)
    minutes = int((sec % sec_hour) // sec_min)
    seconds = int((sec % sec_min))
    time = f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"
    return time


class Timer:
    def __init__(self, resume_iter, total_iter):
        self.start = time.time()
        self.count = 0
        self.ms_min = 60
        self.ms_hour = self.ms_min * 60
        self.ms_day = self.ms_hour * 24
        self.resume_iter = resume_iter
        self.total_iter = total_iter

    def increase(self):
        self.count += 1

    def avg_cost(self):
        current = time.time()
        cost = current - self.start
        return cost / self.count

    def expect(self):
        current = time.time()
        cost = current - self.start
        e = cost * (self.total_iter - self.resume_iter - self.count) / self.count
        return self.format(e)

    def takes(self):
        current = time.time()
        cost = current - self.start
        return self.format(cost)

    def format(self, sec):
        return format_sec(sec)


def set_lr(optim):
    for g in optim.param_groups:
        g['lr'] = 0.001


def copy_model_average(model, model_s):
    """
    Computing moving average of model and save the result to model_s
    :param model: original model
    :param model_s: stable model
    :return: None
    """
    for name, m_s in model_s.items():
        copy_average(model[name], m_s, beta=0.999)


def copy_average(model, model_avg, beta=0.999):
    for param, param_avg in zip(model.parameters(), model_avg.parameters()):
        param_avg.data = torch.lerp(param.data, param_avg.data, beta)


def show_imgs(imgs):
    n_imgs = len(imgs)
    plt.rcParams['figure.figsize'] = [3 * n_imgs, 3]
    fig, axarr = plt.subplots(nrows=1, ncols=n_imgs)
    for i in range(n_imgs):
        axarr[i].axis('off')
        axarr[i].imshow(imgs[i])
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.show()


def denormalize(x):
    x = (x + 1) / 2
    return x.clamp_(0, 1)


def get_neutral_code(n_sample, dtype, config):
    c = torch.zeros((n_sample, config.code_dim), dtype=dtype).to(config.device)
    return c


def get_neutral_code(n_sample, dtype, config):
    c = torch.zeros((n_sample, config.code_dim), dtype=dtype).to(config.device)
    return c


# CODE_MIN = -0.5
# CODE_MAX = 0.5
# # CODE_DIFF_EXP = (CODE_MAX - CODE_MIN) / 3.  # code diff expectation
# CODE_DIFF_EXP_FRACTION = 3 / (CODE_MAX - CODE_MIN)


def generate_rand_code(n_sample, config):
    if config.code_distribution == "normal":
        c = torch.normal(0, config.code_max, size=(n_sample, config.code_dim)) \
            .to(config.device)
    else:
        # std = sqrt( (1+1)**2/12 ) = 0.5773
        # std = sqrt( (0.5+0.5)**2/12 ) = 0.288675
        # std = sqrt( (0.2+0.2)**2/12 ) = 0.1154
        c = FloatTensor(np.random.uniform(config.code_min, config.code_max, (n_sample, config.code_dim)))
        # std = sqrt( (0.25+0.25)**2/12 ) = 0.144
        # c = FloatTensor(np.random.uniform(-0.25, 0.25, (n_sample, config.code_dim)))
        c = c.to(config.device)

    return c


def add_y_to_imgs_(img, y, need_add_y):
    if need_add_y:
        return add_y_to_imgs(img, y)
    return img


@torch.no_grad()
def generate_output_img(model_s, config, x_n, x_e, y_e, need_add_y, n_col, display_by_col=True):
    cols = []
    cols.append(denormalize_RGB(x_n))
    cols.append(add_y_to_imgs_(denormalize_RGB(x_e), y_e, need_add_y))
    c_e = model_s.encoder(x_e)
    x_fake_e = model_s.generator(x_n, c_e)
    cols.append(add_y_to_imgs_(denormalize_RGB(x_fake_e), y_e, need_add_y))

    if n_col > 3:
        c_e_1 = generate_rand_code(x_n.size(0), config)
        x_fake_e_1 = model_s.generator(x_n, c_e_1)
        cols.append(denormalize_RGB(x_fake_e_1))

        c_n_real = torch.full_like(c_e, fill_value=0)
        x_fake_n = model_s.generator(x_e, c_n_real)

        x_n_back = model_s.generator(x_fake_e_1, c_n_real)
        x_e_back = model_s.generator(x_fake_n, c_e)

        cols.append(denormalize_RGB(x_n_back))
        cols.append(denormalize_RGB(x_fake_n))
        cols.append(add_y_to_imgs_(denormalize_RGB(x_e_back), y_e, need_add_y))

    x_concat = torch.stack(cols, dim=0)
    ncol = x_concat.size(0)
    if display_by_col:
        x_concat = torch.swapaxes(x_concat, 0, 1)
    else:
        ncol = x_concat.size(1)

    x_concat = x_concat.flatten(start_dim=0, end_dim=1)

    return x_concat, ncol


@torch.no_grad()
def generate_output_img_ENE(model_s, config, x_e, x_e_r, y_e_r, need_add_y, n_col, display_by_col=True):
    cols = []

    c_e_r = model_s.encoder(x_e_r)
    c_n_real = torch.full_like(c_e_r, fill_value=0)
    x_fake_n = model_s.generator(x_e, c_n_real)
    x_fake_e = model_s.generator(x_fake_n, c_e_r)

    cols.append(denormalize_RGB(x_e))
    # cols.append(denormalize_RGB(x_fake_n))
    cols.append(add_y_to_imgs_(denormalize_RGB(x_e_r), y_e_r, need_add_y))
    cols.append(add_y_to_imgs_(denormalize_RGB(x_fake_e), y_e_r, need_add_y))

    x_concat = torch.stack(cols, dim=0)
    ncol = x_concat.size(0)
    if display_by_col:
        x_concat = torch.swapaxes(x_concat, 0, 1)
    else:
        ncol = x_concat.size(1)

    x_concat = x_concat.flatten(start_dim=0, end_dim=1)

    return x_concat, ncol


def generate_output_and_save(model_s, config, x_n, x_e, y_e, step, number, name):
    n_group = config.test_batch_size
    need_add_y = True
    n_col = 7
    generate_output_and_save_(model_s, config, x_n, x_e, y_e, step, number, name, n_group, need_add_y, n_col)


@torch.no_grad()
def generate_output_and_save_(model_s, config, x_n, x_e, y_e, step, number, name, n_group, need_add_y, n_col,
                              display_by_col=True):
    x_n = x_n[0:n_group]
    x_e = x_e[0:n_group]
    y_e = y_e[0:n_group]
    images, ncol = generate_output_img(model_s, config, x_n, x_e, y_e, need_add_y, n_col, display_by_col)
    filename1 = path.join(config.output_dir, f"{step:06d}_{name}_{number}.jpg")
    print(f"Generate output: {filename1}.")
    if config.img_size == 128:
        images = resize_imgs(images, 2)
    save_image_RGB(images, ncol, filename1)


@torch.no_grad()
def generate_output_and_save_ENE_(model_s, config, x_e, x_e_r, y_e_r, step, number, name, n_group, need_add_y, n_col,
                                  display_by_col=True):
    x_e = x_e[0:n_group]
    x_e_r = x_e_r[0:n_group]
    y_e_r = y_e_r[0:n_group]
    images, ncol = generate_output_img_ENE(model_s, config, x_e, x_e_r, y_e_r, need_add_y, n_col, display_by_col)
    filename1 = path.join(config.output_dir, f"{step:06d}_{name}_{number}.jpg")
    print(f"Generate output: {filename1}.")
    if config.img_size == 128:
        images = resize_imgs(images, 2)
    save_image_RGB(images, ncol, filename1)


def generate_output_test(model_s, config, step, sample_getter_test):
    sample1 = sample_getter_test.next_sample()
    sample2 = sample_getter_test.next_sample()
    generate_output(model_s, config, sample1, sample2, step, "test")


def generate_output(model_s, config, sample1, sample2, step, name):
    generate_output_and_save(model_s, config, sample1.x_n, sample1.x_e, sample1.y_e, step, 1, name)
    generate_output_and_save(model_s, config, sample1.x_n, sample2.x_e, sample2.y_e, step, 2, name)
    # generate_output_and_save(model_s, config, sample2.x_n, sample2.x_e, step, 3, name)


@torch.no_grad()
def generate_output_vertical(model_s, config, sample, step):
    x_n, x_e = sample.x_n, sample.x_e
    c_e = model_s.encoder(x_e)
    x_fake_e = model_s.generator(x_n, c_e)

    c_n_real = torch.full_like(c_e, fill_value=0)
    x_fake_n = model_s.generator(x_e, c_n_real)

    x_concat = [x_n, x_e, x_fake_e, x_fake_n]
    x_concat = torch.cat(x_concat, dim=0)

    ncol = x_n.size(0)

    filename = path.join(config.output_dir, f"{step:06d}_sample.jpg")
    print(f"Generate output: {filename}.")
    save_image(x_concat, ncol, filename)


def resize_imgs(imgs, scale):
    ims = []
    for im in imgs:
        ims.append(cv2.resize(im.numpy(), (int(im.shape[1] * scale), int(im.shape[0] * scale))))
    ims = np.array(ims)
    return torch.from_numpy(ims)


def denormalize_RGB(x):
    return denormalize(x).mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8)


def add_y_to_imgs(x, y):
    x_ref_arr = []
    font_size = 1.0
    for i in range(len(x)):
        y_ = y[i]
        x_ = x[i].numpy().copy()
        # print(f"shape:{x_.shape}, max:{x_.max()}, min:{x_.min()}")
        text = f"{y_}"
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_size, 1)[0]
        cv2.putText(x_, text, (2, 2 + t_size[1] + 1), cv2.FONT_HERSHEY_PLAIN, font_size, [0, 0, 0], 1)
        x_ref_arr.append(x_)
    x = np.array(x_ref_arr)
    x = torch.from_numpy(x).to("cpu", torch.uint8)
    return x


def save_image_RGB(imgs, ncol, filename):
    imgs = imgs.permute(0, 3, 1, 2)
    grid = vutils.make_grid(imgs, ncol, padding=0)
    grid = grid.permute(1, 2, 0)
    im = Image.fromarray(grid.numpy())
    im.save(filename, format=None)


def save_image(x, ncol, filename):
    """

    :param x: could be the direct output of the generator
    :param ncol:
    :param filename:
    :return:
    """
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def get_model_filename(model_name, step):
    return f"{step:06d}_{model_name}.ckpt"


def get_step(file_name):
    if file_name.endswith(".ckpt"):
        return int(file_name.split("_")[0])
    return None


def get_max_step(models_dir):
    max_step = 0
    for file in os.listdir(models_dir):
        if file.endswith("model.ckpt"):
            step = get_step(file)
            max_step = max(max_step, step)
    return max_step


def delete_model_(models_dir, model_name, step):
    fname = os.path.join(models_dir, get_model_filename(model_name, step))
    if os.path.exists(fname):
        os.remove(os.path.join(models_dir, fname))


def delete_model(models_dir, step):
    delete_model_(models_dir, "model", step)
    delete_model_(models_dir, "model_s", step)
    delete_model_(models_dir, "optims", step)


def should_save(step, save_every):
    return step == 1 or (step % save_every == 0)


def delete_model_backup(models_dir, del_step, save_every):
    if should_save(del_step, save_every):
        return
    delete_model(models_dir, del_step)  # only keep last one


def save_model(config, model, model_name, parallel, step):
    os.makedirs(os.path.dirname(config.models_dir), exist_ok=True)
    fname = get_model_filename(model_name, step)
    fname = os.path.join(config.models_dir, fname)
    print(f'Saving model [{model_name}] into {fname}')
    outdict = {}
    for name, module in model.items():
        if parallel:
            module = module.module
        outdict[name] = module.state_dict()

    torch.save(outdict, fname)


def save_model_all(config, model, model_s, optims, step):
    save_model(config, model, "model", True, step)
    save_model(config, model_s, "model_s", True, step)
    save_model(config, optims, "optims", False, step)


def get_model_full_file(config, model_name, step):
    fname = get_model_filename(model_name, step)
    fname = os.path.join(config.models_dir, fname)
    return fname


def model_exist(config, model_name, step):
    fname = get_model_full_file(config, model_name, step)
    return os.path.exists(fname)


def load_model(config, model, model_name, parallel, step):
    try:
        fname = get_model_full_file(config, model_name, step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print(f'Loading model from {fname}')
        module_dict = torch.load(fname, map_location=config.device)
        for name, module in model.items():
            print(f'Loading model [{model_name}.{name}]...')
            try:
                if parallel:
                    module = module.module
                module.load_state_dict(module_dict[name])
            except RuntimeError as err:
                print(f'Load model [{model_name}.{name}] RuntimeError!', err)
    except Exception as err:
        print(f'Load model [{model_name}] Exception!', err)


def load_model_all(config, model, model_s, optims, step):
    load_model(config, model, "model", True, step)
    load_model(config, model_s, "model_s", True, step)
    load_model(config, optims, "optims", False, step)


def losses_average(losses_avg, losses, alpha=0.02):
    '''
    exponential weighted moving average
    '''
    for name, value_new in losses.items():
        if value_new is None:
            continue

        if (name not in losses_avg) or (losses_avg[name] is None):
            losses_avg[name] = value_new
            continue

        value_avg = losses_avg[name]
        losses_avg[name] = (1 - alpha) * value_avg + alpha * value_new


def losses_to_str(losses, keys=None):
    strs = []
    # for key in sorted(losses.keys()):
    for key in losses.keys():
        if keys is not None and key not in keys:
            continue
        value = losses[key]
        if value is None:
            continue
        if key in ['c_e_std']:
            strs.append(f'{key}: {value:.7f}')
        else:
            strs.append(f'{key}: {value:.4f}')
    return ', '.join(strs)


def generate_log(step, timer, d_losses_avg, g_losses_avg, config):
    log = f"[{step}/{config.total_iter}] takes: {timer.takes()}, avg: {timer.avg_cost():.2f} sec, expect finish in: {timer.expect()}."
    log += f"\r\n D loss ---- " + losses_to_str(d_losses_avg)
    code_keys = {"c_e_r_mean", "c_e_r_abs_mean", "c_e_r_abs_max", "c_e_x_mean", "c_e_x_abs_mean", "c_e_x_abs_max"}
    g_keys = set(g_losses_avg.keys()) - code_keys
    log += f"\r\n G loss ---- " + losses_to_str(g_losses_avg, g_keys)
    log += f"\r\n Code sts -- " + losses_to_str(g_losses_avg, code_keys)
    # log += f"\r\n E info ---- " + losses_to_str(e_losses_avg)
    return log


def create_dir(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
        return

    if overwrite and os.path.isdir(path):
        shutil.rmtree(path)
        os.makedirs(path)


def recreate_dir(path):
    create_dir(path, True)


def parse_lambda_config(formular):
    vs = formular.split(",")
    start_step = int(vs[0])
    end_step = int(vs[1])
    start_v = float(vs[2])
    end_v = float(vs[3])
    return (start_step, end_step, start_v, end_v)


def calculate_lambda(current_step, start_step, end_step, start_v, end_v):
    if current_step < start_step:
        return start_v * (current_step / start_step)
    if current_step > end_step:
        return end_v
    return start_v + (end_v - start_v) * (current_step - start_step) / (end_step - start_step)


def code_distribution(code):
    c_mean = torch.mean(code, dim=0).detach()
    c_abs = torch.abs(code).detach()

    c_mean = torch.mean(c_mean).item()
    c_abs_mean = torch.mean(c_abs).item()
    c_abs_max = torch.max(c_abs).item()
    return c_mean, c_abs_mean, c_abs_max

# import math
# print(math.sqrt((0.20 + 0.20) ** 2 / 12))
# code_range="[-0.2,0.2]"
# code_min,code_max = code_range[1:-1].split(',')
# print(float(code_min))
# print(float(code_max))

# print(145000*30)
# print(145000*30/5743)
