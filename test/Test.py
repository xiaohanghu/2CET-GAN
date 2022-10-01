"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import unittest
import torch
from munch import Munch
from torchvision.datasets import ImageFolder
from torch.utils import data

import Utils
from DataLoader import create_sample_getter
from Train import train
from Utils import save_image, Timer, losses_to_str, losses_average
import numpy as np
from main import get_config
from torchvision import transforms
import cv2
import os
import random
from torch import FloatTensor


class Test(unittest.TestCase):
    DATASETS_ROOT = "/Users/xiaohanghu/Documents/Repositories/datasets"
    EG_VERSION = "V2.8.7"
    EG_DATASET = "expression_CFEE_128"

    def test_lerp(self):
        d1 = torch.tensor([2.])
        d2 = torch.tensor([12.])
        for i in range(10000):
            d2 = torch.lerp(d1, d2, 0.999)
            print(d2)

    def test_data_loader(self):
        config = get_config(None)
        config.train_dir = Test.DATASETS_ROOT + "/FaceWarehouse/image_256"
        sample_getter = create_sample_getter(config)

        sample = sample_getter.next_sample()

        imgs = torch.cat([sample.x_org, sample.x_ref_1], dim=0)
        print(imgs.shape)
        save_image(imgs, config.batch_size, "../../test.png")

    def test(self):
        print(1e-4)
        timer = Timer()
        sec = 200000 * 1.26
        print(timer.format(sec))

    def test_main(self):
        config = get_config()
        config.test = True
        config.train_dir = Test.DATASETS_ROOT + f"/{Test.EG_DATASET}/train"
        config.test_dir = Test.DATASETS_ROOT + f"/{Test.EG_DATASET}/test"
        config.img_size = 128
        config.lambda_cyc_e_config = "0, 100000, 1.0, 2.0"
        config.lambda_c_e_config = "6000, 100000, 6.0, 20.0"
        config.lambda_ds_e_config = "0, 20000, 2.0, 0.00001"
        config.code_dim = 32
        config.to_grey = False
        config.num_domains = 2
        config.batch_size = 2
        config.output_dir = f"/Users/xiaohanghu/Documents/Repositories/expression-GAN/{Test.EG_VERSION}/test/output"
        config.models_dir = f"/Users/xiaohanghu/Documents/Repositories/expression-GAN/{Test.EG_VERSION}/test/models"
        config.resume_iter = 0
        train(config)

    def test_bincount(self):
        bc = np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
        print(bc)

    def test_zeros(self):
        x_e = torch.tensor([[1], [1]])
        print(x_e)
        y = torch.zeros(x_e.shape[0])
        print(y)

    def test_losses_to_str(self):
        losses = Munch(c_n=1.12323,
                       fake_e=0.234243,
                       fake_n=0.3434,
                       adv_e=0.45452342,
                       code=0.93249)
        print(losses_to_str(losses))

    def test_file_existes(self):
        file = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_1.0/b/14_16.png"
        print(os.path.exists(file))

    def test_random(self):
        print(random.random())
        arr = np.array([[[0, 0.1], [1, 1.1], [2, 2.1]], [[3, 3.1], [4, 4.1], [5, 5.1]]])
        print(arr)
        print(arr.shape)
        arr = np.swapaxes(arr, 0, 1)
        print(arr)
        print(arr.shape)

    def test_losses_average(self):
        losses = Munch(a=2, b=1)
        losses_avg = Munch()
        alpha = 0.1
        losses_average(losses_avg, losses, alpha=alpha)
        losses_average(losses_avg, losses, alpha=alpha)
        print(losses_avg)

    def test_calculate_lambda(self):
        start_step = 6000
        end_step = 100000
        start_v = 4
        end_v = 40.

        self.assertEqual(0.0, Utils.calculate_lambda(0, start_step, end_step, start_v, end_v))
        self.assertEqual(start_v / 2, Utils.calculate_lambda(3000, start_step, end_step, start_v, end_v))
        self.assertEqual(start_v, Utils.calculate_lambda(6000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(7000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(10000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(20000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(50000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(100000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(200000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(60000, start_step, end_step, start_v, end_v))

    def test_calculate_lambda_desc(self):
        start_step = 2000
        end_step = 20000
        start_v = 1
        end_v = 0.0001

        print(Utils.calculate_lambda(0, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(5, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(10, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(1000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(2000, start_step, end_step, start_v, end_v))
        print(Utils.calculate_lambda(10000, start_step, end_step, start_v, end_v))
        print(f"{Utils.calculate_lambda(20000, start_step, end_step, start_v, end_v):.4f}")
        print(Utils.calculate_lambda(30000, start_step, end_step, start_v, end_v))

    def test_parse_lambda_formular(self):
        formular = "5000,200000,1.0,20.0"
        lambda_config = Utils.parse_lambda_config(formular)
        print(f"{lambda_config}")
        print(Utils.calculate_lambda(20000, *lambda_config))

    def test_calculate_time(self):
        avg = 4.  # sec
        steps = 44000
        sec = avg * steps
        print(Utils.format_sec(sec))
        avg = 2.28  # sec
        steps = 200000
        sec = avg * steps
        print(Utils.format_sec(sec))
        import platform
        print(platform.platform())

    def test_code(self):
        # std = sqrt( (0.5+0.5)**2/12 ) = 0.288675
        c1 = FloatTensor(np.random.uniform(-0.5, 0.5, (8, 1000)))
        c2 = FloatTensor(np.random.uniform(-0.5, 0.5, (8, 1000)))
        c_e_diff = torch.mean(torch.abs(c1.detach() - c2.detach()))
        # expectation of diff is (0.5+0.5)/3
        print(c_e_diff)

    def test_save_image_RGB(self):
        config = get_config(None)
        config.batch_size = 3
        config.train_dir = Test.DATASETS_ROOT + "/expression_CFEE_128/train/"
        config.test_dir = Test.DATASETS_ROOT + "/expression_CFEE_128/test/"
        sample_getter, _ = create_sample_getter(config)

        sample = sample_getter.next_sample()
        x_n = sample.x_n
        x_e = sample.x_e
        y_e = sample.y_e
        image = torch.cat(
            [Utils.denormalize_RGB(x_n), Utils.add_y_to_imgs(Utils.denormalize_RGB(x_e), y_e)], dim=0)
        # imgs = torch.cat([sample.x_org], dim=0)
        # save_image(x_refs, config.batch_size, "test.png")
        scale = 2
        ims = []
        for im in image:
            print(f"image.shape{image.shape}")
            ims.append(cv2.resize(im.numpy(), (int(im.shape[1] * scale),int(im.shape[0] * scale))))
        # image = cv2.resize(image.numpy().copy(), (image.shape[1] * 2, image.shape[0] * 2))
        image = torch.tensor(ims)
        filename = "output/test.png"
        Utils.save_image_RGB(image, config.batch_size, filename)
