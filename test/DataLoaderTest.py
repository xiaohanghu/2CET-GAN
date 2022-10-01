"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import unittest

from torch import nn

from DataLoader import *

from Utils import show_img
from torchvision import transforms
import torch.nn.functional as F


class HighPass(nn.Module):
    def __init__(self):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[1., 1., 1.]
                                              , [1., -8., 1.]
                                              , [1., 1., 1.]], dtype=torch.float32))

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        print(f"filter.shape:{filter.shape}")
        print(f"x.size(1):{x.size(1)}")
        return F.conv2d(x, filter, stride=1, padding=1, groups=x.size(1))


class DataLoaderTest(unittest.TestCase):

    def show_img(self, imgs):
        n_imgs = len(imgs)
        plt.rcParams['figure.figsize'] = [4 * 2, 4]
        fig, axarr = plt.subplots(nrows=1, ncols=n_imgs)
        for i in range(n_imgs):
            axarr[i].axis('off')
            # axarr[i].imshow(np.transpose(vutils.make_grid(imgs[i], padding=2, normalize=True).cpu(), (1, 2, 0)))
            axarr[i].imshow(np.transpose(imgs[i], (1, 2, 0)))
        plt.show()

    def test_ImageFolder(self):
        ts = transforms.Compose([transforms.ToTensor()])
        dataset = ImageFolder("/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFEE_p_128/train", ts)
        print(dataset.targets)

        train_loader = data.DataLoader(dataset=dataset,
                                       batch_size=2,
                                       # num_workers=num_workers,
                                       pin_memory=True,
                                       drop_last=True)
        print(dataset.classes)
        highPass = HighPass()
        for n, (real_samples, labels) in enumerate(train_loader):
            print(f"real_samples.shape:{real_samples.shape}")
            real_samples = highPass(real_samples)
            print(f"real_samples.shape:{real_samples.shape}")
            self.show_img(real_samples)
            break

    def test(self):
        root_dir = "/Users/xiaohanghu/Documents/Repositories/datasets"
        output_dir = root_dir + "/expression_1.0/a_n"
        ts = transforms.Compose([transforms.ToTensor()])
        dataset = ExpressionDataset(output_dir, ts)
        # print(dataset.targets)

        train_loader = data.DataLoader(dataset=dataset,
                                       batch_size=77,
                                       # num_workers=num_workers,
                                       # sampler=sampler,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)

        for i, t in enumerate(train_loader):
            print(i, len(t[0]))
            show_img(t[0:-1])
            break

    def test_to_neutral_expression_file(self):
        root_dir = "/Users/xiaohanghu/Documents/Repositories/datasets"
        output_dir = root_dir + "/expression_1.0"
        ts = transforms.Compose([transforms.ToTensor()])
        dataset = ExpressionPairedDataset(output_dir, ts)

        f = "CFD-BF-001_a.png"
        print(dataset.to_neutral_expression_file(f))

        for i in range(10):
            print(random.randint(0, 2))

    def test_create_sample_getter(self):
        root_dir = "/Users/xiaohanghu/Documents/Repositories/datasets"
        output_dir = root_dir + "/expression_CFEE_1.0/train"
        config = Munch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.device = device
        config.train_dir = output_dir
        config.img_size = 256
        config.batch_size = 7

        sample_getter = create_sample_getter(config)
        sample = sample_getter.next_sample()
        print(sample.x_n.shape)
        sample.x_n = sample.x_n.mean(dim=1, keepdims=True)
        print(sample.x_n.shape)
        show_img([sample.x_n, sample.x_e])
