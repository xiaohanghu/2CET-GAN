"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

Models was heavily based on models from StarGAN v2: https://github.com/clovaai/stargan-v2/blob/master/core/model.py
Users should be careful about adopting these models in any commercial matters.
https://github.com/clovaai/stargan-v2#license
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munch import Munch
import copy


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


# adaptive instance normalization to inject s into G
class AdaIN(nn.Module):
    def __init__(self, code_dim, num_features):  # num_features relate to the image
        super().__init__()
        self.fc = nn.Linear(code_dim, num_features * 2)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, c):
        h = self.fc(c)
        h = h.view(h.size(0), h.size(1), 1, 1)
        # gamma:variance, beta: mean
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, code_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, code_dim)

    def _build_weights(self, dim_in, dim_out, code_dim):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(code_dim, dim_in)
        self.norm2 = AdaIN(code_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, c):
        x = self.norm1(x, c)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, c)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, c):
        out = self._residual(x, c)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, img_dim=3, img_size=256, code_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(img_dim, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, img_dim, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, code_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, code_dim))

    def forward(self, x, c):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, c)
        return self.to_rgb(x)


class EncoderBlk(nn.Module):
    def __init__(self, img_dim=3, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(img_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.dim_out = dim_out

    def forward(self, x):
        return self.shared(x)


class ExpressionEncoder(nn.Module):
    def __init__(self, encoderBlk, code_dim=64, encoder_grey=False):
        super().__init__()
        self.encoder_grey = encoder_grey
        self.shared = encoderBlk
        # self.unshared = nn.Linear(sharedEncoder.dim_out, code_dim)
        self.unshared = nn.Sequential(
            # nn.Linear(sharedEncoder.dim_out, sharedEncoder.dim_out),
            nn.Linear(encoderBlk.dim_out, code_dim))

    def forward(self, x):
        if self.encoder_grey:
            # to greyscale:
            x = x.mean(dim=1, keepdims=True)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = self.unshared(h)
        # out = torch.stack(out, dim=1)  # (batch, num_domains, code_dim)
        return out


class Discriminator(nn.Module):
    def __init__(self, encoderBlk, num_domains=2, ):
        super().__init__()
        self.main = nn.Sequential(encoderBlk
                                  , nn.Conv2d(encoderBlk.dim_out, num_domains, 1, 1, 0))

    def forward(self, x, y):
        # print(f"x.shape:{x.shape}")
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # print(f"x.shape:{x.shape}")
        # print(f"y.shape:{y.shape}")
        # print(idx)
        out = out[idx, y]  # (batch)
        # print(f"Discriminator out.shape: {out.shape}")
        return out


def create_model(config):
    generator = nn.DataParallel(
        Generator(img_dim=config.img_dim, img_size=config.img_size, code_dim=config.code_dim)).to(config.device)

    encoder_img_dim = config.img_dim
    if config.encoder_grey:
        encoder_img_dim = 1
    encoder = nn.DataParallel(
        ExpressionEncoder(EncoderBlk(img_dim=encoder_img_dim, img_size=config.img_size), code_dim=config.code_dim,
                          encoder_grey=config.encoder_grey)).to(
        config.device)
    discriminator = nn.DataParallel(
        Discriminator(EncoderBlk(img_dim=config.img_dim, img_size=config.img_size), num_domains=config.num_domains)).to(
        config.device)
    generator_s = copy.deepcopy(generator)
    # latent_encoder_s = copy.deepcopy(latent_encoder)
    encoder_s = copy.deepcopy(encoder)

    modules = Munch(generator=generator,
                    # latent_encoder=latent_encoder,
                    encoder=encoder,
                    discriminator=discriminator)
    # stable modules
    modules_s = Munch(generator=generator_s,
                      # latent_encoder=latent_encoder_s,
                      encoder=encoder_s)

    return modules, modules_s
