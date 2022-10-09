"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import torch
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms

PICTURE_SIZE = 64

transform = transforms.Compose([
    # transforms.CenterCrop(178),
    transforms.Resize(PICTURE_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dir = "./dataset/celeba/"
filename = dir + "identity_Celeba.txt"

# own_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

# stat face numbers of same people
map = {}
with open(filename, 'r') as file_read:
    for line in file_read:
        id = line.split()[1]
        if id in map:
            map[id] = map[id] + 1
        else:
            map[id] = 1

print(map.values())
l = sorted(list(map.values()))
l.reverse()
print(l)  # 35, 35, 35, 34, 34, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31


