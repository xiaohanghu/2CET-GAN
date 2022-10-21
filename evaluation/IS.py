"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.


"""

import numpy as np
from munch import Munch
from torchvision import models
from DataLoader import create_data_loader_eval
from tqdm import tqdm
import torch
from torch.nn import functional as F
from scipy.stats import entropy


# Inception Score (IS)

def calculate_inception_score_by_predict(p_yx):
    p_y = np.mean(p_yx, axis=0)
    scores = []
    for i in range(p_yx.shape[0]):
        pyx = p_yx[i, :]
        scores.append(entropy(pyx, p_y))
    is_score = np.exp(np.mean(scores))
    return is_score


def pred(x, inception):
    x = inception(x)
    result = F.softmax(x, dim=1).data
    return result.cpu().numpy()


def calculate_inception_score(dir, config, batch_size=50):
    # load inception v3 model
    inception = models.inception_v3(pretrained=True).eval().to(config.device)

    loader = create_data_loader_eval(dir, None, config.img_size, batch_size)
    scores = []
    for (x, id, cls, _) in tqdm(loader, total=len(loader)):
        # print(f"x.shape111:{x.shape}")
        yhat = pred(x.to(config.device), inception)
        # print(f"x.shape:{x.shape}, yhat.shape:{yhat.shape}")
        is_score = calculate_inception_score_by_predict(yhat)
        scores.append(is_score)
        # print(f"is_score:{is_score}")
    return np.mean(scores), np.std(scores)


# return actvs


def test():
    data_dir = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFEE_128/test"
    eval_dir = "/Users/xiaohanghu/Documents/Repositories/expression-GAN/V2.6.6/test/eval/test"
    eval_dir = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFEE_128/test"
    config = Munch()
    config.img_size = 128
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # calculate_inception_score_by_dir(eval_dir + "/b_e_r", config, batch_size=10)
    IS = calculate_inception_score(eval_dir + "/a_n", config, batch_size=10)
    print("IS:", IS)
    # calculate_inception_score_by_dir(data_dir + "/b_e", config, batch_size=10)


if __name__ == '__main__':
    test()
    None
# p_yx_1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# p_yx_2 = np.array([[.33, .33, .33], [.33, .33, .33], [.33, .33, .33]])
# print("p_yx_1:",calculate_inception_score_by_predict(p_yx_1)) #3.0
# print("p_yx_2:",calculate_inception_score_by_predict(p_yx_2)) #1.0
