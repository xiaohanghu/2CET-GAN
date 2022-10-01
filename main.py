"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import argparse

from Train import train
import torch
from evaluation import Evaluation


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(parser=None):
    if None == parser:
        parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'eval_all'],
                        help='Train of evalution')
    parser.add_argument('--test', type=str2bool, default='False',
                        help='Whether it is test mode')
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Dir of training set')
    parser.add_argument('--test_dir', type=str, default='test',
                        help='Dir of test set')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Dir for output samples')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Dir for saving models')
    parser.add_argument('--eval_dir', type=str, default='eval',
                        help='Dir for saving evaluation samples')
    parser.add_argument('--eval_model_step', type=int, default=0,
                        help='Model step for evaluation')
    parser.add_argument('--eval_generate_data', type=str2bool, default='True',
                        help='Whether to generate new evaluation dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='Test batch size')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--total_iter', type=int, default='200000',
                        help='Total training iterations/steps')

    parser.add_argument('--img_size', type=int, default=128,
                        help='Square image size (pixels)')
    parser.add_argument('--code_dim', type=int, default='64',
                        help='Dimension of expression code')
    parser.add_argument('--to_grey', type=str2bool, default='False',
                        help='Whether images are converted to gray')
    parser.add_argument('--encoder_grey', type=str2bool, default='True',
                        help='Whether the Encoder converts the image to grayscale')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lr_e', type=float, default=0.000001,
                        help='Learning rate for the Encoder')

    parser.add_argument('--lambda_adv_n', type=float, default=1.,
                        help='Weight for adversarial loss on neutral face')
    parser.add_argument('--lambda_adv_e', type=float, default=1.,
                        help='Weight for adversarial loss on emotional face')
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for zero-centered gradient penalty (R1 regularization) for real images')
    parser.add_argument('--lambda_cyc_e_config', type=str, default=1.,
                        help='Weight for cycle consistency loss on emotional faces. Format: {1:int},{2:int},{3:float},{4:float}. meaning: increase to {3} at step {1}, and then increase/decrease to {4} at step {2}.')
    parser.add_argument('--lambda_ds_e_config', type=str, default=1.,
                        help='Weight for diversity sensitive loss on emotional faces. Format: {1:int},{2:int},{3:float},{4:float}. meaning: increase to {3} at step {1}, and then increase/decrease to {4} at step {2}.')
    parser.add_argument('--lambda_c_e_config', type=str, default=0.001,
                        help='Weight for expression code loss on emotional faces. Format: {1:int},{2:int},{3:float},{4:float}. meaning: increase to {3} at step {1}, and then increase/decrease to {4} at step {2}.')

    parser.add_argument('--lambda_c', type=float, default=1,
                        help='Weight for Encoder')
    parser.add_argument('--lambda_cyc_n', type=float, default=1,
                        help='Weight for cycle consistency loss on neutral faces.')

    parser.add_argument('--log_every', type=int, default=10, help='Do logging after every n steps')
    parser.add_argument('--output_every', type=int, default=5000, help='Generate sample after every n steps')
    parser.add_argument('--save_every', type=int, default=10000, help='Save model after every n steps')

    config = parser.parse_args()
    config.beta1 = 0.0
    config.beta2 = 0.99
    config.weight_decay = 1e-4
    config.img_dim = 3
    config.num_domains = 2
    if config.to_grey:
        config.img_dim = 1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends and torch.backends.mps and torch.backends.mps.is_available():
    #     device = 'mps'
    device = torch.device(device)
    config.device = device

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = get_config(parser)
    if config.mode == 'train':
        train(config)
    elif config.mode == 'eval':
        Evaluation.evaluate(config)
    elif config.mode == 'eval_all':
        Evaluation.evaluate_all(config)
