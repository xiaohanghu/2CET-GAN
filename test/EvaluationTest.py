"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import unittest
import torch
from munch import Munch

import DataLoader
from evaluation import EvalDataGenerator
from evaluation import Evaluation
from main import get_config


class Test(unittest.TestCase):
    DATASETS_ROOT = "/Users/xiaohanghu/Documents/Repositories/datasets"
    EG_VERSION = "V2.6.6"
    EG_DATASET = "expression_CFEE_128"

    def get_config(self):
        config = get_config()
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.num_domains = 2
        config.test = False
        config.to_grey = False
        config.img_size = 128
        config.img_size = 256
        config.code_dim = 64
        config.encoder_grey = False
        config.eval_model_step = 44000
        config.models_dir = "../test/models"
        config.test_dir = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFEE_128/test"
        config.train_dir = "/Users/xiaohanghu/Documents/Repositories/datasets/expression_CFEE_128/train"
        return config

    def test_SingleFolderDataset(self):
        config = self.get_config()
        dataset = DataLoader.SingleFolderDataset(config.train_dir + "/b_e", None, None)
        print("dataset.get_cls_index_map():", sorted(dataset.get_cls_index_map().keys()))

    def test_EvalDataGenerator(self):
        config = self.get_config()
        models_s = EvalDataGenerator.get_models(config, 44000)
        transform = EvalDataGenerator.create_transform_test(config)
        # config.eval_dir = "/Users/xiaohanghu/Documents/Repositories/expression-GAN/V2.6.6/test/eval"
        # EvalDataGenerator.generate_eval(config.test_dir, config.eval_dir + "/test", models_s, transform, config)

        config.eval_dir = "/Users/xiaohanghu/Documents/Repositories/expression-GAN/V2.6.6/test/eval/train"
        EvalDataGenerator.generate_eval(config.train_dir, config.eval_dir + "/train", models_s, transform, config)

    def test_evaluate_encoder(self):
        config = self.get_config()
        models_s = Evaluation.get_models(config, config.eval_model_step)
        transform = DataLoader.create_transform_test(config)

        Evaluation.evaluate_encoder(config.test_dir, models_s, transform)
        # Evaluation.evaluate_encoder(config.train_dir, models_s, transform)
