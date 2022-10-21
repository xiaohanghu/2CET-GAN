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
from Utils import load_model, model_exist
from DataLoader import create_transform_test, img_files
import numpy as np
import torch
from munch import Munch


def get_models(config, step):
    models, models_s = create_model(config)
    # load_models(config, models, "models", True, step)
    models_name = "model_s"
    if not model_exist(config, models_name, step):
        raise Exception(f'Model [{models_name}:{step}] dose not exist!')
    load_model(config, models_s, models_name, True, step)
    del models
    return models_s


@torch.no_grad()
def evaluate_encoder(source_data_dir, models_s, transform):
    e_dir = source_data_dir + "/b_e"
    print(f"evaluate_encoder() e_dir:{e_dir}")
    e_dataset = SingleFolderDataset(e_dir, None, transform)
    cls_index_map = e_dataset.get_cls_index_map()
    result = []
    for cls, img_indexes in sorted(cls_index_map.items()):
        c_e_s = []
        result_cls = Munch()
        result_cls_arr = []
        result.append(result_cls)
        print(f"Start to calculate class [{cls}].")
        for img_index in img_indexes:
            img_e, _, _, _ = e_dataset[img_index]
            c_e = models_s.encoder(torch.unsqueeze(img_e, dim=0))
            c_e_s.append(c_e.cpu().detach().numpy()[0])
        c_e_s = np.array(c_e_s)
        c_e_std_by_c = c_e_s.std(axis=0)  # std of each colum
        c_e_std_by_c = c_e_std_by_c.mean()
        c_e_std_by_r = c_e_s.std(axis=1).mean()
        c_e_mean = c_e_s.mean()
        result_cls.cls = cls
        result_cls.mean = c_e_mean
        result_cls.std_by_c = c_e_std_by_c
        result_cls.std_by_r = c_e_std_by_r
        result_cls_arr.append([c_e_mean, c_e_std_by_c, c_e_std_by_r])

    result_cls_arr = np.array(result_cls_arr)
    result_cls_arr = result_cls_arr.mean(axis=0)

    result_all = Munch()
    result_all["cls"] = "all"
    result_all["mean"] = result_cls_arr[0]
    result_all["std_by_c"] = result_cls_arr[1]
    result_all["std_by_r"] = result_cls_arr[2]
    result.append(result_all)

    return result


def append_report(report, text):
    report += "\r\n"
    report += text
    return report


def evaluate(config):
    models_s = get_models(config, config.eval_model_step)
    transform = create_transform_test(config)
    if config.eval_generate_data:
        num_each_cls = 3
        generate_eval_data(num_each_cls, models_s, transform, config)

    matrix = Munch()
    matrix.fid_n, matrix.fid_e_z, matrix.fid_e_r = calculate_all_fid(config.test_dir, config.eval_dir + "/test", config,
                                                                     batch_size=50)
    matrix.fid_n_1, matrix.fid_e_z_1, matrix.fid_e_r_1 = calculate_all_fid(config.train_dir, config.eval_dir + "/train",
                                                                           config,
                                                                           batch_size=50)

    matrix.cls_fids = calculate_fid_by_cls(config.train_dir + "/b_e", config.eval_dir + "/train/b_e_r", config)

    matrix.encoder_figures_test = evaluate_encoder(config.test_dir, models_s, transform)
    matrix.encoder_figures_train = evaluate_encoder(config.train_dir, models_s, transform)

    matrix.is_test_n_mean, matrix.is_test_n_std = IS.calculate_inception_score(config.test_dir + "/a_n", config)
    matrix.is_test_e_mean, matrix.is_test_e_std = IS.calculate_inception_score(config.test_dir + "/b_e", config)
    matrix.is_eval_test_n_mean, matrix.is_eval_test_n_std = IS.calculate_inception_score(config.eval_dir + "/test/a_n",
                                                                                         config)
    matrix.is_eval_test_e_r_mean, matrix.is_eval_test_e_r_std = IS.calculate_inception_score(
        config.eval_dir + "/test/b_e_r",
        config)
    matrix.is_eval_test_e_z_mean, matrix.is_eval_test_e_z_std = IS.calculate_inception_score(
        config.eval_dir + "/test/b_e_z",
        config)

    matrix.is_train_n_mean, matrix.is_train_n_std = IS.calculate_inception_score(config.train_dir + "/a_n", config)
    matrix.is_train_e_mean, matrix.is_train_e_std = IS.calculate_inception_score(config.train_dir + "/b_e", config)
    matrix.is_eval_train_n_mean, matrix.is_eval_train_n_std = IS.calculate_inception_score(
        config.eval_dir + "/train/a_n", config)
    matrix.is_eval_train_e_r_mean, matrix.is_eval_train_e_r_std = IS.calculate_inception_score(
        config.eval_dir + "/train/b_e_r",
        config)
    matrix.is_eval_train_e_z_mean, matrix.is_eval_train_e_z_std = IS.calculate_inception_score(
        config.eval_dir + "/train/b_e_z",
        config)

    report = "\r\n------------------------------------------------------------------------"
    report = append_report(report, f"REPORT")
    report = append_report(report, "FID--------------------------------")
    report = append_report(report,
                           f"FID on test dataset: fid_n={matrix.fid_n:.1f}, fid_e_z={matrix.fid_e_z:.1f}, fid_e_r={matrix.fid_e_r:.1f}")
    report = append_report(report,
                           f"FID on train dataset: fid_n={matrix.fid_n_1:.1f}, fid_e_z={matrix.fid_e_z_1:.1f}, fid_e_r={matrix.fid_e_r_1:.1f}")

    report = append_report(report, f"FID on train dataset by class:")
    for cls, fid in matrix.cls_fids.items():
        report = append_report(report, f"  Class:{cls}, fid:{fid:.1f}")

    report = append_report(report, "")
    report = append_report(report, "Encoder--------------------------------")
    report = append_report(report, f"Encoder on test dataset:")
    for result_cls in matrix.encoder_figures_test:
        report = append_report(report,
                               f"  Class:{result_cls.cls}, std between code: {result_cls.std_by_c:.5f}, std in code: {result_cls.std_by_r:.5f}, mean:{result_cls.mean:.5f}")

    report = append_report(report, f"Encoder on train dataset:")
    for result_cls in matrix.encoder_figures_train:
        report = append_report(report,
                               f"  Class:{result_cls.cls}, std between code: {result_cls.std_by_c:.5f}, std in code: {result_cls.std_by_r:.5f}, mean:{result_cls.mean:.5f}")

    report = append_report(report, "")
    report = append_report(report, "IS--------------------------------")
    report = append_report(report,
                           f"is_test_n_mean:{matrix.is_test_n_mean:.2f}, is_test_n_is_std:{matrix.is_test_n_std:.2f}")
    report = append_report(report,
                           f"is_test_e_mean:{matrix.is_test_e_mean:.2f}, is_test_e_is_std:{matrix.is_test_e_std:.2f}")
    report = append_report(report,
                           f"is_eval_test_n_mean:{matrix.is_eval_test_n_mean:.2f}, is_eval_test_n_is_std:{matrix.is_eval_test_n_std:.2f}")
    report = append_report(report,
                           f"is_eval_test_e_r_mean:{matrix.is_eval_test_e_r_mean:.2f}, is_eval_test_e_r_is_std:{matrix.is_eval_test_e_r_std:.2f}")
    report = append_report(report,
                           f"is_eval_test_e_z_mean:{matrix.is_eval_test_e_z_mean:.2f}, is_eval_test_e_z_is_std:{matrix.is_eval_test_e_z_std:.2f}")
    report = append_report(report, "")
    report = append_report(report,
                           f"is_train_n_mean:{matrix.is_train_n_mean:.2f}, is_train_n_is_std:{matrix.is_train_n_std:.2f}")
    report = append_report(report,
                           f"is_train_e_mean:{matrix.is_train_e_mean:.2f}, is_train_e_is_std:{matrix.is_train_e_std:.2f}")
    report = append_report(report,
                           f"is_eval_train_n_mean:{matrix.is_eval_train_n_mean:.2f}, is_eval_train_n_is_std:{matrix.is_eval_train_n_std:.2f}")
    report = append_report(report,
                           f"is_eval_train_e_r_mean:{matrix.is_eval_train_e_r_mean:.2f}, is_eval_train_e_r_is_std:{matrix.is_eval_train_e_r_std:.2f}")
    report = append_report(report,
                           f"is_eval_train_e_z_mean:{matrix.is_eval_train_e_z_mean:.2f}, is_eval_train_e_z_is_std:{matrix.is_eval_train_e_z_std:.2f}")
    config.logger_eval.log(report)


def evaluate_all(config):
    num_each_cls = 3
    transform = create_transform_test(config)
    save_every = config.save_every
    config.eval_model_step = config.eval_model_step

    # fid_e_z
    fid_e_z_test = []
    fid_e_z_train = []
    fid_e_r_test = []
    fid_e_r_train = []

    while (True):
        models_s = get_models(config, config.eval_model_step)
        if models_s is None:
            break

        generate_eval_data(num_each_cls, models_s, transform, config)

        matrix = Munch()
        matrix.fid_n, matrix.fid_e_z, matrix.fid_e_r = calculate_all_fid(config.test_dir, config.eval_dir + "/test",
                                                                         config,
                                                                         batch_size=50)
        matrix.fid_n_1, matrix.fid_e_z_1, matrix.fid_e_r_1 = calculate_all_fid(config.train_dir,
                                                                               config.eval_dir + "/train",
                                                                               config,
                                                                               batch_size=50)
        report = "\r\n------------------------------------------------------------------------"
        report = append_report(report,
                               f"step {config.eval_model_step}:")
        report = append_report(report,
                               f"  On test: fid_n={matrix.fid_n:.1f}, fid_e_z={matrix.fid_e_z:.1f}, fid_e_r={matrix.fid_e_r:.1f}")
        report = append_report(report,
                               f"  On train: fid_n={matrix.fid_n_1:.1f}, fid_e_z={matrix.fid_e_z_1:.1f}, fid_e_r={matrix.fid_e_r_1:.1f}")

        fid_e_z_test.append(matrix.fid_e_z)
        fid_e_z_train.append(matrix.fid_e_z_1)

        fid_e_r_test.append(matrix.fid_e_r)
        fid_e_r_train.append(matrix.fid_e_r_1)

        config.eval_model_step = config.eval_model_step + save_every

    report = append_report(report, f"fid_e_z_test = {np.around(fid_e_z_test, 1)}")
    report = append_report(report, f"fid_e_z_train = {np.around(fid_e_z_train, 1)}")
    report = append_report(report, f"fid_e_r_test = {np.around(fid_e_r_test, 1)}")
    report = append_report(report, f"fid_e_r_train = {np.around(fid_e_r_train, 1)}")
    config.logger_eval.log(report)
