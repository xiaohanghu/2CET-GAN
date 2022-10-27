"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.


The format of the dataset:
root:
  |
  +---train:
  |    |
  |    +---a_n:
  |    |    |---{id1}_{class1}.png
  |    |    |---{id2}_{class1}.png
  |    |    |--- ...
  |    +---b_e:
  |    |    |---{id1}_{class2}.png
  |    |    |---{id2}_{class3}.png
  |    |    |--- ...
  +---test:
  |    +---a_n:
  |    |    |---{id3}_{class2}.png
  |    |    |--- ...
  |    +---b_e:
  |    |    |---{id4}_{class3}.png
  |    |    |--- ...

"""

from torch.utils import data
from PIL import Image
import random
import os
from torchvision import transforms
from munch import Munch


def img_files(dname):
    # fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
    #                       for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    files = os.listdir(dname)
    files = [f for f in files if f.endswith(".png")]
    return files


def parse_file_name(filename):
    filename = filename.rsplit('/', 1)[-1]
    filename = filename.replace(".png", "")
    # print(filename)
    ss = filename.split("_")
    id = ss[0]
    c = ss[-1]
    return id, c


class SingleFolderDataset(data.Dataset):
    def __init__(self, dir, file_paths=None, transform=None):
        if file_paths is not None:
            self.samples = file_paths
        else:
            self.samples = [dir + "/" + f for f in img_files(dir)]
        self.samples.sort()
        self.transform = transform
        self.targets = None
        # print(f"Create SingleFolderDataset. dir:{dir}, len:{len(self.samples)}")

    def getitem(self, index, return_img=True):
        file = self.samples[index]
        full_name = file.rsplit('/', 1)[-1]
        full_name = full_name.replace(".png", "")
        id, cls = parse_file_name(full_name)
        if not return_img:
            return id, cls
        img = Image.open(file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, id, cls, full_name

    def get_cls_index_map(self):
        n_e = self.__len__()
        cls_index_map = Munch()
        for index_e_i in range(n_e):
            id_r, cls_r = self.getitem(index_e_i, False)
            if (cls_r not in cls_index_map) or (cls_index_map[cls_r] is None):
                cls_index_map[cls_r] = []
            cls_imgs = cls_index_map[cls_r]
            cls_imgs.append(index_e_i)
        return cls_index_map

    def get_filepath(self, index):
        return self.samples[index]

    def __getitem__(self, index):
        return self.getitem(index, True)

    def __len__(self):
        return len(self.samples)


class ExpressionPairedDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets, self.n_neutral, self.n_emotional = self._make_dataset(root)
        self.transform = transform

    def to_neutral_expression_file(self, file_e):
        id = file_e.split("_")[0]
        return id + "_n.png"

    def _make_dataset(self, root):
        dir_n = os.path.join(root, "a_n")
        dir_e = os.path.join(root, "b_e")
        files_n = img_files(dir_n)
        files_e = img_files(dir_e)

        n_n = len(files_n)
        files_n = files_n[0:n_n // 2]  # reduce neutral face as ref

        files_all = files_n + files_e
        n_n = len(files_n)
        n_e = len(files_e)
        print(f"Number of neutral faces:{n_n}, number of emotional faces:{n_e}")
        n_all = len(files_all)

        files_n_fullpath = []
        files_e_fullpath = []
        files_r_fullpath = []
        labels = []
        for file_e in files_e:
            file_n = self.to_neutral_expression_file(file_e)
            file_e_fullpath = os.path.join(dir_e, file_e)
            files_e_fullpath.append(file_e_fullpath)
            file_n_fullpath = os.path.join(dir_n, file_n)
            files_n_fullpath.append(file_n_fullpath)
            reference_i = random.randint(0, n_all - 1)
            file_r = files_all[reference_i]
            # print(files_all[n_n - 1])
            # print(files_all[n_n])
            if reference_i < n_n:
                file_r_fullpath = os.path.join(dir_n, file_r)
                labels.append(0)
            else:
                file_r_fullpath = os.path.join(dir_e, file_r)
                labels.append(1)
            files_r_fullpath.append(file_r_fullpath)

        samples = list(zip(files_n_fullpath, files_e_fullpath, files_r_fullpath))
        # print(f"len(files_e):{len(files_e)}")
        # print(f"len(samples):{len(samples)}")

        return samples, labels, n_n, n_e

    def __getitem__(self, index):
        # print(f"index:{index}")
        f_n, f_e, f_r = self.samples[index]
        label = self.targets[index]
        img_n = Image.open(f_n).convert('RGB')
        img_e = Image.open(f_e).convert('RGB')
        img_r = Image.open(f_r).convert('RGB')
        if self.transform is not None:
            img_n = self.transform(img_n)
            img_e = self.transform(img_e)
            img_r = self.transform(img_r)
        return img_n, img_e, img_r, label

    def __len__(self):
        return len(self.targets)


class ExpressionDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def to_neutral_expression_file(self, file_e):
        id = file_e.split("_")[0]
        return id + "_n.png"

    def _make_dataset(self, root):
        files = img_files(root)

        n = len(files)
        # print(f"Number of images under [{root}] is [{n}]")

        samples = []
        labels = []
        for file in files:
            id, c = parse_file_name(file)
            file_fullpath = os.path.join(root, file)
            samples.append(file_fullpath)
            # print(files_all[n_n - 1])
            # print(files_all[n_n])
            labels.append(c)

        return samples, labels

    def __getitem__(self, index):
        # print(f"index:{index}")
        f = self.samples[index]
        label = self.targets[index]
        img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.targets)


class SampleGetter:
    def __init__(self, config, data_loader_n, data_loader_e):
        self.device = config.device
        self.data_loader_n = data_loader_n
        self.data_loader_e = data_loader_e
        self.data_loader_n_iter = iter(self.data_loader_n)
        self.data_loader_e_iter = iter(self.data_loader_e)

    def next_sample(self):
        x_n, y_n = self.next_n()
        x_e, y_e = self.next_e()
        # x_e_2, y_e_2 = self.next_e()

        sample = Munch()
        sample.x_n = x_n
        sample.x_e = x_e
        # sample.x_e_2 = x_e_2
        sample.y_e = y_e
        # sample.y_e_2 = y_e_2
        # sample.z = z.to(self.device)
        # sample.z_2 = z_2.to(self.device)
        return sample

    def next_n(self):
        try:
            x_n, y_n = next(self.data_loader_n_iter)
            return x_n, y_n
        except (StopIteration):
            self.data_loader_n_iter = iter(self.data_loader_n)
        return self.next_n()

    def next_e(self):
        try:
            x_e, y_e = next(self.data_loader_e_iter)
            return x_e, y_e
        except (StopIteration):
            self.data_loader_e_iter = iter(self.data_loader_e)
        return self.next_e()


def create_transform_test(config):
    transform = transforms.Compose([
        transforms.Resize([config.img_size, config.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    return transform


def create_data_loader_train(config, subdir):
    prob = 0.5
    crop = transforms.RandomResizedCrop(
        config.img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform_arr = [rand_crop,
                     transforms.Resize([config.img_size, config.img_size])]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if config.to_grey:
        transform_arr.append(transforms.Grayscale(num_output_channels=1))
        mean = [0.5]
        std = [0.5]
    transform_arr += [transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=mean,
                                           std=std)]
    transform = transforms.Compose(transform_arr)

    dir = os.path.join(config.train_dir, subdir)
    dataset = ExpressionDataset(dir, transform)

    # sampler = RandomSampler()
    return data.DataLoader(dataset=dataset,
                           batch_size=config.batch_size,
                           # sampler=sampler,
                           shuffle=True,
                           # num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def create_data_loader_test(config, subdir):
    transform_arr = [
        transforms.Resize([config.img_size, config.img_size])]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if config.to_grey:
        transform_arr.append(transforms.Grayscale(num_output_channels=1))
        mean = [0.5]
        std = [0.5]
    transform_arr += [transforms.ToTensor(),
                      transforms.Normalize(mean=mean,
                                           std=std)]
    transform = transforms.Compose(transform_arr)

    dir = os.path.join(config.test_dir, subdir)
    dataset = ExpressionDataset(dir, transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=config.test_batch_size,
                           # sampler=sampler,
                           shuffle=True,
                           # num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def create_data_loader_eval(root, img_paths=None, img_size=256, batch_size=32,
                            imagenet_normalize=True, shuffle=True,
                            num_workers=4, drop_last=False):
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = SingleFolderDataset(root, img_paths, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def create_sample_getter(config):
    train_loader_n = create_data_loader_train(config, "a_n", )
    train_loader_e = create_data_loader_train(config, "b_e", )
    config.sample_num_n = len(train_loader_n.dataset)
    config.sample_num_e = len(train_loader_e.dataset)
    config.sample_n_ratio = config.sample_num_n / config.sample_num_e
    print(
        f"number of samples: n = {config.sample_num_n}, e={config.sample_num_e}, n/e={config.sample_n_ratio:.4f}")

    print("Create dataset loader.")
    sample_getter_train = SampleGetter(config, train_loader_n, train_loader_e)

    train_loader_n_test = create_data_loader_test(config, "a_n", )
    train_loader_e_test = create_data_loader_test(config, "b_e", )
    sample_getter_test = SampleGetter(config, train_loader_n_test, train_loader_e_test)

    return sample_getter_train, sample_getter_test
