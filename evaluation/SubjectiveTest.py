import Demo
import DataLoader
from munch import Munch
import random


def sample_CFEE(each=3):
    # config = Demo.get_config_()
    # sample_getter_train, sample_getter_test = create_sample_getter(config)
    dir = Demo.DATASETS_ROOT + "/expression_CFEE_id_128/train" + "/b_e"
    fs = DataLoader.img_files(dir)
    fs = [f.replace(".png", "") for f in fs]
    cls_map = Munch()
    for f in fs:
        id, c = DataLoader.parse_file_name(f)
        if (c not in cls_map) or (cls_map[c] is None):
            cls_map[c] = []
        cls_imgs = cls_map[c]
        cls_imgs.append(f)

    each = 3
    fs_reference = []
    for fs_c in cls_map.values():
        fs_reference.extend(random.sample(fs_c, each))

    n = len(fs_reference)
    print(f"{n} for CFEE")
    fs_target = random.sample(fs, n)
    print(f"target:{fs_target}")
    print(f"reference:{fs_reference}")


def parse_file_name_RaFD(filename):
    # f"{id}_{angle}_{gaze}_{expression:02d}"
    filename = filename.replace(".png", "")
    # print(filename)
    ss = filename.split("_")
    id = ss[0]
    angle = int(ss[1])
    gaze = ss[2]
    expression = ss[3]
    return id, angle, gaze, expression


def sample_RaFD(each=3):
    # config = Demo.get_config_()
    # sample_getter_train, sample_getter_test = create_sample_getter(config)
    dir = Demo.DATASETS_ROOT + "/expression_RafD_gaze_id_128/train" + "/b_e"
    fs = DataLoader.img_files(dir)

    def filter(f):
        id, angle, gaze, expression = parse_file_name_RaFD(f)
        return angle == 90

    fs = [f.replace(".png", "") for f in fs if filter(f)]
    # print(fs)
    cls_map = Munch()
    for f in fs:
        id, c = DataLoader.parse_file_name(f)
        if (c not in cls_map) or (cls_map[c] is None):
            cls_map[c] = []
        cls_imgs = cls_map[c]
        cls_imgs.append(f)

    fs_reference = []
    for fs_c in cls_map.values():
        fs_reference.extend(random.sample(fs_c, each))

    n = len(fs_reference)
    print(f"{n} for RaFD")
    fs_target = random.sample(fs, n)
    print(f"target:{fs_target}")
    print(f"reference:{fs_reference}")


sample_CFEE(3)
