import cv2
import DataUtils
from munch import Munch
import matplotlib.pyplot as plt

expression_mapping = {
    "neutral": 1,
    "happy": 2,
    "sad": 3,
    "angry": 5,
    "surprised": 6,
    "disgusted": 7,
    "fearful": 8,
    "contemptuous": 9
}

gaze_mapping = {
    "frontal": 0,
    "left": 1,
    "right": 2,
}


def parse_file_name(filename):
    filename = filename.replace(".jpg", "")
    # print(filename)
    ss = filename.split("_")
    angle = ss[0].lower().replace("rafd", "")
    angle = int(angle)
    id = ss[1]
    race = ss[2].lower()
    gender = ss[3].lower()
    expression = expression_mapping[ss[4].lower()]
    gaze = gaze_mapping[ss[5].lower()]
    return angle, id, race, gender, expression, gaze


def crop(img):
    y = 90
    x = 30
    w = 681 - 2 * x
    img = img[y:y + w, x:x + w, :]
    return img


def get_file_name(angle, id, expression, gaze):
    return f"{id}_{angle}_{gaze}_{expression:02d}"


def split(source_dir, output_dir, save=False):
    if save:
        DataUtils.recreate_dir(output_dir)
        for c in ["a_n", "b_e"]:
            class_dir = output_dir + "/" + c
            DataUtils.recreate_dir(class_dir)

    stat = Munch()
    size = 256
    img_files = DataUtils.get_image_files(source_dir)
    angles = set()
    ids = set()
    races = set()
    total_count = 0
    outputs = set()
    for img_file in img_files:
        angle, id, race, gender, expression, gaze = parse_file_name(img_file)
        if angle == 0 or angle == 180:
            continue
        total_count += 1
        races.add(race)
        angles.add(angle)
        ids.add(id + "_" + gender)

        group_dir = "b_e"
        if expression == 1 and gaze == 0:
            group_dir = "a_n"
        # if angle == 90:
        #     group_dir = "a_n"
        out_file_name = get_file_name(angle, id, expression, gaze)
        outputs.add(out_file_name)
        target_img_file_full = f"{output_dir}/{group_dir}/{out_file_name}.png"
        print(f"Save {target_img_file_full}")
        if save:
            img = cv2.imread(source_dir + "/" + img_file)
            img = crop(img)
            img = cv2.resize(img, (size, size))
            cv2.imwrite(target_img_file_full, img)

        key = f"{id}"
        if key not in stat:
            stat[key] = 1
        else:
            stat[key] += 1

    # for key, count in stat.items():
    #     if count != 72:
    #         print(f"!!!{key}:{count}")
    # print(sorted(angles))
    print(f"total_count:{total_count}")
    print(f"total output count:{len(outputs)}")
    print(f"people count:{len(ids)}")
    print(f"each people count:{len(outputs) / len(ids)}")
    print(f"races:{races}")


root_dir = "/Users/xiaohanghu/Documents/Repositories/datasets"


def test_crop(source_dir):
    img_file = "Rafd090_30_Caucasian_male_disgusted_frontal.jpg"
    size = 256
    img = cv2.imread(source_dir + "/" + img_file)
    img = crop(img)
    img = cv2.resize(img, (size, size))
    print(img.shape)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# total output count:4824
# people count:67
# each people count:72
races: {'moroccan', 'caucasian', 'kid'}

source_dir = root_dir + "/RafD_full"
# test_crop()
# split(source_dir, root_dir + "/expression_RafD_angle_gaze", False)

# DataUtils.split_data(root_dir + "/expression_RafD_angle_gaze", root_dir + "/expression_RafD_angle_gaze_id_256", ids_test=["12", "15", "21", "57", "67", "43"])
# DataUtils.rszie_images(root_dir + "/expression_RafD_angle_gaze_id_256", root_dir + "/expression_RafD_angle_gaze_id_128", (128, 128))

# split(source_dir, root_dir + "/expression_RafD_gaze", True)
# DataUtils.split_data(root_dir + "/expression_RafD_gaze", root_dir + "/expression_RafD_gaze_id_256", ids_test=["12", "15", "21", "57", "67", "43"])
DataUtils.rszie_images(root_dir + "/expression_RafD_gaze_id_256", root_dir + "/expression_RafD_gaze_id_128", (128, 128))
