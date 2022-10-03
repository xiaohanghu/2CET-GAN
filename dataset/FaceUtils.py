"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import math

import face_recognition
import matplotlib.pyplot as plt
import numpy as np


def get_max_face_location(face_locations):
    if (len(face_locations) == 1):
        return face_locations[0]

    max_h = -1
    max_face_location = None
    for face_location in face_locations:
        y, right, bottom, x = face_location
        h = bottom - y
        # w = right - x
        if h > max_h:
            max_h = h
            max_face_location = face_location
    return max_face_location


def location_mark(image, face_location, border_w):
    y, right, bottom, x = face_location
    image[y:bottom, x:x + border_w, :] = [0, 255, 0]
    image[y:bottom, right:right + border_w, :] = [0, 255, 0]
    image[y:y + border_w, x:right, :] = [0, 255, 0]
    image[bottom:bottom + border_w, x:right, :] = [0, 255, 0]


# square
def location_to_square(location):
    y, right, bottom, x = location
    h = bottom - y
    w = right - x
    offset = (h - w) // 2
    x = x - offset
    right = right + offset
    w = right - x
    right = right + (h - w)
    return (y, right, bottom, x)


def location_up(location, up):
    y, right, bottom, x = location
    h = bottom - y
    up_pixel = int(up * h)
    y = y - up_pixel
    bottom = bottom - up_pixel
    # print(up_pixel, y,bottom)
    return (y, right, bottom, x)


def location_scale(location, scale):
    y, right, bottom, x = location
    h = bottom - y
    w = right - x
    w_1 = int(w * scale)
    h_1 = int(h * scale)
    w_offset = (w_1 - w) // 2
    h_offset = (h_1 - h) // 2
    y = y - h_offset
    bottom = bottom + h_offset
    x = x - w_offset
    right = right + w_offset
    return (y, right, bottom, x)


def crop(image, location):
    y, right, bottom, x = location
    return image[y:bottom, x:right, :]


def location_scale_by_pixel(face_location, offset):
    y, right, bottom, x = face_location
    y = y + offset
    x = x + offset
    right = right - offset
    bottom = bottom - offset
    return (y, right, bottom, x)


def location_scale_auto_move(face_location, scale, img_h, img_w):
    face_location_r = location_scale(face_location, scale)
    y, right, bottom, x = face_location_r
    h = bottom - y
    if h > img_h:
        offset = math.ceil((h - img_h) / 2)
        y, right, bottom, x = location_scale_by_pixel(face_location_r, offset)
    w = right - x
    if w > img_w:
        offset = math.ceil((w - img_w) / 2)
        y, right, bottom, x = location_scale_by_pixel(face_location_r, offset)
    if x < 0:
        offset = -x
        x = x + offset
        right = right + offset
    if y < 0:
        offset = -y
        y = y + offset
        bottom = bottom + offset
    if right > img_w:
        offset = right - img_w
        right = right - offset
        x = x - offset
    if bottom > img_h:
        offset = bottom - img_h
        bottom = bottom - offset
        y = y - offset
    return (y, right, bottom, x)


def location_scale_auto_decrease(face_location, scale, img_h, img_w):
    face_location_r = location_scale(face_location, scale)
    y, right, bottom, x = face_location_r
    if x < 0 or y < 0 or right > img_w or bottom > img_h:
        scale = scale - 0.01
        return location_scale_auto_decrease(face_location, scale, img_h, img_w)
    return face_location_r


def extract_face(image, up=-0.05, scale=1.9):
    """

    :param file:
    :return: RGB
    """
    # image = face_recognition.load_image_file(file)
    img_h = image.shape[0]
    img_w = image.shape[1]
    # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
    # face_location = get_max_face_location(face_locations)
    face_location = get_face_location_by_landmarks(image)
    face_location = location_to_square(face_location)
    # print(f"location_to_square: {face_location}")
    face_location = location_up(face_location, up)
    # print(f"location_up: {face_location}")
    # face_location = location_scale_auto_decrease(face_location, scale, img_h, img_w)
    face_location = location_scale_auto_move(face_location, scale, img_h, img_w)

    # location_mark(image, face_location, 2)
    image = crop(image, face_location)
    return image


def extract_face_fixed(img):
    """
    for CFD
    :param img:
    :return:
    """
    # print(f"Read image [{img_file}]")
    h = img.shape[0]
    offset = int(h * 0.02)
    img = img[offset:, :, :]

    w_min = (img.shape[1] - img.shape[0]) // 2
    w_max = img.shape[1] - w_min
    # if img.shape[0] != 1718 or img.shape[1] != 2444:
    #     print(f"Image [{img_file}] has a wrong size, shape:{img.shape}!")
    # continue

    img = img[:, w_min:w_max]
    # print(img.shape)
    return img


def mark_face_location(image):
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1
                                                     # ,model="cnn"
                                                     )
    face_location = get_max_face_location(face_locations)
    location_mark(image, face_location, 2)


def mark_point(image, x, y, size):
    image[(y - size):(y + size), (x - size):(x + size), :] = [0, 255, 0]


def mark_landmarks_small(image):
    landmarks = face_recognition.face_landmarks(image, model="small")
    mark_landmarks(image, landmarks)


def mark_landmarks(image, landmarks):
    size = 10
    for landmark in landmarks:
        for points in landmark.values():
            for point in points:
                mark_point(image, point[0], point[1], size)


def get_face_location_by_landmarks(image):
    landmarks = face_recognition.face_landmarks(image, model="large")
    xys = (0, 0, 0, 0)
    for landmark in landmarks:
        min_x = image.shape[1] + 1
        max_x = -1
        min_x_y = image.shape[0] + 1
        max_x_y = -1
        for points in landmark.values():
            for point in points:
                x = point[0]
                y = point[1]
                if x < min_x:
                    min_x = x
                    min_x_y = y
                if x > max_x:
                    max_x = x
                    max_x_y = y
        xys0 = (min_x, max_x, min_x_y, max_x_y)
        if (xys0[1] - xys0[0]) > (xys[1] - xys[0]):  # multiple faces: get the largest one
            xys = xys0

    min_x, max_x, min_x_y, max_x_y = xys
    w = max_x - min_x

    offset = w // 2
    c_x = (min_x + max_x) // 2
    c_y = (min_x_y + max_x_y) // 2
    # return (y, right, bottom, x)
    return (c_y - offset, c_x + offset, c_y + offset, c_x - offset)


def get_face_location_by_landmarks_simple(image):
    landmarks = face_recognition.face_landmarks(image, model="small")
    map = landmarks[0]
    nose_tip = map['nose_tip']
    left_eye = map['left_eye']
    right_eye = map['right_eye']
    eye_points = left_eye + right_eye
    x_s = [i[0] for i in eye_points]
    min_x = min(x_s)
    max_x = max(x_s)
    y_s = []
    y_s.append(right_eye[0][1])
    y_s.append(left_eye[0][1])
    # y_s.append(nose_tip[0][1])
    # min_y = min(y_s)
    # mark_point(image, min_x, min_y, 10)
    # mark_point(image, max_x, min_y, 10)
    w = max_x - min_x
    offset = w // 2
    c_x = nose_tip[0][0]
    c_y = int(np.mean(y_s))
    # return (y, right, bottom, x)
    return (c_y - offset, c_x + offset, c_y + offset, c_x - offset)
