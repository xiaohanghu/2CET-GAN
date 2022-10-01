"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
"""

import unittest

from FaceUtils import *


class FaceUtilsTest(unittest.TestCase):
    root_dir = "/Users/xiaohanghu/Documents/Repositories/datasets"

    def test_get_face_location_by_landmarks(self):
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/09_110_2555.jpg"
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/09_128_4191.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/10_110_2559.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/06_111_2638.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/23_178_6632.jpg"

        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/01-04/Images1/03_190_8097.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/17-20/Images5/20_206_9936.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/13-16/Images4/13_121_3512.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/04_221.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/26_222.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/02_236.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/260_19.png.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/260_19.png.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/21-26/Images6/268_19.png.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/13-16/Images4/13_268_6288.jpg"

        scale = 1.9
        image1 = face_recognition.load_image_file(file1)
        img_h = image1.shape[0]
        img_w = image1.shape[1]
        print("size: ", img_h, img_w)
        face_location1 = get_face_location_by_landmarks(image1)
        image2 = face_recognition.load_image_file(file2)
        face_location2 = get_face_location_by_landmarks(image2)
        # face_location1 = location_scale_auto_decrease(face_location1, scale, img_h, img_w)
        # face_location2 = location_scale_auto_decrease(face_location2, scale, img_h, img_w)
        face_location1 = location_scale_auto_move(face_location1, scale, img_h, img_w)
        face_location2 = location_scale_auto_move(face_location2, scale, img_h, img_w)
        location_mark(image1, face_location1, 2)
        location_mark(image2, face_location2, 2)

        plt.rcParams['figure.figsize'] = [15, 6]
        f, axarr = plt.subplots(nrows=1, ncols=3)
        # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axarr[0].imshow(image1)
        axarr[1].imshow(image2)
        axarr[2].imshow(image2)
        plt.show()

    def test_mark_landmarks(self):
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/09_110_2555.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/10_110_2559.jpg"
        file3 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/05_140_5449.jpg"

        image1 = face_recognition.load_image_file(file1)
        mark_landmarks_small(image1)
        image2 = face_recognition.load_image_file(file2)
        mark_landmarks_small(image2)
        image3 = face_recognition.load_image_file(file3)
        mark_landmarks_small(image3)

        plt.rcParams['figure.figsize'] = [15, 6]
        f, axarr = plt.subplots(nrows=1, ncols=3)
        # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axarr[0].imshow(image1)
        axarr[1].imshow(image2)
        axarr[2].imshow(image3)
        plt.show()

    def test_face_locations(self):
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/09_110_2555.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/10_110_2559.jpg"

        image1 = face_recognition.load_image_file(file1)
        mark_face_location(image1)
        image2 = face_recognition.load_image_file(file2)
        mark_face_location(image2)

        plt.rcParams['figure.figsize'] = [15, 6]
        f, axarr = plt.subplots(nrows=1, ncols=3)
        # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axarr[0].imshow(image1)
        axarr[1].imshow(image2)
        axarr[2].imshow(image2)
        plt.show()

    def mark_detail(self, file, up, scale):
        image = face_recognition.load_image_file(file)
        print(image.shape)
        img_h = image.shape[0]
        img_w = image.shape[1]
        # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model='hog')
        face_location = get_face_location_by_landmarks(image)
        landmarks = face_recognition.face_landmarks(image, model="large")
        mark_landmarks(image, landmarks)
        print(f"face_location: {face_location}")
        face_location = location_to_square(face_location)
        print(f"location_to_square: {face_location}")
        face_location = location_up(face_location, up)
        print(f"location_up: {face_location}")
        face_location = location_scale_auto_decrease(face_location, scale, img_h, img_w)
        # print(f"location_scale: {face_location}")

        location_mark(image, face_location, 2)
        return image

    def test_detail(self):
        file1 = FaceUtilsTest.root_dir + "/CFD_3.0/Images/CFD/AF-200/CFD-AF-200-228-N.jpg"
        file1 = FaceUtilsTest.root_dir + "/CFD_3.0/Images/CFD/AF-202/CFD-AF-202-122-N.jpg"
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/01-04/Images1/01_126_3963.jpg"
        file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/05_140_5449.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/06_111_2638.jpg"
        file3 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/09_110_2555.jpg"
        # file3 = FaceUtilsTest.root_dir + "/FaceWarehouse/FaceWarehouse_Data/Tester_124/TrainingPose/pose_11.png"
        # file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/10_110_2559.jpg"
        # file1 = FaceUtilsTest.root_dir + "/CFEE_Database_230/01-04/Images1/02_339_3470.jpg"
        # image1 = face_recognition.load_image_file(file1)
        up = -0.05
        scale = 1.9
        image1 = self.mark_detail(file1, up, scale)
        image2 = self.mark_detail(file2, up, scale)
        image3 = self.mark_detail(file3, up, scale)

        # image = crop(image, face_location)

        plt.rcParams['figure.figsize'] = [15, 6]
        f, axarr = plt.subplots(nrows=1, ncols=3)
        # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(image1.shape)
        print(image2.shape)
        print(image3.shape)
        axarr[0].imshow(image1)
        axarr[1].imshow(image2)
        axarr[2].imshow(image3)
        plt.show()

    def test_extract_face(self):
        file1 = FaceUtilsTest.root_dir + "/CFD_3.0/Images/CFD/AF-200/CFD-AF-200-228-N.jpg"
        # file1 = FaceUtilsTest.root_dir + "/CFD_3.0/Images/CFD/AF-202/CFD-AF-202-122-N.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFD_3.0/Images/CFD/AF-202/CFD-AF-202-122-N.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/01-04/Images1/02_339_3470.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/09-12/Images3/10_110_2559.jpg"
        # file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/05_140_5449.jpg"
        file2 = FaceUtilsTest.root_dir + "/CFEE_Database_230/05-08/Images2/06_111_2638.jpg"
        file3 = FaceUtilsTest.root_dir + "/FaceWarehouse/FaceWarehouse_Data/Tester_124/TrainingPose/pose_11.png"
        # image1 = face_recognition.load_image_file(file1)
        image1 = extract_face_fixed(face_recognition.load_image_file(file1))
        image2 = extract_face(face_recognition.load_image_file(file2))
        image3 = extract_face(face_recognition.load_image_file(file3))

        plt.rcParams['figure.figsize'] = [15, 6]
        f, axarr = plt.subplots(nrows=1, ncols=3)
        # axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(image1.shape)
        print(image2.shape)
        print(image3.shape)
        axarr[0].imshow(image1)
        axarr[1].imshow(image2)
        axarr[2].imshow(image3)
        plt.show()
