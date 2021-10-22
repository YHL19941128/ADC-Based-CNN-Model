# -*- coding = utf-8 -*-

import pydicom # 用来解析dicom格式图像的像素值
import numpy as np
import cv2 # 用于保存图片
import os


def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):

    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype('uint8')

    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

count = 1
path = r'C:\Users\...'
filename = os.listdir(path)
print(filename)

for i in filename:
    document = os.path.join(path, i)
    outputpath = r'C:\Users\...'
    countname = str(count)
    countfullname = countname + '.jpeg'
    output_jpg_path = os.path.join(outputpath, countfullname)

    ds = pydicom.dcmread(document)
    img_array = ds.pixel_array
    # ds_array = sitk.ReadImage(document)
    # img_array = sitk.GetArrayFromImage(ds_array)
    # shape = img_array.shape  # name.shape
    # img_array = np.reshape(img_array, (shape[1], shape[2]))
    high = np.max(img_array)
    low = np.min(img_array)

    convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)
    count += 1

