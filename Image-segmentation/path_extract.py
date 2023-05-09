import os
import SimpleITK as sitk
import numpy as np
import cv2


def find_center(img):
    M = cv2.moments(img.astype(np.float64))
    cy = int(M["m10"] / M["m00"])
    cx = int(M["m01"] / M["m00"])
    return cy, cx


def create_patch(img, label, pathch_size):
    patch = []

    label_sitk = sitk.GetImageFromArray(label)

    label_cc = sitk.ConnectedComponent(label_sitk, True)
    label_cc = sitk.RelabelComponent(label_cc, minimumObjectSize=10, sortByObjectSize=True)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(label_cc)
    num_component = int(stats.GetMaximum())

    for i in range(1, num_component + 1):
        label_cc_one = label_cc == i
        label_cc_one = label_cc_one > 0
        x, y = find_center(sitk.GetArrayFromImage(label_cc_one))
        crop_path = img[x - pathch_size:x + (pathch_size + 1), y - pathch_size:y + (pathch_size + 1)]
        patch.append(crop_path)

    return patch


work_path = r'D:\Second-year\Image-segmentation\dataset\TCGA-09-0364'
input_path = os.path.join(work_path, 'CT_cut.nii.gz')
label_path = os.path.join(work_path, 'label3_cut.nii.gz')
pathch_size = 9

img = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
label = sitk.GetArrayFromImage(sitk.ReadImage(input_path))

all_patches = []

for slide in range(len(label)):
    patch = create_patch(img[slide], label[slide], round(pathch_size/2))
    all_patches.append(patch)


a=1