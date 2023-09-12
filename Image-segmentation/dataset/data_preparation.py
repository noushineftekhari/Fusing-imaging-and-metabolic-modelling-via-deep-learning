import numpy as np
import os
from coreutils import *
import SimpleITK as sitk
import csv

work_dir = r'D:\Unet\dataset'

raw_path = os.listdir(work_dir)

for element in raw_path:
    path = os.path.join(work_dir, element)
    if os.path.isdir(path):
        print(path)
        niffti_path = os.path.join(path, 'CT.nii.gz')
        ct = sitk.ReadImage(niffti_path)
        ct_np = sitk.GetArrayFromImage(ct)

        label_path = os.path.join(path, 'label2.nii.gz')

        label = sitk.ReadImage(label_path)
        label_np = sitk.GetArrayFromImage(label)

        new_label_np = np.zeros(label_np.shape)
        minimum = -200
        maximum = 300
        new_label_np[np.logical_and(ct_np < maximum, ct_np >= minimum)] = 1

        mask_np = label_np * new_label_np
        new_label = get_sitk_image(mask_np, ct)

        new_label_path = os.path.join(path, 'label3.nii.gz')
        sitk.WriteImage(new_label, new_label_path)

