import numpy as np
import os
from coreutils import *
import SimpleITK as sitk
import csv

header = ['path', 'min', 'max', 'med']
data = []

work_dir = r'D:\Unet\dataset'

raw_path = os.listdir(work_dir)

for element in raw_path:
    path = os.path.join(work_dir, element)
    if os.path.isdir(path):
        niffti_path = os.path.join(path, 'CT.nii.gz')
        ct = sitk.ReadImage(niffti_path)
        ct_np = sitk.GetArrayFromImage(ct)

        label_path = os.path.join(path, 'label.nii.gz')

        label = sitk.ReadImage(label_path)
        label_np = sitk.GetArrayFromImage(label)

        mask_np = ct_np * label_np
        maximum = np.max(mask_np)
        minimum = np.min(mask_np)
        median = np.median(mask_np[mask_np != 0])

        print(path)
        # print('maximum:    ' + str(maximum))
        # print('minimum:    ' + str(minimum))
        # print('median:     ' + str(median))
        data.append(list([path, maximum, minimum, median]))

path_csv = os.path.join(work_dir, 'ct_info1.csv')
with open(path_csv, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
