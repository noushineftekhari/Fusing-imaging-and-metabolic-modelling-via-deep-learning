import numpy
import os
from coreutils import *
import SimpleITK as sitk

# work_dir = r'D:\Unet\dataset'
work_dir = r'D:\non-label'

raw_path = os.listdir(work_dir)

for element in raw_path:
    path = os.path.join(work_dir, element)
    dicom_path = os.path.join(path, 'Dicom')
    niffti_path = os.path.join(path, 'CT.nii.gz')
    read_dicom(dicom_path, niffti_path)

    # nrrd_path = os.path.join(path, 'label.nrrd')
    # new_label_path = os.path.join(path, 'label.nii.gz')
    #
    # nrrd_img = sitk.ReadImage(nrrd_path)
    #
    # sitk.WriteImage(nrrd_img, new_label_path)
