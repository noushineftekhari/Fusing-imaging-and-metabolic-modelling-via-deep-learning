import SimpleITK as sitk
import os


work_dir = r'D:\Second-year\Image-segmentation\dataset\removed_data\radiologist-no-ovary'
id = 'TCGA-13-1505_'

min_slice = 0
max_slice = 30

path_img = os.path.join(work_dir, id, 'CT.nii.gz')

img = sitk.ReadImage(path_img)

new_img = img[:, :, min_slice:max_slice]

new_path_img = os.path.join(work_dir, id, 'CT_cut.nii.gz')
sitk.WriteImage(new_img, new_path_img)


path_label = os.path.join(work_dir, id, 'label.nii.gz')
label = sitk.ReadImage(path_label)

new_label = label[:, :, min_slice:max_slice]
new_path_label = os.path.join(work_dir, id, 'label_cut.nii.gz')
sitk.WriteImage(new_label, new_path_label)