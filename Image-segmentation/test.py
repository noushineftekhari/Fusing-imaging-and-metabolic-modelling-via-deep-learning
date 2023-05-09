import os
from tqdm import tqdm

# loading_path = r'/scratch2/jac/Documents/201121_Carotid_hMRI_revisited/CAREII/'
# filter_file = r'/scratch2/jac/Documents/201121_Carotid_hMRI_revisited/CAREII/list_not_ok_selection2.txt'

loading_path = r'/scratch1/ssd_workdir/new_annotation/data/dt1/'
# filter_file = r'/scratch1/ssd_workdir/new_annotation/data/dt1/1.txt'
filter_file = r'/scratch1/ssd_workdir/new_annotation/data/dt1/list_not_ok_selection2.txt'

# loading_path = r'/scratch1/ssd_workdir/new_annotation/data/test/Huashan_Hospital/'
# loading_path = r'/scratch1/ssd_workdir/new_annotation/data/test/ZJHospital-Carotid-1/'
# loading_path = r'/scratch1/ssd_workdir/new_annotation/data/test/Zhongshan-segmented_01/'
# filter_file = r'/scratch1/ssd_workdir/new_annotation/data/test/Huashan_Hospital/list_not_ok_selection2.txt'

list_ok = []


if filter_file:
    with open(filter_file, 'r') as file:
        for line in file:
            list_ok.append(line.strip())
    # already_in_data_list.extend(list_ok)

# os.chdir(loading_path)
raw_path = os.listdir(loading_path)

for element in tqdm(raw_path):
# for element in raw_path:
    path = os.path.join(loading_path, element)
    centroids_csv_path = os.path.join(path, 'centroids_avg.csv')
    # if element in list_ok and os.path.isdir(path):
    if element == '175':
    # if True:
    # if os.path.isdir(path):
        print(element)

        # harmonize
        os.system('python Tenoke_harmonise_dimension.py {} {}'.format(path, 'T1.nii.gz'))
        os.system('python Tenoke_harmonise_dimension.py {} {}'.format(path, 'T2.nii.gz'))
        os.system('python Tenoke_harmonise_dimension.py {} {}'.format(path, 'TOF.nii.gz'))
        os.system('python Tenoke_harmonise_dimension.py {} {}'.format(path, 'tc2_L_ROIs.nii.gz'))
        os.system('python Tenoke_harmonise_dimension.py {} {}'.format(path, 'tc2_R_ROIs.nii.gz'))

        # cetroid
        os.system('python ovarian_unet_predict.py {} {} {} {} {} {}'.format(
            path, 'centroids', 9, 'T1_harmonised.nii.gz', 'T2_harmonised.nii.gz', 'TOF_harmonised.nii.gz'))

        os.system('python Tenoke_crop_restacking.py {} {} {} {} {} {} {}'.format(
            path, centroids_csv_path, 128, 3, 'T1_harmonised.nii.gz', 'T2_harmonised.nii.gz', 'TOF_harmonised.nii.gz'))
            # 'tc2_L_ROIs_harmonised.nii.gz', 'tc2_R_ROIs_harmonised.nii.gz'))

        # os.system('python Tenoke_copy_info.py {} {} {}'.format(
        #     path, 'T1_harmonised_L_crop128.nii.gz', 'TOF_harmonised_L_crop128.nii.gz'))
        # os.system('python Tenoke_copy_info.py {} {} {}'.format(
        #     path, 'T1_harmonised_R_crop128.nii.gz', 'TOF_harmonised_R_crop128.nii.gz'))
        #



