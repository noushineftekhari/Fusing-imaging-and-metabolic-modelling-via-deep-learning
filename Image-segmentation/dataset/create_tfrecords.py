import os
from threading import Thread
import nibabel as nib
import numpy as np
from data.tfrecords import convert_tfrecords

item = 0


def open_nii_gz(_path):
    data = nib.load(_path).get_fdata()
    # data[data < 0] = 0
    maximum = np.percentile(data, 97.5)
    # minimum = np.percentile(data, 2.5)
    data = np.clip(data, -1000, maximum)
    data -= np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    return data


def open_label(_path, name):
    data = nib.load(os.path.join(_path, name)).get_fdata()
    data[data < 0.5] = 0
    data[data > 0.5] = 1
    return data


def create(_path, _file):
    global item
    ct = open_nii_gz(os.path.join(_path, 'CT_cut.nii.gz'))

    label = open_label(_path, 'label3_cut.nii.gz')

    for index in range(ct.shape[-1]):
        # max_label = np.max(label[..., index])
        # if max_label > 0:
        convert_tfrecords(ct[..., index], label[..., index], os.path.join(saving_path, _file + '_' + str(item)))
        item += 1


if __name__ == '__main__':

    loading_path = r'D:\Unet\dataset'
    saving_path = r'D:\Unet\tfrecords\3\\'

    ratio = .9

    # filter file path, default False or None
    filter_file = False

    list_ok = []
    list_not_ok = []

    already_in_data_list = []
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    if not os.path.exists(os.path.join(saving_path, 'train')):
        os.mkdir(os.path.join(saving_path, 'train'))
    else:
        already_in_data_list.extend(os.listdir(os.path.join(saving_path, 'train')))

    if not os.path.exists(os.path.join(saving_path, 'eval')):
        os.mkdir(os.path.join(saving_path, 'eval'))
    else:
        already_in_data_list.extend(os.listdir(os.path.join(saving_path, 'eval')))

    os.chdir(loading_path)
    raw_path = os.listdir(loading_path)

    if filter_file:
        with open(filter_file, 'r') as file:
            for line in file:
                list_ok.append(line.strip())
    already_in_data_list.extend(list_ok)

    threads = []
    file = 0

    # for element in tqdm(raw_path):
    for element in raw_path:
        print(element)
        if element not in already_in_data_list:
            type_of_data = np.random.random(1)

            if type_of_data <= ratio:
                thread = Thread(target=create, args=(os.path.join(loading_path, element),
                                                     os.path.join('train', element)))
            else:
                thread = Thread(target=create, args=(os.path.join(loading_path, element),
                                                     os.path.join('eval', element)))
            thread.start()
            threads.append(thread)
            file += 1
            if file % 18 == 0:
                for worker in threads:
                    worker.join()
    for worker in threads:
        worker.join()
