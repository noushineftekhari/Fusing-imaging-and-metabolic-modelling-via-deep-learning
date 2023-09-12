import os
import sys
import nibabel as nib
import numpy as np
from coreutils import get_niftipath, get_dirpath, get_dirname
import SimpleITK as sitk
from tqdm import tqdm
import tensorflow as tf

# workdir = r'D:\Second-year\Image-segmentation\dataset'
workdir = r'D:\non-label'
list_dir = os.listdir(workdir)


programdir = get_dirname(os.path.abspath(sys.argv[0]))
model_number = 6
num_classes = 1
gpu_flag = True

# defaul gpu usage, no gpu
if not gpu_flag:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if gpu_flag:
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

model_path = os.path.join(programdir, 'saved_model', 'model{}').format(model_number)
saved_model = tf.saved_model.load(model_path)

for id in tqdm(list_dir):
    # id = list_dir[i]
    _workdir = os.path.join(workdir, id)

    input_path = os.path.join(_workdir, 'CT_cut.nii.gz')

    outroot = 'CNN_prob' + str(model_number)
    outpath = os.path.join(_workdir, 'CNN_prob' + str(model_number) + '.nii.gz')

    print(input_path)


    # default prediction name
    if not outroot:
        outroot = 'CNN_prob'
        outpath = os.path.join(_workdir, 'CNN_prob.nii.gz')

    # ----------------------------------------------------------------------------------------------------------------------
    class Output:
        """
        Main output class
        the model is loaded in the class and makes the prediction
        """
        def __init__(self, saved_model, num_classes, outroot, outpath):
            """
                   Class constructor

                   Parameters
                   ----------
                   model_path: str
                       path to the saved model to be load
                   outpath: str
                       path of where the output is gooin to be save
                   """
            self._model = saved_model
            self.output_name = outpath
            self.output_root = outroot
            self.num_classes = num_classes

        def __call__(self, path, input_path):
            """
            Class calling method

            Used to make the predictions of the model that is already loaded in the constructor

            Parameters
            ----------
            path: str
                main working path
            input_path: str
                t2 image absolute path

            Returns
            -------

            """
            # img, original_img, shape = self.open_nii_data(input_path)
            img, shape, header, affine = self.open_nii_data(input_path)

            # result = np.zeros(img.shape)
            result = np.zeros([img.shape[0], img.shape[1], img.shape[2], self.num_classes])

            for patch in tqdm(range(len(img))):
                result[patch] = self.predict(tf.expand_dims(tf.convert_to_tensor(img[patch], dtype=tf.float32),
                                                       0))['output'].numpy()

            result = tf.sigmoid(result).numpy()
            new_result = np.zeros([img.shape[2], img.shape[1], img.shape[0], self.num_classes])
            for i in range(self.num_classes):
                new_result[..., i] = result[..., i].T

            self.save(new_result, affine, header, os.path.join(path, self.output_name))

            # if num_classes == 9:
            #     new_result[..., 8] = new_result[..., 8]*0

            thresh = .3
            new_result[new_result > thresh] = 1

            mask = new_result[..., 0]
            for i in range(1, num_classes):
                    mask = np.squeeze(mask + new_result[..., i] * (i+1))

            # mask = np.squeeze(result[..., 0] + result[..., 1] * 2)
            # mask[mask > 2] = 2
            mask = np.cast["uint8"](mask)
            self.save(mask, affine, header, os.path.join(path, self.output_root + '_labelmap.nii.gz'))


            # result = sitk.GetImageFromArray(result)
            # result.SetOrigin(original_img.GetOrigin())
            # result.SetSpacing(original_img.GetSpacing())
            # result.SetDirection(original_img.GetDirection())
            # sitk.WriteImage(result, os.path.join(path, f'prob_{model_number}.nii.gz'))
            #
            # prob = ants.image_read(os.path.join(path, f'prob_{model_number}.nii.gz'))
            #
            # mask = ants.get_mask(prob, low_thresh=0.05, cleanup=0)
            #
            # ants.image_write(mask, os.path.join(path, f'mask_{model_number}.nii.gz'))

        def predict(self, x):
            """
            function forr the model prediction
            Parameters
            ----------
            x: tensorflow tensor
                feature to be predicted, has to be in the format [t1, t2, tof] concatenated in the last channel

            Returns
            -------
            tensorflow tensor
                the model prediction
            """
            return self._model.signatures["serving_default"](x)

        @staticmethod
        def save_image(array, original_image, path):
            image = sitk.GetImageFromArray(array)
            image.CopyInformation(original_image)
            sitk.WriteImage(image, path)

        @staticmethod
        def save(mask, affine, header, path):
            """
            Function for saving data in nifti format
            Parameters
            ----------
            mask: numpy array
                data to be save
            affine: iterable
                affine information of the image, use a 4x4 identity matrix in case of don't have one
            header: dict
                image header information
            path: str
                absolute path where to save the image

            Returns
            -------

            """
            _mask = nib.Nifti1Image(mask, affine, header)
            nib.save(_mask, path)

        @staticmethod
        def open_nii_data(path):

            def open_nii_gz(_path):
                image = nib.load(_path)
                data = image.get_fdata().T
                # maximum = np.percentile(data, 97.5)
                maximum = np.max(data)
                data = np.clip(data, -1000, maximum)
                data -= np.min(data)
                if np.max(data) != 0:
                    data = data / np.max(data)

                _shape = data.shape
                _header = image.header
                _affine = image.affine

                return data, _shape, _header, _affine

            img, shape, header, affine = open_nii_gz(path)

            return np.expand_dims(img, -1).astype('float32'), shape, header, affine

        @staticmethod
        def from_numpy(elements):
            data = tf.data.Dataset.from_tensor_slices((elements, elements))
            data = data.batch(1)
            return data
    # ----------------------------------------------------------------------------------------------------------------------

    # create output instance
    out = Output(saved_model, num_classes, outroot, outpath)

    # make de prediction
    out(_workdir, input_path)


