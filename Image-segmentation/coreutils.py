def read_dicom(dcm_dir_path=None, niftipath=None):

    # from SimpleITK import ImageSeriesReader, WriteImage
    import SimpleITK as sitk

    print("Reading DICOM directory:", dcm_dir_path)

    # Initialising series reader
    reader = sitk.ImageSeriesReader()

    # Configure the reader to load all DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    # List all DICOM files
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir_path, recursive=True)

    # series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir_path)
    # dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir_path, series_IDs[0])

    reader.SetFileNames(dicom_names)

    # Read image
    image = reader.Execute()

    # Truncate intensities
    image = sitk.Threshold(image, -1024, 2500, outsideValue=-1024)

    # Print matrix size
    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])

    # Force RAI orientation
    # image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    print("Writing NIFTI file:", niftipath)
    sitk.WriteImage(image, niftipath)

    return reader

# ----------------------------------------------------------------------------------------------------------------------

def write_dicom(niftipath=None, dcm_dir_path=None, sitk_reader=None, series_number=0, Tenoke_description='Tenoke'):

    from SimpleITK import ReadImage, ImageFileWriter, CastImageFilter, sitkInt16
    import os
    from time import strftime
    from shutil import rmtree

    print("Reading NIFTI file:", niftipath)
    img = ReadImage(niftipath)

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate operation and requires knowledge of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #            http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    writer = ImageFileWriter()

    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are also
    # accessible).
    tags_to_copy = ["0010|0010",  # Patient Name
                    "0010|0020",  # Patient ID
                    "0010|0030",  # Patient Birth Date
                    "0020|000D",  # Study Instance UID, for machine consumption
                    "0020|0010",  # Study ID, for human consumption
                    "0008|0020",  # Study Date
                    "0008|0030",  # Study Time
                    "0008|0050",  # Accession Number
                    "0008|0060"   # Modality
                    ]

    modification_time = strftime("%H%M%S")
    modification_date = strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = img.GetDirection()
    series_tag_values = [(k, sitk_reader.GetMetaData(0, k)) for k in tags_to_copy if sitk_reader.HasMetaDataKey(0, k)] + \
                        [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),  # Series Instance UID
                         ("0020|0037",
                          '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
                         ("0008|103e", sitk_reader.GetMetaData(0, "0008|103e") + " (" + Tenoke_description + ")")]  # Series Description

    # Make output DICOM dir.
    print("Writing DICOM dir:", dcm_dir_path)
    if os.path.exists(dcm_dir_path):
        rmtree(dcm_dir_path)
    os.makedirs(dcm_dir_path)

    # Convert to int16 type, set metadata, and write out DICOM series.
    castFilter = CastImageFilter()
    castFilter.SetOutputPixelType(sitkInt16)
    img = castFilter.Execute(img)
    for i in range(img.GetDepth()):
        slice = img[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            slice.SetMetaData(tag, value)
        # Slice specific tags.
        slice.SetMetaData("0008|0012", strftime("%Y%m%d"))  # Instance Creation Date
        slice.SetMetaData("0008|0013", strftime("%H%M%S"))  # Instance Creation Time
        slice.SetMetaData("0020|0032",
                          '\\'.join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        slice.SetMetaData("0020|0013", str(i))  # Instance Number
        slice.SetMetaData("0020|0011", str(series_number))  # Series Number
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(dcm_dir_path, str(i) + '.dcm'))
        writer.Execute(slice)

# ----------------------------------------------------------------------------------------------------------------------

def get_dirname(path_string):
    """
    Extract folder name from full or relative path to file or folder.

    :param path_string: File/folder path.

    :return: Folder name.
    """

    import os

    dirname = os.path.dirname(path_string)

    return dirname

# ----------------------------------------------------------------------------------------------------------------------


def get_niftipath(dialogue_title='Select NIFTI file'):
    """
    Open dialogue to select a NIFTI file, and return full path.
    """

    import os
    import sys

    from tkinter import Tk, filedialog


    Tk().withdraw()

    filepath = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title=dialogue_title,
                                          filetypes=(("NIFTI files", "*.nii, *.nii.gz"), ("all files", "*.*")))

    if filepath == "":
        print(dialogue_title + '...')
        print('ERROR: File not selected.')
        sys.exit(0)
    else:
        return filepath


def get_dirpath(dialogue_title='Select folder'):
    """
    Open dialogue to select a folder, and return full path.
    """

    import os
    import sys
    from tkinter import Tk, filedialog


    Tk().withdraw()

    dirpath = filedialog.askdirectory(initialdir=os.getcwd(),
                                      title=dialogue_title,
                                      mustexist=True)

    if dirpath == "":
        print(dialogue_title + '...')
        print('ERROR: Folder not selected.')
        sys.exit(0)
    else:
        return dirpath

# ----------------------------------------------------------------------------------------------------------------------


def get_filename(path_string):
    """
    Extract file name from full or relative path string.

    :param path_string: File path.

    :return: File name.
    """

    import os

    filename = os.path.basename(path_string)

    return filename


# ----------------------------------------------------------------------------------------------------------------------

def get_fileroot(path_string):
    """
    Extract file name root from file path.

    e.g. 'testdata/nifti_01605836/3_oax_t1_flair.nii' -> '3_oax_t1_flair'

    Compressed (.EXT.gz) file extensions are permitted (n.b. .gz only).

    e.g. 'testdata/nifti_01605836/3_oax_t1_flair.nii.gz' -> '3_oax_t1_flair'

    :param path_string: File path.

    :return: File root.
    """

    import os

    filename = get_filename(path_string)

    fname_split = os.path.splitext(filename)

    last_extension = fname_split[1]

    if last_extension == '.gz':

        fname_split = os.path.splitext(fname_split[0])

    fileroot = fname_split[0]

    return fileroot


def get_dirname(path_string):
    """
    Extract folder name from full or relative path to file or folder.

    :param path_string: File/folder path.

    :return: Folder name.
    """

    import os

    dirname = os.path.dirname(path_string)

    return dirname


# ----------------------------------------------------------------------------------------------------------------------


def add_prefix(path_string, prefix):
    """
    Prepend a string to the file or directory name implicit in a path string.

    e.g. coreutils.add_prefix('testdata/01605836', 'nifti')
      -> 'testdata/nifti_01605836'

    e.g. coreutils.add_prefix('testdata/nifti_01605836/3_oax_t1_flair.nii.gz', 'biascorr')
      -> 'testdata/nifti_01605836/biascorr_3_oax_t1_flair.nii.gz'

    :param path_string: Path to file or directory.
    :param prefix: String to be prepended to file name.

    :return: Path to new file or directory.
    """

    import os

    path_split = os.path.split(path_string)

    if not path_split[1]:

        path_split = os.path.split(path_split[0])

    new_basename = prefix + '_' + path_split[1]

    new_path = os.path.join(path_split[0], new_basename)

    return new_path


# ----------------------------------------------------------------------------------------------------------------------

def move_all_files(srcDir, dstDir):

    import shutil, os, glob

    # Check if both are directories
    if os.path.isdir(srcDir) and os.path.isdir(dstDir):

        # Iterate over all the files in source directory
        for filePath in glob.glob(srcDir + '\*'):
            if not os.path.isdir(filePath):
                # Move each file to destination Directory
                shutil.move(filePath, dstDir)

    else:
        print("srcDir & dstDir should be folders")


def get_sitk_image(arr, sitk_img):
    import SimpleITK as sitk
    new_sitk_img = sitk.GetImageFromArray(arr)
    new_sitk_img.SetDirection(sitk_img.GetDirection())
    new_sitk_img.SetOrigin(sitk_img.GetOrigin())
    new_sitk_img.SetSpacing(sitk_img.GetSpacing())

    return new_sitk_img