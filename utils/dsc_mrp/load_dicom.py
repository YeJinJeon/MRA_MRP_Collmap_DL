import os
import gc
import pickle
import pydicom
import glob
import time
import numpy as np
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
from pydicom import dcmread
from utils.commons.preprocess import PreprocessDSC
from utils.commons.thread_pool import thread_pool
from utils.commons.misc import D_TYPE, NUM_WORKERS


class Resize:
    def __init__(self, ratio_height, ratio_width):
        self.ratio = (ratio_height, ratio_width)

    def new_size_(self, x):
        return int(np.ceil(x.shape[-2] * self.ratio[0])), int(np.ceil(x.shape[-2] * self.ratio[1]))

    def resize_(self, x):
        return resize(x, self.new_size_(x))

    def resize_dataset_(self, ds):
        return self.resize_(ds.pixel_array)


def get_instance_number(has_in_stack_pos_num, vendor, num_slices, num_files, num_series):
    """Get InstanceNumber from dataset instance using available data attributes"""

    def from_in_stack_pos_num(ds):
        return int(ds.InStackPositionNumber) - 1

    def from_instance_number(ds):
        return int(num_slices - (np.ceil((num_files - ds.InstanceNumber + 1.0) / num_series) - 1) - 1)

    def for_siemens(ds):
        return int(ds.InstanceNumber - ((ds.AcquisitionNumber - 1) * num_slices) - 1)

    if vendor == 'S':
        return for_siemens

    if has_in_stack_pos_num:
        return from_in_stack_pos_num

    return from_instance_number
    

def load_dsc(dir_in, *args, save_npy=False):
    """
    Load DSC-MRP in 'dir_in' folder
    :param save_npy:
    :param dir_in: the directory of the input image
    :param window:
    :return: preprocessed input, a brain mask and datasets (containing headers)
    """
    print('\n\nLoading DICOM files from %s\nPlease wait...' % dir_in)

    dir_npy = '%s/NpyFiles' % (dir_in)
    if not os.path.exists(dir_npy):
        os.makedirs(dir_npy)

    source_dir = dir_in.split('/')[:-1] 
    source_dir = "/".join(source_dir) + '/PWI_source'

    tic0 = time.time()
    series_filenames = glob.glob('%s/*.dcm' % source_dir)
    series_filenames = glob.glob('%s/*.IMA' % source_dir) if len(series_filenames) == 0 else series_filenames
    if len(series_filenames) == 0:
        ms = 'The specified folder does not contain any DICOM images. Please choose other folders!'
        print(ms)
        if save_npy:
            return (None,) * 4
        else:
            return (None,) * 3

    gc.collect()
    n = len(series_filenames)

    datasets = []
    with ThreadPoolExecutor(NUM_WORKERS) as executor:
        for (i, result) in enumerate(executor.map(dcmread, series_filenames)):
            datasets.append(result)

    # pydicom.dcmread has a different set of fields than one obtain by dicomheader in MATLAB
    tic1 = time.time()
    loading_time = tic1 - tic0
    number_of_files = len(datasets)

    if 'siemens' in datasets[0].Manufacturer.lower():
        vendor = 'S'
        acq_num = [int(ds.AcquisitionNumber) for ds in datasets]
    else:
        vendor = 'G'
        acq_num = [int(ds.NumberOfTemporalPositions) for ds in datasets]
    number_of_series = max(acq_num)

    number_of_slice = int(number_of_files / number_of_series)

    # Check the pixel_spacing to determine whether interpolation is needed
    pixel_width, pixel_height = datasets[0].PixelSpacing[0], datasets[0].PixelSpacing[1]
    # If interpolation is needed, set upscale to a bi-linear transformation else to a Null function
    if pixel_width > 1:
        upscale = Resize(pixel_height, pixel_width).resize_
        new_size = Resize(pixel_height, pixel_width).new_size_(datasets[0].pixel_array)
        img = np.zeros((number_of_series, number_of_slice) + new_size, dtype=D_TYPE)
    else:
        def foo(x):
            return x

        upscale = foo
        img = np.zeros((number_of_series, number_of_slice) + datasets[0].pixel_array.shape, dtype=D_TYPE)

    hdr = [None, ] * number_of_slice

    # Define a function to get InstanceNumber from dataset instance
    get_instance_number_fn = get_instance_number(hasattr(datasets[0], 'InStackPositionNumber'), vendor,
                                                 number_of_slice, number_of_files, number_of_series)

    def interpolate_s(idx):
        """Interpolation function for Siemens"""
        ds = datasets[idx]
        if vendor == 'S':
            InstanceNumber = get_instance_number_fn(ds)
            img[ds.AcquisitionNumber - 1, InstanceNumber] = upscale(ds.pixel_array.astype(D_TYPE))
            if ds.AcquisitionNumber == 1:
                hdr[InstanceNumber] = ds
        return None

    def interpolate(idx):
        """Interpolation function for non-Siemens data"""
        ds = datasets[idx]
        InstanceNumber = get_instance_number_fn(ds)
        img[ds.TemporalPositionIdentifier - 1, InstanceNumber] = upscale(ds.pixel_array.astype(D_TYPE))
        if ds.TemporalPositionIdentifier == 1:
            hdr[InstanceNumber] = ds
        return None

    # Define interpolation function depending on the imaging source
    interpolate_fn = interpolate_s if vendor == 'S' else interpolate

    # Perform image interpolation to scale the image spatial spacing to 1mm x 1mm
    with ThreadPoolExecutor(NUM_WORKERS) as executor:
        executor.map(interpolate_fn, range(len(datasets)))

    # Mask generation & input normalization
    img, mask = PreprocessDSC().preprocess_raw_input(img)
    preprocessing_time = time.time() - tic1
    gc.collect()

    # Record the new image shape (After processed) to the first instance of datasets
    hdr[0].new_height = img.shape[-2]
    hdr[0].new_width = img.shape[-1]

    # Change the tag so that RadiAnt DICOM Viewer can read the the whole volumde as an image series
    StudyInstanceUID = pydicom.uid.generate_uid()
    for slice_loop in range(number_of_slice):
        hdr[slice_loop].StudyInstanceUID = StudyInstanceUID

    @thread_pool
    def _save_npy():
        np.save('%s/IMG.npy' % dir_npy, img)
        np.save('%s/mask.npy' % dir_npy, mask)
        with open('%s/hdr' % dir_npy, 'wb') as fp:
            pickle.dump(hdr, fp)
        print('Done saving Numpy files')

    if save_npy:
        _save_npy()

    message = 'Done!\nVendor: %s\nTotal time: %.2f s\n   Loading time: %.2f s\n   Preprocessing time: %.2f s' % (
        datasets[0].Manufacturer, time.time() - tic0, loading_time, preprocessing_time)
    print(message)
    
    return img, mask, hdr