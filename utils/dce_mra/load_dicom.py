import os
import gc
import glob
import time
import pickle
import pydicom
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['toolbar'] = 'None'
from pydicom import dcmread
from functools import partial
from skimage.transform import resize
from matplotlib.widgets import RectangleSelector
from concurrent.futures import ThreadPoolExecutor

from utils.commons.thread_pool import thread_pool
from utils.commons.auto_crop import auto_crop as _auto_crop
from utils.commons.misc import timer, NUM_WORKERS, D_TYPE
from utils.commons.skull_stripping import get_foreground, extract_brain


def interpolate(img, new_size):
    new_img = resize(img, new_size, order=1, anti_aliasing=False, clip=False, preserve_range=True)
    img = None
    return new_img


def is_valid(_filename):
    if not os.path.isdir(_filename) and (_filename.split('\\')[-1][:3] != 'PS_') and (
            _filename.split('\\')[-1][:3] != 'XX_'):
        return True
    else:
        return False


class CropImage:
    def __init__(self, view='Sagittal'):
        self.window = None
        self.fig, self.rs, self.event, self.pressed, self.fignum = (None,) * 5
        self.cropped = False
        self.ec, self.er = None, None
        self.view = view

    def toggle_selector(self, event):
        if event.dblclick:
            event.key = 'q'
            self.pressed = True
        if event.key in ['Q', 'q'] and self.rs.active:
            self.rs.set_active(False)
        if event.key in ['A', 'a'] and not self.rs.active:
            self.rs.set_active(True)
        self.event = event

    def get_selector(self, ax):
        self.rs = RectangleSelector(ax, self.onselect, drawtype='box', useblit=True, button=[1], minspanx=2,
                                    minspany=2,
                                    spancoords='pixels', interactive=True,
                                    rectprops=dict(facecolor='r', edgecolor='r', alpha=1, fill=False),
                                    state_modifier_keys=dict(move=' ', clear='escape', square='shift', center='ctrl'),
                                    marker_props=dict(marker='x', markersize=None, markeredgecolor='r'),
                                    )

    def onselect(self, eclick, erelease):
        """eclick and erelease are matplotlib events at press and release"""
        self.ec = int(eclick.xdata), int(eclick.ydata)
        self.er = int(erelease.xdata), int(erelease.ydata)


    def crop_image(self, _img):
        self.fignum = f'Crop Image in {self.view} View'
        fig_ratio = _img.shape[2] / _img.shape[1] if self.view == 'Sagittal' else _img.shape[3] / _img.shape[1]
        fig, ax = plt.subplots(1, 1, num=self.fignum, figsize=(2, 2 * fig_ratio))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        if self.view == 'Sagittal':
            ax.imshow(_img[0, :, :, int(_img.shape[-1] / 2)].T, cmap='gray')
        elif self.view == 'Axial':
            ax.imshow(_img[0, :, int(_img.shape[-2] / 2), :].T, cmap='gray')
        ax.axis('off')
        fig.canvas.mpl_connect('button_press_event', self.toggle_selector)

        self.get_selector(ax)
        self.pause()
        plt.close(fig)

        if (self.ec is not None) and (self.er is not None):
            if self.view == 'Sagittal':
                _img = _img[:, self.ec[0]:self.er[0], self.ec[1]:self.er[1], :]
                print('Cropped in sagittal view')
            elif self.view == 'Axial':
                _img = _img[:, self.ec[0]:self.er[0], :, self.ec[1]:self.er[1]]
                print('Cropped in axial view')
            self.cropped = True
        return _img


class Resize:
    def __init__(self, ratio_height, ratio_width):
        self.ratio = (ratio_height, ratio_width)
        self._new_size = None

    def new_size_(self, x, fix_size=(None, None)):
        new_h = int(np.ceil(x.shape[-2] * self.ratio[0])) if fix_size[0] is None else fix_size[0]
        new_w = int(np.ceil(x.shape[-1] * self.ratio[1])) if fix_size[1] is None else fix_size[1]
        self._new_size = new_h, new_w
        return new_h, new_w

    def resize_(self, x):
        return resize(x, self._new_size)


def get_series(dir_in):
    series_filenames_tmp = glob.glob('%s/*' % dir_in)

    series_filenames = [] + series_filenames_tmp
    for filename in series_filenames_tmp:
        if os.path.isdir(filename):
            series_filenames += glob.glob('%s/*' % filename)
    return [filename for filename in series_filenames if is_valid(filename)]


@timer
def load_dce(dir_in, to_crop=True, auto_crop=True, save_npy=False, window=None):
    print('\n\nLoading DICOM files from %s\nPlease wait...' % dir_in)
    tic0 = time.time()

    if save_npy:
        dir_npy = '%s/NpyFiles' % (dir_in)
        if not os.path.exists(dir_npy):
            os.makedirs(dir_npy)
    else:
        dir_npy = None

    source_dir = dir_in.split('/')[:-1] 
    source_dir = "/".join(source_dir) + '/DMRA_source'
    # source_dir = "/".join(source_dir) + '/TWIST-MRA'

    dir_size = len(os.walk(source_dir).__next__()[1])  # will be number_of_series for SIEMENS datasets
    series_filenames = get_series(source_dir)

    print('Loading...')
    datasets = []
    n = len(series_filenames)
    gc.collect()

    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, series_filenames)):
            datasets.append(result)

    if len(datasets) < 100:
        ms = 'The selected folder may not contain DICOM images. Please try choosing a more specific folder.'
        print(ms)
        return (None,) * 4
    # pydicom.dcmread has a different set of fields than one obtain by dicomheader in MATLAB

    
    if len(np.unique([ds.Rows for ds in datasets])) > 1:
        ms = 'The selected folder may contain more than one scanning sequence. Please try choosing a more specific folder that contains exactly one scanning sequence.'
        print(ms)
        return (None,) * 4
    
    number_of_files = len(datasets)

    loading_time = time.time() - tic0

    print('preprocessing...')
    tic1 = time.time()

    if 'siemens' in datasets[0].Manufacturer.lower():
        vendor = 'S'
        number_of_series = dir_size
        slice_thickness = float(datasets[0].SliceThickness)
        acq_num = [int(ds.AcquisitionNumber) for ds in datasets]
    else:
        vendor = 'G'
        acq_num = []
        for idx, ds in enumerate(datasets):
            acq_num.append(int(ds.NumberOfTemporalPositions))
        number_of_series = max(acq_num)  # int(datasets[0].NumberOfTemporalPositions)
        slice_thickness = min(float(datasets[0].SliceThickness), float(datasets[0].SpacingBetweenSlices))

    # for SIEMENS, multi phase images in single directory
    number_of_series = max(acq_num) if number_of_files == number_of_series else number_of_series

    number_of_slice = int(number_of_files / number_of_series)

    # Check the pixel_spacing to determine whether interpolation is needed
    pixel_width, pixel_height = float(datasets[0].PixelSpacing[0]), float(datasets[0].PixelSpacing[1])
    hdr = [None, ] * number_of_slice
    acq_time = np.zeros((number_of_series,))
    img = np.zeros((number_of_series, number_of_slice) + datasets[0].pixel_array.shape, dtype=D_TYPE)

    # Slices alignment
    for i in range(len(datasets)):
        ds = datasets[i]
        AcquisitionNumber = int(ds.AcquisitionNumber) if vendor == 'S' else int(ds.TemporalPositionIdentifier)
        InstanceNumber = int(ds.InstanceNumber)
        if InstanceNumber > number_of_slice:
            InstanceNumber = InstanceNumber - ((AcquisitionNumber - 1) * 2 * number_of_slice)
        AcquisitionTime = ds.AcquisitionTime if vendor == 'S' else ds.TriggerTime / 1000
        if isinstance(AcquisitionTime, str):
            if len(AcquisitionTime) == 0:
                AcquisitionTime = float(ds.SeriesTime) / 1000

        img[AcquisitionNumber - 1, InstanceNumber - 1] = ds.pixel_array.astype(D_TYPE)
        ds._pixel_array = None
        if AcquisitionNumber == 1:
            hdr[InstanceNumber - 1] = ds
        if InstanceNumber == 1:
            acq_time[AcquisitionNumber - 1] = AcquisitionTime
        datasets[i] = None
    ds = None

    # Crop image
    if to_crop:
        if auto_crop:
            print('Auto-cropping...')
            try:
                img = _auto_crop(img, dir_npy, pixel_height)
            except Exception as e:
                print(e)
                # auto_crop = False
                return
        if not auto_crop:
            plt.switch_backend('TkAgg')
            img = CropImage('Sagittal').crop_image(img)

            img = CropImage('Axial').crop_image(img)
            plt.switch_backend('Agg')

    # If interpolation is needed, set upscale to a bi-linear transformation else to a Null function
    print('Interpolation...')
    new_sizes = np.ceil([img.shape[1] * slice_thickness, img.shape[2] * pixel_width, img.shape[3] * pixel_height])
    new_sizes = list(new_sizes.astype(int))
    img = list(img)

    iip = partial(interpolate, new_size=new_sizes)
    with ThreadPoolExecutor(NUM_WORKERS) as executor:
        futures = executor.map(iip, img)

    img = np.asarray([result for result in futures])

    # # Mask generation
    print('Remove background...')
    mask = get_foreground(img)[np.newaxis]
    img *= mask

    # Pixel Value Scaling like Siemens
    img = (img / img.max()) * 2000. if vendor == 'G' else img
    preprocessing_time = time.time() - tic1

    # Record the new image shape (After processed) to the first instance of datasets
    hdr[0].new_height = img.shape[-2]
    hdr[0].new_width = img.shape[-1]

    # Change the tag so that RadiAnt DICOM Viewer can read the the whole volumde as an image series
    StudyInstanceUID = pydicom.uid.generate_uid()
    for slice_loop in range(number_of_slice):
        hdr[slice_loop].StudyInstanceUID = StudyInstanceUID

    # transpose for preprocessing
    img = img.transpose([2, 3, 0 ,1])

    message = 'Done!\nVendor: %s\nTotal time: %.2f s\n   Loading time: %.2f s\n   Preprocessing time: %.2f s' % (
        hdr[0].Manufacturer, time.time() - tic0, loading_time, preprocessing_time)
    print(message)
    return img, mask, hdr


if __name__ == "__main__":

    IMG, MASK, HDR, DIR_NPY = load_dce('/media/yejin/새 볼륨/mra/BrainMRA_Nov2021_Anonymized/CMC_DATA/Abnormal_No597/2016Y/20160701_KU_675687/DMRA_source', True, True, False)

    pass