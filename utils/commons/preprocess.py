import numpy as np
from skimage.morphology import erosion, dilation, closing, disk
from utils.commons.normalization import norm_ab
from utils.commons.misc import D_TYPE


def norm_01(x):
    """Normalize input to 0-1 range"""
    return (x - x.min()) / (x.max() - x.min())


class Preprocess:
    """Data Preprocessing functions for DSC-MRP"""

    def __init__(self, thr_percent=.75):
        files = {'IMG', 'ColArt', 'ColCap', 'ColDel', 'ColEVen', 'ColLVen'}
        self.find_coronal_idx_only = False
        self.min_size = 224
        # Raw input preproc options
        self.se = disk(6)
        self.n_slices_uthr = 50
        self.crop_size = 224
        self.c = -1
        self.norm_type = 'norm01'
        self.values_range = [0, 1]
        # self.thr = 5e-2
        self.thr_percent = thr_percent
        self.to_mask_input = True
        self.exclude_outliers = True
        # Raw ground truth preproc options
        self.file_idx = np.arange(1, 5)
        self.show_smooth_fig = False
        self.kernel_size = 5
        self.outlier = [np.NaN, 2, 4, 8, 10, 10]
        self.ColSF = 1
        self.start_slice = 7
        self.end_slice = 12
        self.rescaling_first = True
        self.test_code = False
        self.use_medfilt = 1
        self.norm_range = [-.9, .9]
        self.show_color_fig = 1
        self.visibility = 'off'

    def preprocess_raw_input(self, im):
        """
        Create the brain mask and normalize inputs to range 0-1
        :param im: a raw image
        :return: preprocessed input and a brain mask
        """
        padsize = np.ceil([0, 0, (self.min_size - im.shape[-2]) / 2, (self.min_size - im.shape[-1]) / 2]).astype(int)
        for k in range(len(padsize)):
            padsize[k] = max(padsize[k], 0)  # in case that padsize is negative
        padsize = [(pz, pz) for pz in padsize]
        im = np.pad(im[:60], padsize, 'constant', constant_values=0) \
            if np.sum(padsize) > 0 else im[:60]  # Pick only the first 60 frames
        mask = self.create_mask(im)
        im = im * mask if self.to_mask_input else im
        mask_4d = np.tile(mask, (im.shape[0], 1, 1, 1,))

        # Normalization to 0 mean (discarding outliers in the computation & assign to lower or upper thresholds)
        x_norm = norm_ab(im, self.values_range[0], self.values_range[1], None, self.exclude_outliers, mask_4d,
                         self.to_mask_input)
        # self.show_im_with_mask(x_norm[0], mask_4d[0])
        return x_norm, mask

    def create_mask(self, im):
        """
        Create a brain mask from input image
        :param im: input DSC-MRP
        :return: a brain mask
        """
        tmp0 = np.squeeze(im.sum(axis=0))
        tmp00 = norm_01(tmp0)
        thr = np.percentile(tmp00, self.thr_percent * 100)
        mask = np.zeros_like(tmp0)
        for j in range(tmp0.shape[0]):
            mask[j] = closing(dilation(erosion((tmp00[j] > thr), self.se), self.se), self.se)
        return mask[np.newaxis]

    @staticmethod
    def show_im_with_mask(im, mask):
        """
        Show the input image with mask contour
        :param im: input DSC-MRP
        :param mask: a binary brain mask
        """
        from skimage.util import montage
        import pylab as plt
        plt.imshow(montage(im), cmap='gray')
        plt.contour(montage(mask), cmap='gray')
        plt.savefig('example.png')


class PreprocessDSC(Preprocess):
    """Data Preprocessing functions for DSC-MRP"""
    def __init__(self, thr_percent=.75):
        super().__init__(thr_percent=thr_percent)


class PreprocessDCE(Preprocess):
    """Data Preprocessing functions for DCE-MRA"""
    def __init__(self, thr_percent=.3):
        super().__init__(thr_percent=thr_percent)

    def preprocess_raw_input(self, im):
        """
        Create the brain mask and normalize inputs to range 0-1
        :param im: a raw image
        :return: preprocessed input and a brain mask
        """
        padsize = np.ceil([0, 0, (self.min_size - im.shape[-2]) / 2, (self.min_size - im.shape[-1]) / 2]).astype(int)
        for k in range(len(padsize)):
            padsize[k] = max(padsize[k], 0)  # in case that padsize is negative
        mask = self.create_mask(im).astype(D_TYPE)
        im = im * mask if self.to_mask_input else im

        return im, mask
