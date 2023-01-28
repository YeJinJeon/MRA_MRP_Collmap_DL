import math

import numpy as np
import pylab as plt

from utils.dce_mra.correct_head_angle import estimate_rotation_angle
from utils.commons.misc import savitzky_golay
from skimage.morphology import erosion, closing, disk
from scipy.signal import convolve2d


def est_slope(x, y):
    """https://math.stackexchange.com/questions/2565098/vector-notation-for-the-slope-of-a-line-using-simple-regression/2642719#2642719"""

    X = x - x.mean()
    Y = y - y.mean()

    slope = (X.dot(Y)) / (X.dot(X))
    return slope


def estimate_noise(image: np.ndarray):
    """

    :param image: a 2D Numpy array
    :return:
    """
    H, W = image.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(image, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

    return sigma


def estimate_thr(MIP):
    hist = plt.histogram(MIP.flat, 100)
    n_pixels = np.prod(MIP.shape)
    thr = None
    for i, p in enumerate(hist[0]):
        if (p / n_pixels) < 1e-2:
            thr = hist[1][i]
            break
    return thr if thr else 300


def save_fig(mip: np.ndarray, idx: tuple, dir_npy: str):
    """
    Save a figure showing the head cropping area
    :param mip: maximum intensity projection array in coronal view
    :param idx: index of row and column of the cropping area
    :param dir_npy: a string indicating the saving folder
    :return:
    """
    idx_row, idx_col = idx
    figsize = 5
    hw_ratio = mip.shape[-2] / mip.shape[-1]
    fig, ax = plt.subplots(1, 1, num='auto_crop', figsize=(figsize, figsize * hw_ratio))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.imshow(mip, cmap='gray')
    box = np.zeros_like(mip)
    box[idx_row[0]:idx_row[-1] + 1, idx_col[0]:idx_col[-1] + 1] = 1
    ax.contour(box)
    ax.axis('off')

    plt.figure('auto_crop')
    plt.savefig(f'{dir_npy}/../auto_crop.png', dpi=200)
    plt.close('auto_crop')


def binarize_mip(mip, thr: float):
    """
    Generate a binary mask from a 2D maximum intensity projection image
    :param mip: 2D maximum intensity projection array
    :param thr: threshold for separating foreground and background
    :return:
    """
    if thr == -1.:
        noise_level = estimate_noise(mip)
        # thr = estimate_thr(mip) if noise_level > 10 else 50
        thr = np.percentile(mip, 55) if noise_level > 10 else np.percentile(mip, 30)
    print(thr)
    mask = mip > thr
    se = disk(6)
    mask = closing(erosion(mask, se), se)
    # mask = binary_closing(binary_opening(mip > thr, iterations=10).astype('uint8'))
    return mask


def _detect_head(mask, border: int = 25, crop_range: tuple = (160, 170)):
    """

    :param mask: 2D binary Numpy array
    :param border: Number of pixels extended on each side (left & right)
    :param crop_range: (min, max) the cropping height (in pixels)
    :return:
    """
    num_pix = mask.sum(axis=1)
    count_trials = 0
    num_pix_smooth = np.NaN
    while np.any(np.isnan(num_pix_smooth)):
        num_pix_smooth = savitzky_golay(num_pix, window_size=25)
        count_trials += 1
        if count_trials > 10:
            break
    X = np.arange(len(num_pix))
    sub_length = 20
    slopes = []
    for i in range(len(X) - sub_length):
        slopes.append(est_slope(X[i:i + sub_length], num_pix_smooth[i:i + sub_length]))
    slopes = np.array(slopes)

    # Remove the first half of the slopes series, since the first peak might be the highest slopes, which we dont want
    slopes[:int(len(slopes) * (2 / 5))] = slopes.mean()
    idx_row = np.where(num_pix[:np.argmin(slopes[:np.argmax(slopes)])])[0]

    # Extend the crop by extra few pixels
    idx_row = list(idx_row) + list(range(idx_row.max(), idx_row.max() + 10))
    idx_row = np.array(idx_row)
    d = num_pix_smooth[idx_row].max()  # diameter
    idx_row = idx_row[idx_row <= idx_row[0] + d]
    idx_col = np.zeros(mask.shape[1])
    idx_col[int(max(0, mask.shape[1] / 2 - d / 2 - border)): min(mask.shape[1], int(mask.shape[1] / 2 + d / 2 + border))] = 1
    idx_col = np.where(idx_col)[0]

    # Adjust idx_row in case the crop length is too short or too long
    crop_length = idx_row[-1] - idx_row[0]
    crop_length = min(max(crop_length, crop_range[0]), crop_range[1])
    idx_row = np.arange(idx_row[0], idx_row[0] + crop_length)

    return idx_row, idx_col


def detect_head(mip: np.ndarray, thr: float, dir_npy: str, crop_range: tuple = (160, 170)):
    """
    Detect the head section from a maximum intensity projection image
    :param mip: maximum intensity projection array
    :param thr: threshold for removing the image background
    :param dir_npy: a string indicating the saving folder for generated figures
    :param crop_range: (min, max) the cropping height (in pixels)
    :return:
    """

    # Generating foreground mask
    mask = binarize_mip(mip, thr)

    # Detect head section
    idx_row, idx_col = _detect_head(mask, crop_range=crop_range)

    # Save figure showing detected head section
    if dir_npy is not None:
        save_fig(mip, (idx_row, idx_col), dir_npy)

    return idx_row, idx_col


def auto_crop(img, dir_npy: str = None, pixel_height: float = 1., crop_range: tuple = (170, 180)):
    """
    Detect the head section and return the 4D image containing only the head section
    :param img: 4D (or 3D) Numpy array
    :param dir_npy: a string indicating the saving folder for generated figures
    :param pixel_height: height of a pixel (in cm)
    :param crop_range: (min, max) of the cropping height (in mm) when pixel height is 1mm
    :return: 4D Numpy array after cropping the head section
    """
    # Estimate the cropping height in term of number of voxels
    crop_range = tuple([int(cr/pixel_height) for cr in crop_range])

    # Extract the maximum intensity projection image
    time_idx = np.argmax(img[:, :, :int(img.shape[2] / 2)].mean(axis=(1, 2, 3)))  # Only consider the top section
    mip = img[time_idx].max(axis=0)

    # Detect head section from the maximum intensity projection image
    idx_row, idx_col = detect_head(mip, -1, dir_npy, crop_range)
    idx_row[0], idx_row[-1] = max(0, idx_row[0] - 5), idx_row[-1] - 5

    # Crop head section from the original image
    img = img[:, :, idx_row[0]:idx_row[-1] + 1, idx_col[0]:idx_col[-1] + 1]

    try:
        for view in ['axial', 'coronal']:
            estimate_rotation_angle(img[time_idx], view, dir_npy)
    except:
        pass

    return img


if __name__ == "__main__":
    _img = np.load('../../img.npy')
    _img = auto_crop(_img)
    print(_img.shape)