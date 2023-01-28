import os
import numpy as np
from skimage import exposure
from skimage.morphology import remove_small_holes
from skimage.transform import rotate
from scipy import ndimage
from scipy.ndimage import binary_erosion
from numba import njit
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from skimage.util import montage
from gen_dicoms_noHeader import mr_collateral_dce_gen_dicoms

def save_color_fig(gt, save_path, rescale=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ind, phase in enumerate(['Art', 'Cap', 'EVen', 'LVen', 'Del']):
        fig, ax = plt.subplots(1, 1, num='result', figsize=(10, 10))
        fig.suptitle(phase)
        if gt[ind].shape[1] > 25:
            ax.imshow(montage(gt[ind][:25, :, :], grid_shape=(5, 5), multichannel= True))
        else:
            ax.imshow(montage(gt[ind][:, :, :], grid_shape=(5, 5), multichannel= True))
        plt.figure('result')
        plt.savefig(f'{save_path}/{rescale}_{phase}.png', dpi=150)
        plt.close('result')

def center_crop(img, right_size):
    _, _ , h, w = img.shape
    start_x = w // 2 - right_size // 2
    start_y = h //2 - right_size // 2
    crop_img = img[:, :, start_y:start_y+right_size, start_x:start_x+right_size]
    return crop_img


def pad(img, min_size):
    # img.shape [ D, C ,H, W ]
    '''
    npad = (y축 위, y축 아래), (x축 위, x축 아래)
    '''
    pad_size_x = [(min_size - img.shape[3])//2, (min_size - img.shape[3])//2]
    pad_size_y = [(min_size - img.shape[2])//2, (min_size - img.shape[2])//2]
    if (sum(pad_size_y)+img.shape[2]) < min_size:
        pad_size_y[1] = pad_size_y[1]+1
    elif (sum(pad_size_x)+img.shape[3]) < min_size:
        pad_size_x[1] = pad_size_x[1]+1
    else:
        pass
    npad = (tuple(pad_size_y), tuple(pad_size_x))
    pad_img = np.zeros((img.shape[0], img.shape[1], min_size, min_size))
    for slice_loop in range(img.shape[1]):
        pad_img[0, slice_loop] = np.pad(img[0, slice_loop], npad, 'constant', constant_values=(0))
    return pad_img


def convert_mat_to_npy(mat_file):
    mat_img = np.load(mat_file)
    k = mat_file.split("/")[-1].split(".")[0]
    npy_img = mat_img.get(k)
    return npy_img


def transpose_and_normalized(input_imgs):
    output_imgs = np.copy(input_imgs)
    output_imgs = np.moveaxis(output_imgs, 3, 0)
    output_imgs = np.moveaxis(output_imgs, 3, 0)
    output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
    return output_imgs


def erode_mask(_mask: np.ndarray, pad_size: int, structure=None):
    """
    Pad and erode the 3D-mask using 3D operation without removing first and last slides and lower regions
    :param structure:
    :param _mask: 3D Numpy array
    :param pad_size: size of erosion (and padding as well)
    :return:
    """
    _mask = np.pad(_mask, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), constant_values=1)
    _mask = np.array([binary_erosion(_m, structure=structure, iterations=pad_size) for _m in _mask])
    _mask = _mask[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size].astype('uint8')
    return _mask


def generate_mask(input_imgs, percentile=2):
    """
    1. calculate avg intensity based on timeframe axis.
    2. rescale intensity.
    3. use threshold_minimum to generate mask.
    4. use closing morphology operation to improve mask.
    :param input_imgs: ndarray
    :param percentile: int
    :return: mask: ndarray
    """
    r_mask = input_imgs.sum(axis=0)
    r_mask = r_mask > 0.0  # (r_mask.max() * mask_thr)
    r_mask = np.array([remove_small_holes(_m, 1e3) for _m in r_mask], dtype=r_mask.dtype)
    r_mask = erode_mask(r_mask, 9)

    return r_mask[np.newaxis]


def match_size_x(img, correct_size):
    n, c, w, h = img.shape
    if w > correct_size:
        start_x = w // 2 -  correct_size // 2 
        resize_imgs = img[:, :, start_x:start_x + correct_size, :]
    else: # w <  correct_size:
        pad_size_x = [(correct_size - w)//2, (correct_size - w)//2]
        if (sum(pad_size_x) + w ) < correct_size:
            pad_size_x[1] = pad_size_x[1] + 1
        npad = ((0,0), (0,0), tuple(pad_size_x), (0, 0))
        resize_imgs = np.pad(img, npad, 'constant', constant_values=(0))
    return resize_imgs


def match_size_y(img, correct_size):
    n, c, w, h = img.shape
    if h > correct_size:
        start_y = h // 2 -  correct_size // 2 
        resize_imgs = img[:, :, :, start_y:start_y + correct_size]
    else: # h <  correct_size:
        pad_size_y = [(correct_size - h)//2, (correct_size - h)//2]
        if (sum(pad_size_y) + h ) < correct_size:
            pad_size_y[1] = pad_size_y[1] + 1
        npad = ((0,0), (0,0), (0,0), tuple(pad_size_y))
        resize_imgs = np.pad(img, npad, 'constant', constant_values=(0))
    return resize_imgs

def resize_4d_image(img, right_size):
    # resize to (224, 224)
    n, c, w, h = img.shape
    if w != right_size:
        img = match_size_x(img, right_size)
    if h != right_size:
        img = match_size_y(img, right_size)
    return img


def preprocess_input(imgs, masks, percentile=2):
    """
    1. rescale intensity in mask with percentile.
    2. rescale to 0 -> 1.
    3. matmul(input, mask).
    :param imgs: ndarray
    :param masks: ndarray
    :param percentile: int
    :return: output_imgs: ndarray
    """
    # normalize the entire time-series volume
    n, c, w, h = imgs.shape
    imgs = imgs * masks
    img_masks= np.tile(masks, (n,1,1,1))
    imgs_min, imgs_max = imgs[img_masks>0].min(), imgs[img_masks>0].max()
    output_imgs = (imgs - imgs_min) / (imgs_max - imgs_min)

    # add or delete time series to be 40
    if output_imgs.shape[0] < 40:
        time = 40 - output_imgs.shape[0]
        extra_time_series = np.tile(output_imgs[-1], (time, 1, 1, 1))
        output_imgs = np.concatenate((output_imgs, extra_time_series), axis=0)
    else: # > 40
        output_imgs = output_imgs[:40]
    return output_imgs


@njit(nogil=True)
def stretch_lim(img, tol_low, tol_high):
    """
    Mimic the stretchlim function in MATLAB
    :param tol_high:
    :param tol_low:
    :param img:
    :return:
    """
    nbins = 65536
    N = np.histogram(img, nbins, [0, img.max()])[0]
    cdf = np.cumsum(N) / np.sum(N)  # cumulative distribution function
    ilow = np.where(cdf > tol_low)[0][0]
    ihigh = np.where(cdf >= tol_high)[0][0]
    if ilow == ihigh:  # this could happen if img is flat
        ilowhigh = np.array([1, nbins])
    else:
        ilowhigh = np.array([ilow, ihigh])
    lowhigh = ilowhigh / (nbins - 1)  # convert to range [0 1]
    return lowhigh


def preprocess_gray_gt_2(gts, masks, percentile=2):
    """
    phase-wise
    1. rescale intensity in mask with percentile.
    2. rescale to -0.9 -> 0.9.
    """
    n, c, w, h = gts.shape
    # # delete slices to match with DSC-MRP
    # gts = np.delete(gts[:,:25,:,:], [0,1,7,17,22], axis=1)

    gts = gts / gts.max() #float64
    gts = ndimage.median_filter(gts, size=5)
    if percentile:
        p2, p98 = np.percentile(gts[masks > 0], (percentile, 100 - percentile))
        gts = exposure.rescale_intensity(gts, in_range=(p2, p98))
    gts = gts * masks
    gts_masks = np.tile(masks, (n,1,1,1))
    gts_min, gts_max = gts[masks>0].min(), gts[masks>0].max()
    output_gts = (gts - gts_min) / (gts_max - gts_min)
    output_gts = 1.8 * output_gts - 0.9
    output_gts = output_gts.reshape(1,c,w,h)
    return output_gts


def dce_reformat(r_input_dce, r_thickness, r_distance, r_slices, r_skip_distance, r_image_rotation, number_of_slice):
    input_dce = np.copy(r_input_dce)
    input_shape = input_dce.shape
    height = input_shape[0]
    width = input_shape[1]
    slice = input_shape[3]
    temp_output = np.zeros((height, width, 1, slice))
    for i in range(width):
        temp_output[:, i, 0, :] = rotate(np.squeeze(input_dce[:, i, 0, :]), r_image_rotation) #######################
    m_output = np.zeros((number_of_slice, width, 1, r_slices))
    avg_m_output = np.zeros((number_of_slice, width, 1, r_slices))
    for slice_loop in range(r_slices):
        slice_center = (r_skip_distance + 1) + slice_loop * r_distance
        slice_merge_start = int(slice_center - np.floor(r_thickness / 2))
        slice_merge_end = int(slice_center + (r_thickness - np.floor(r_thickness / 2)) - 1)
        if slice_merge_end > temp_output.shape[0]:
            slice_merge_end = temp_output.shape[0]-1
        for slice_merge_loop in range(slice_merge_start, slice_merge_end):
            m_output[:, :, 0, slice_loop] = m_output[:, :, 0, slice_loop] + \
                                            np.transpose(np.squeeze(temp_output[slice_merge_loop, :, 0, :]))
        avg_m_output[:, :, 0, slice_loop] = m_output[:, :, 0, slice_loop] / r_thickness
    return avg_m_output


def preprocess_dce(input_imgs, phases, properties):

    art_phase, cap_phase, even_phase, lven_phase, del_phase = phases[0], phases[1], phases[2], phases[3], phases[4]
    r_thickness, r_distance, r_slices, r_skip_distance, r_image_rotation, number_of_slice = properties

    # reformat input image
    input_imgs_l = []
    for i in range(input_imgs.shape[2]):
        temp_img = input_imgs[:, :, i, :].reshape(input_imgs.shape[0], input_imgs.shape[1], 1, input_imgs.shape[3])
        reformated_input_img = dce_reformat(temp_img,
                                            r_thickness, r_distance, r_slices, r_skip_distance,
                                            r_image_rotation, number_of_slice)
        input_imgs_l.append(reformated_input_img)
    
    reformated_input_imgs = transpose_and_normalized(np.concatenate(input_imgs_l, axis=2))
    mask = generate_mask(reformated_input_imgs)

    #preprocess input
    preprocessed_input = preprocess_input(reformated_input_imgs, mask)

    #preprocess gray target
    preprocessed_art_gray = preprocess_gray_gt_2(art_phase, mask, 2)
    preprocessed_cap_gray= preprocess_gray_gt_2(cap_phase, mask, 4)
    preprocessed_even_gray = preprocess_gray_gt_2(even_phase, mask, 8)
    preprocessed_lven_gray = preprocess_gray_gt_2(lven_phase, mask, 10)
    preprocessed_del_gray = preprocess_gray_gt_2(del_phase, mask, 10)
    phase_maps_gray = np.concatenate([preprocessed_art_gray, preprocessed_cap_gray, preprocessed_even_gray,
                                    preprocessed_lven_gray, preprocessed_del_gray])

    # match size 
    preprocessed_input = resize_4d_image(preprocessed_input, 224)
    mask = resize_4d_image(mask, 224)
    phase_maps_gray = resize_4d_image(phase_maps_gray, 224)
    # color_gt_gray = mr_collateral_dce_gen_dicoms('/data1/yejin/compu/mra_v2/mrod/result', None, phase_maps_gray, mask, suffix='dl', rescaling_first=True, insert_bar=True, from_deep_learning=False)
    # save_color_fig(color_gt_gray, os.path.join('/data1/yejin/compu/mra_v2/mrod/result', "NpyFiles"), rescale="color_gt_gray")

    return preprocessed_input, phase_maps_gray, mask