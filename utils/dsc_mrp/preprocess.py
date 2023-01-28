import os
import glob
import numpy as np
import csv
from numba import njit
from skimage import exposure
from skimage.filters import threshold_minimum
# from skimage.morphology import closing, opening, remove_small_holes
from skimage.morphology import erosion, dilation, closing, disk
from scipy import ndimage
import pathlib
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from skimage.util import montage
import sys
sys.path.append("/home/yejin/Projects/MRP_3D_MROD_Net")
from gen_dicoms_noHeader import mr_collateral_dsc_gen_dicoms
# from preprocess_data.generate_ordinal_regression_labels import spacing_decreasing_discretization
from utils.commons.phase_plot import get_phase_plot
from utils.commons.ROIs import get_roi
from utils.commons.ROIs_proposal import ROIsProposalDSC


def draw_hist(art, cap, even, lven, delay, mask):
    phases = [art, cap, even, lven, delay]
    phase_name = ['Art', 'Cap', 'Even', 'Lven', 'Del']
    for a, name in zip(phases, phase_name):
        real_phase_map = a[mask > 0]
        plt.hist(real_phase_map, 50)
        plt.savefig(f'/home/yejin/Desktop/data_fig/RAW_rs_intensity_{name}.png')
        plt.close()

def save_fig(img, gt, dir_in):

    # save img with gt contour
    fig, ax = plt.subplots(1, 1, num='Final_Input', figsize=(10, 10))
    fig.suptitle("Preprocessed Input with Phasemap contour")
    ax.imshow(montage(img[1, :, :, :], grid_shape=(5, 5)), cmap='gray')
    ax.contour(montage(gt[1, :, :, :], grid_shape=(5, 5)))
    plt.figure('Final_Input')
    plt.savefig(f'{dir_in}/Final_Input.png', dpi=150)
    plt.close('Final_Input')

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

def find_time_range(img, mask):
    sub_img = -img + img[0]
    roi_proposals = ROIsProposalDSC()
    roi_proposals.propose_ROIs(sub_img, mask, visualize=False)
    if roi_proposals.ROIs_best_loc == None:
        return 0, 40
    prefix = "dsc"
    roi = get_roi(prefix)
    roi_dict = {}
    for i_phase, current_phase in enumerate([f'_{prefix}_roi_art_', f'_{prefix}_roi_vein_']):
        sl_idx = roi_proposals.ROIs_best_loc[i_phase][0] + 1  # To match with the slice-idx on the GUI
        mask = np.zeros_like(img[0, 0]).astype(int)
        mask[tuple(roi_proposals.ROIs_best_loc[i_phase][1:])] = 1
        roi_dict[current_phase] = roi(mask, sl_idx, sub_img, current_phase)
    
    time_points_keys = [f'_{prefix}_sap_', f'_{prefix}_eap_', f'_{prefix}_svp_', f'_{prefix}_evp_'] 
    phase_plot = get_phase_plot(prefix)(time_points_keys)
    values = {}
    values['_dsc_auto_roi_phase_'] = True
    art_start, art_peak, even_peak, lven_end, even_end, time_points = phase_plot.compute_time_points(roi_dict['_dsc_roi_art_'], roi_dict['_dsc_roi_vein_'], values)
    delay_end = min(roi_dict['_dsc_roi_art_'].mean_values.shape[0], lven_end + (lven_end - even_peak))
    
    return int(art_start), int(delay_end)

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
    # make the number of time to 40 
    start_prop = 0.3
    start_time, end_time = find_time_range(imgs, masks)
    time_range = end_time - start_time + 1
    if time_range < 40:
        total_extra_time = 40 - time_range
        extra_start_time = int(total_extra_time  * start_prop)
        new_start_time = start_time - extra_start_time
        if new_start_time <= 0:
            imgs = imgs[:40]
        else:
            extra_end_time = total_extra_time - extra_start_time
            new_end_time = end_time + extra_end_time
            imgs = imgs[new_start_time:new_end_time+1]
    else:
        imgs = imgs[:40]
    
    # reverse slice order
    imgs = imgs[:,::-1,:,:]

    # preprocess
    n, c, w, h = imgs.shape
    outputs = []
    for i in range(n):
        nest_outputs = []
        for j in range(c):
            img = imgs[i][j]
            mask = masks[0][j]
            if np.all(img==0): #empty slice
                img = img.reshape(1, w, h)
                img = img * mask
                nest_outputs.append(img)
            else: 
                img = img.reshape(1, w, h)
                img = img * mask
                img = (img - img.min()) / (img.max() - img.min())
                nest_outputs.append(img)
        outputs.append(np.concatenate(nest_outputs).reshape(1, c, w, h))
    output_imgs = np.concatenate(outputs)

    return output_imgs

def preprocess_input_v4(imgs, masks):
    """
    1. rescale intensity in mask with percentile.
    2. rescale to 0 -> 1.
    3. matmul(input, mask).
    :param imgs: ndarray
    :param masks: ndarray
    :param percentile: int
    :return: output_imgs: ndarray
    """
    # make the number of time to 40 using ROI proposals
    start_prop = 0.3
    start_time, end_time = find_time_range(imgs, masks)
    time_range = end_time - start_time + 1
    if time_range < 40:
        total_extra_time = 40 - time_range
        extra_start_time = int(total_extra_time  * start_prop)
        new_start_time = start_time - extra_start_time
        if new_start_time <= 0:
            imgs = imgs[:40]
        else:
            extra_end_time = total_extra_time - extra_start_time
            new_end_time = end_time + extra_end_time
            imgs = imgs[new_start_time:new_end_time+1]
    else:
        imgs = imgs[:40]
    
    # # add or delete time series to be 40
    # if imgs.shape[0] < 40:
    #     time = 40 - imgs.shape[0]
    #     extra_time_series = np.tile(imgs[-1], (time, 1, 1, 1))
    #     imgs = np.concatenate((imgs, extra_time_series), axis=0)
    # else: # > 40
    #     imgs = imgs[:40]

    # reverse slice order
    imgs = imgs[:,::-1,:,:]

    # normalize the entire time-series volume - data version 4
    n, c, w, h = imgs.shape
    imgs = imgs * masks
    img_masks= np.tile(masks, (n,1,1,1))
    imgs_min, imgs_max = imgs[img_masks>0].min(), imgs[img_masks>0].max()
    output_imgs = (imgs - imgs_min) / (imgs_max - imgs_min)

    return output_imgs

def preprocess_gt(gt, masks, percentile=2):
    """
    slice-wise
    1. rescale intensity in mask with percentile.
    2. rescale to -0.9 -> 0.9.
    3. matmul(input, mask).
    :param imgs: ndarray
    :param masks: ndarray
    :param percentile: int
    :return: output_imgs: ndarray
    """
    # reverse slice order
    gt = gt[::-1,:,:]

    c, w, h = gt.shape
    outputs = []
    for slice in range(c):
        img = gt[slice]
        mask = masks[0][slice]
        if np.all(img==0): # empty slice
            img = img * mask
            outputs.append(img)
        else: 
            img = ndimage.median_filter(img, size=5)
            if percentile:
                p2, p98 = np.percentile(img[mask > 0], (percentile, 100 - percentile))
                img = exposure.rescale_intensity(img, in_range=(p2, p98))
            img = img * mask
            img = (img - img.min()) / (img.max() - img.min())
            img = 1.8 * img - 0.9
            outputs.append(img)
    output_gt = np.concatenate(outputs).reshape(1, c, w, h)
    
    return output_gt
    
def preprocess_gt_v4(gts, masks, rs_percentile):
    """
    phase-wise
    1. rescale intensity in mask with percentile.
    2. rescale to -0.9 -> 0.9.
    3. matmul(input, mask).
    :param imgs: ndarray
    :param masks: ndarray
    :param percentile: int
    :return: output_imgs: ndarray
    """
    c, w, h = gts.shape
    # reverse slice order
    print(f"{gts.min()}, {gts.max()}")
    gts = gts[::-1,:,:]
    # median filter & normalize -0.9 ~ 0.9
    # gts = gts / gts.max() #float64
    gts = ndimage.median_filter(gts, size=5)
    gt_masks = masks[0]
    if rs_percentile:
        p2, p98 = np.percentile(gts[gt_masks > 0], (rs_percentile, 100 - rs_percentile))
        gts = exposure.rescale_intensity(gts, in_range=(p2, p98))
    gts = gts * gt_masks
    gts_min, gts_max = gts[gt_masks>0].min(), gts[gt_masks>0].max()
    # output_gts = (gts - gts_min) / (gts_max - gts_min)
    # output_gts = 1.8 * output_gts - 0.9
    # output_gts = output_gts.reshape(1,c,w,h)
    output_gts = gts.reshape(1,c,w,h)
    return output_gts

def generate_weight_mask(phase_map_raw, mask, bins=50):
    phase_l = []
    phase_map = np.copy(phase_map_raw)
    for i in range(5):
        phase = phase_map[i]
        real_phase_map = phase[mask[0] > 0]
        temp_weight_map = np.histogram(real_phase_map, bins)
        temp_weight_map[0][temp_weight_map[0] == 0] = 1
        weight_map_tuple = (np.log(1 / (temp_weight_map[0] / real_phase_map.shape[0])), temp_weight_map[1])
        for idx in range(len(weight_map_tuple[0])):
            if idx < len(weight_map_tuple[0]) - 1:
                phase[(phase >= weight_map_tuple[1][idx]) & (phase < weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
            else:
                phase[(phase >= weight_map_tuple[1][idx]) & (phase <= weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
        d, w, h = phase.shape
        phase = phase.reshape(1, d, w, h)
        phase *= mask[0]
        phase_l.append(phase)
    return np.concatenate(phase_l)

def generate_mask(input_imgs, thr_percent=.75):
    """
    1. calculate avg intensity based on timeframe axis.
    2. rescale intensity.
    3. use threshold_minimum to generate mask.
    4. use closing morphology operation to improve mask.
    :param input_imgs: ndarray
    :param percentile: int
    :return: mask: ndarray
    """
    se = disk(6)
    tmp0 = np.squeeze(input_imgs.sum(axis=0))
    tmp00 = (tmp0- tmp0.min()) / (tmp0.max() - tmp0.min())
    thr = np.percentile(tmp00, thr_percent * 100)
    mask = np.zeros_like(tmp0)
    for j in range(tmp0.shape[0]):
        mask[j] = closing(dilation(erosion((tmp00[j] > thr), se), se), se)
    return mask[np.newaxis]

def take_file_name(dataset_file):
    outputs = []
    with open(dataset_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            outputs.append(row[0])
    return outputs

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

def preprocess_dsc(input_imgs, phasemaps, masks):

    art_phase, cap_phase, even_phase, lven_phase, del_phase = phasemaps[0], phasemaps[1], phasemaps[2], phasemaps[3], phasemaps[4]

    # slicewise preprocess Ground Truth
    mask = masks.copy()[:, ::-1, :, :] - np.zeros_like(masks) # prevent negative stride
    preprocessed_input = preprocess_input_v4(input_imgs, mask)

    # phasewise preprocess Ground Truth
    preprocessed_art_2 = preprocess_gt_v4(art_phase, mask, 2)
    preprocessed_cap_2 = preprocess_gt_v4(cap_phase, mask, 4)
    preprocessed_even_2 = preprocess_gt_v4(even_phase, mask, 8)
    preprocessed_lven_2 = preprocess_gt_v4(lven_phase, mask, 10)
    preprocessed_del_2 = preprocess_gt_v4(del_phase, mask, 10)
    phase_maps_2 = np.concatenate([preprocessed_art_2, preprocessed_cap_2, preprocessed_even_2,
                                    preprocessed_lven_2, preprocessed_del_2])
    wm_2 = generate_weight_mask(phase_maps_2, mask)
    # label_ordinal_2 = spacing_decreasing_discretization(14, phase_maps_2)

    # match size 
    preprocessed_input = resize_4d_image(preprocessed_input, 224)
    mask = resize_4d_image(mask, 224)
    phase_maps_2 = resize_4d_image(phase_maps_2, 224)

    # # save distribution
    # phases = ["Art", "Cap", "EVen", "LVen", "Del"]
    # px_data += 1
    # total_pixel = 0
    # fig, axes = plt.subplots(1, 5, figsize=(20,4),sharex=True, sharey=True)
    # for p in range(5):
    #     pm = phase_maps_2[p][mask[0]>0]
    #     pm = pm[pm<100]
    #     hist_info = axes[p].hist(pm, 100) # hist_info = plt.hist(pm, 50)
    #     axes[p].set_title(phases[p])
    #     hist, bin_list = hist_info[0], hist_info[1] 
    #     print(sum(hist))
    #     total_pixel += sum(hist)
    #     px_num_list[p] += hist
    #     bin_edges = bin_list
    # print(total_pixel)
    # plt.tight_layout()
    # plt.savefig(f"/home/yejin/Projects/MRP_3D_MROD_Net/All_px_dist.png")
    # plt.close()

    print("==================================")

    # check 
    print(preprocessed_input.shape)
    print(phase_maps_2.shape)
    print(mask.shape)
    # plt.imsave(os.path.join(file_name, "NpyFiles/Preprocessed_Input.png"), montage(preprocessed_input[10, :, :, :], grid_shape=(5,5)), cmap='gray')
    # plt.imsave(os.path.join(file_name, "NpyFiles/Preprocessed_GT.png"), montage(phase_maps[1, :, :, :], grid_shape=(5,5)), cmap='gray')
    # plt.imsave(os.path.join(file_name, "NpyFiles/Preprocessed_Ordinal_GT.png"), montage(label_ordinal[1, :, :, :], grid_shape=(5,5)), cmap='gray')
    # plt.imsave(os.path.join(file_name, "NpyFiles/Preprocessed_GT_2.png"), montage(phase_maps_2[1, :, :, :], grid_shape=(5,5)), cmap='gray')
    # plt.imsave(os.path.join(file_name, "NpyFiles/Preprocessed_Ordinal_GT_2.png"), montage(label_ordinal_2[1, :, :, :], grid_shape=(5,5)), cmap='gray')
    # plt.imsave(os.path.join(file_name, "NpyFiles/mask.png"), montage(mask[0, :, :, :], grid_shape=(5,5)), cmap='gray')
    # save_fig(preprocessed_input, phase_maps, save_fig_path)
    # color_gt = mr_collateral_dsc_gen_dicoms(save_fig_path, None, phase_maps, mask, suffix='', rescaling_first=False)
    # save_color_fig(color_gt, save_fig_path, rescale="color_gt_color")
    # color_ordinal_gt = mr_ordinal_dsc_gen_dicoms(save_fig_path, None, label_ordinal.astype('float'), mask, suffix='', rescaling_first=False)
    # save_color_fig(color_ordinal_gt, save_fig_path, rescale="ordinal_gt_color")
    # color_gt_2 = mr_collateral_dsc_gen_dicoms(save_fig_path, None, phase_maps_2, mask, suffix='', rescaling_first=False)
    # save_color_fig(color_gt_2, save_fig_path, rescale="color_gt_gray")
    # color_ordinal_gt_2 = mr_ordinal_dsc_gen_dicoms(save_fig_path, None, label_ordinal_2.astype('float'), mask, suffix='', rescaling_first=False)
    # save_color_fig(color_ordinal_gt_2, save_fig_path, rescale="ordinal_gt_gray")
    return preprocessed_input, phase_maps_2, mask

if __name__ == "__main__":
     preprocess_dsc("/home/yejin/Projects/MRP_3D_MROD_Net/dataset/ready_dsc_dataset.csv")