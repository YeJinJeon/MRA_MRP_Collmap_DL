import os
import time
import numpy as np
from functools import partial

from utils.commons.misc import my_rotate



def mr_collateral_dce_gen_trans_ref_dicoms(save_dir_path, img, hdr, trans_ref,
                                           suffix='', insert_bar=True,fs=3, mask_thr=.0,
                                           overlapped=True, start_count=0, max_count=50, base_sn=0):
    
    
    ## angle =  can find in ground truth phase map folder name  
    img = partial(my_rotate, img, -angle)
    img = img.transpose([0, 2, 1, 3])

    # import matplotlib.pyplot as plt
    # for i in range(img.shape[1]):
    #     save_path  = '/home/yejin/data/img_trans/' + str(i) +'.png'
    #     plt.imsave(save_path, img[0, i, :, :], cmap = 'gray')
    # exit()

    # Thresholding
    r_img *= (r_img > 0)
    r_TTPmap *= (r_TTPmap > 0)  # * mask

    r_mask = r_img.sum(axis=0)
    # r_mask = r_mask > mask_thr  # (r_mask.max() * mask_thr)
    # r_mask = np.array([remove_small_holes(_m, 1e3) for _m in r_mask], dtype=r_mask.dtype)
    # r_mask = erode_mask(r_mask, 9)[np.newaxis]

    # for i in range(len(r_hdr)):
    #     r_hdr[i].ImagePositionPatient = swap_axis_coor(r_hdr[i].ImagePositionPatient)
    # hdr = r_hdr

    # """Color TTPmap"""
    # number_of_slice = r_phase_map.shape[1]  # update the new number of slice
    # color_bar = ColorBar(*r_phase_map.shape[-2:]) if insert_bar else None
    # """Normalize phase_maps for grayscale DICOM generation"""
    # r_TTPmap_normed = r_TTPmap.copy()
    # for m in [r_phase_map, mip_phase_map, inv_mip_phase_map, r_TTPmap_normed]:  # r_p_phase_map
    #     m[:] = (m / m.max())  # * d
    # update_prog_bar(.35)

    # # Color TTPmap
    # cR = np.reshape([1.0, 0.0, 0.0], (3, 1))  # RED
    # cO = np.reshape([1.0, 0.5, 0.0], (3, 1))  # ORANGE
    # cY = np.reshape([1.0, 1.0, 0.0], (3, 1))  # YELLOW
    # cG = np.reshape([0.0, 1.0, 0.0], (3, 1))  # GREEN
    # cB = np.reshape([0.0, 0.0, 1.0], (3, 1))  # BLUE
    # cN = np.reshape([0.0, 0.0, 0.5], (3, 1))  # NAVY
    # cP = np.reshape([0.5, 0.0, 0.5], (3, 1))  # PURPLE

    # cTTPmapAP = np.zeros((3, nos, r_phase_map.shape[-2], r_phase_map.shape[-1]), dtype=D_TYPE)
    # for c in range(3):
    #     cTTPmapAP[c] = r_TTPmap[0] / r_TTPmap[0].max()
    # cTTPmapVP = cTTPmapAP.copy()

    # peak_time1 = time_points[1] * minTempRes
    # for z in range(cTTPmapAP.shape[1]):
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1) & (r_TTPmap[0, z] < peak_time1 + minTempRes))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cR
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 1 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 2 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cO
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 2 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 3 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cY
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 3 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 4 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cG
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 4 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 5 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cB
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 5 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 6 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cN
    #     loc = np.where((r_TTPmap[0, z] >= peak_time1 + 6 * minTempRes) & (
    #             r_TTPmap[0, z] < (peak_time1 + minTempRes + 7 * minTempRes)))
    #     cTTPmapAP[:, z, loc[0], loc[1]] = cP

    # _, eap, svp, evp = [int(tp) for tp in time_points]
    # mvp = svp + int(round((evp - svp) / 2.))
    # try:
    #     bar_txt = np.load('%s/mr_collateral_barTXT.npy' % resource_path('extra_data'))
    # except:
    #     bar_txt = np.load(f'{os.path.abspath(os.curdir)}/extra_data/mr_collateral_barTXT.npy')

    # for z in range(cTTPmapVP.shape[1]):
    #     loc = np.where((r_TTPmap[0, z] == 0))
    #     cTTPmapVP[:, z, loc[0], loc[1]] = r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()
    #     loc = np.where((r_TTPmap[0, z] > 0) & (r_TTPmap[0, z] <= eap * minTempRes))  # 1
    #     cTTPmapVP[:, z, loc[0], loc[1]] = cR * (r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()) * 2
    #     loc = np.where((r_TTPmap[0, z] > eap * minTempRes) & (r_TTPmap[0, z] <= svp * minTempRes))  # 2
    #     cTTPmapVP[:, z, loc[0], loc[1]] = cY * (r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()) * 2
    #     loc = np.where((r_TTPmap[0, z] > svp * minTempRes) & (r_TTPmap[0, z] <= mvp * minTempRes))  # 3
    #     cTTPmapVP[:, z, loc[0], loc[1]] = cG * (r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()) * 2
    #     loc = np.where((r_TTPmap[0, z] > mvp * minTempRes) & (r_TTPmap[0, z] <= evp * minTempRes))  # 4
    #     cTTPmapVP[:, z, loc[0], loc[1]] = cB * (r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()) * 2
    #     loc = np.where((r_TTPmap[0, z] > evp * minTempRes))
    #     cTTPmapVP[:, z, loc[0], loc[1]] = cP * (r_TTPmap[0, z, loc[0], loc[1]] / r_TTPmap.max()) * 2  # 5

    # cTTPmapVP = cTTPmapVP * r_mask
    # for z in range(cTTPmapVP.shape[1]):
    #     cTTPmapVP[:, z, -42:-3, -27:-3] = bar_txt
    # update_prog_bar(.4)

    # """Save Dicom Files"""
    # loop_idx_g = itertools.product(range(r_phase_map.shape[0]), range(number_of_slice))
    # pm = []
    # n = len(list(copy.deepcopy(loop_idx_g)))

    # suffix = f'_{tck}/{dis}mm'
    # strd = partial(save_trans_ref_dicom, col_dir_paths, r_mask, hdr, col_sf, (outliers, outliers_c_collateral),
    #                color_bar,
    #                r_phase_map, mip_phase_map, inv_mip_phase_map, r_TTPmap_normed, cTTPmapVP, base_sn, suffix)

    # with ThreadPoolExecutor(NUM_WORKERS) as executor:
    #     for i, result in enumerate(executor.map(strd, loop_idx_g)):
    #         pm.append(result)
    #         if ((i + 1) % (n // 10) == 0) or ((i + 1) == n):
    #             update_prog_bar(0.4 + 0.6 * ((i + 1) / n))

    # window['_dce_prog_bar_ref_'].update(100, visible=False) if max_count == 100 else None

    # message = 'Done!\n   Total time: %.2f\n   DICOM are stored in %s' % (time.time() - tic, col_dir_org)
    # print(message)

    # return np.asarray(pm).reshape((r_phase_map.shape[0], number_of_slice,) + pm[0].shape[-3:])
