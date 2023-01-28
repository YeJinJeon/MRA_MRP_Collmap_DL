import pickle

import numpy as np
from scipy.ndimage import gaussian_filter as gf

from utils.commons.skull_stripping import create_collateral_mask

saved_vars = ['PreIMG', 'ColArt', 'ColCap', 'ColEVen', 'ColLVen', 'ColDel',
              'sap', 'eap', 'svp', 'mvp', 'evp', 'sigma'
              ]


def gen_phase_maps(IMG, roi, time_points, time_points_keys, sg, window, npy_folder, prefix='dsc', save_npy=True):
    """Generate phase maps from time_points and save each phase map to files"""
    # Checking if phase maps generation is feasible or not
    invalid_time_point = False
    ROIs_available = True if (len(roi) >= 2) else False
    if npy_folder is None:
        sg.Popup('Please load the data first!', keep_on_top=True, title='Message')
        return None, None
    if not ROIs_available:
        if len(roi) == 0:
            sg.Popup('Please select ROIs first!', keep_on_top=True, title='Message')
        else:
            if f'_{prefix}_roi_art_' not in roi.keys():
                sg.Popup('Please select Artery ROI first!', keep_on_top=True, title='Message')
            else:
                sg.Popup('Please select Vein ROI first!', keep_on_top=True, title='Message')
    else:
        delta_t = np.array([[_t - t for t in time_points] for _t in time_points])
        for i, dt in enumerate(delta_t):
            for j in range(i + 1, len(dt)):
                if dt[j] >= -1:
                    invalid_time_point = True
                    sg.Popup('%s must be at least 2 seconds greater than %s' % (
                        window['%stext_' % time_points_keys[j]].DisplayText,
                        window['%stext_' % time_points_keys[i]].DisplayText), keep_on_top=True, title='Message')
                    break
            if invalid_time_point:
                break
    # If all conditions satisfy
    if (not invalid_time_point) and ROIs_available:
        print('Generating collateral images...')
        sap, eap, svp, evp = [int(tp) for tp in time_points]
        mvp = int(round((evp + svp) / 2.)) if prefix == 'dsc' else svp + int(round((evp - svp) / 2.))

        gfs = int(window[f'_{prefix}_gaus_'].Get()[0])
        sigma = [0, gfs, gfs]

        ColMask = IMG.mean(axis=0) > 0
        if sap > 0:
            PreIMG = IMG[:sap].mean(axis=0)
            # ColMask = PreIMG > 0
            PreIMG = gf(PreIMG * ColMask, sigma)
        else:
            PreIMG = np.zeros_like(IMG[0]) + IMG[0]
        sign_coef = [-1, 1] if prefix == 'dsc' else [1, -1]
        ColArt = gf(IMG[sap:eap + 1].mean(axis=0) * ColMask, sigma) * sign_coef[0] + PreIMG * sign_coef[1]
        ColCap = gf(IMG[eap + 1:svp].mean(axis=0) * ColMask, sigma) * sign_coef[0] + PreIMG * sign_coef[1]
        ColEVen = gf(IMG[svp:mvp + 1].mean(axis=0) * ColMask, sigma) * sign_coef[0] + PreIMG * sign_coef[1]
        ColLVen = gf(IMG[mvp + 1:evp + 1].mean(axis=0) * ColMask, sigma) * sign_coef[0] + PreIMG * sign_coef[1]
        ColDel = gf(IMG[evp + 1: min(len(IMG), 2 * (evp + 1) - svp)].mean(axis=0) * ColMask, sigma) * sign_coef[0] + PreIMG * sign_coef[1]

        mask = create_collateral_mask(ColDel)[np.newaxis] if prefix == 'dce' else PreIMG[np.newaxis]

        if save_npy:
            for var in saved_vars:
                np.save('%s/%s.npy' % (npy_folder, var), eval(var))
            with open('%s/ROIs' % npy_folder, 'wb') as fp:
                pickle.dump(roi, fp)
        phase_maps = np.concatenate((ColArt[np.newaxis], ColCap[np.newaxis],
                                     ColEVen[np.newaxis], ColLVen[np.newaxis], ColDel[np.newaxis]), axis=0)
        phase_maps *= mask

        print('Done!')

        return phase_maps, gfs
    return (None,) * 2
