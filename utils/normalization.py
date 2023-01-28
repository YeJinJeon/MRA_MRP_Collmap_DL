import numpy as np


def est_outlier_thr(x, thr):
    """"""
    x_s = np.sort(x.flat)
    lthr = x_s[int(np.floor(len(x_s) * thr) - 1)]  # lower threshold
    uthr = x_s[int(np.floor(len(x_s) * (1 - thr) - 1))]  # upper threshold
    return lthr, uthr


def norm_ab(x, a, b, predefined_minmax=None, exclude_outliers=False, mask=None, to_mask=False):
    """Normalize input to a-b range"""
    mask = np.ones_like(x) if mask is None else mask
    x_norm = x[:]

    if predefined_minmax is not None:
        xmin = predefined_minmax[0]
        xmax = predefined_minmax[1]
    else:
        if exclude_outliers:
            xmin, xmax = est_outlier_thr(x[mask > 0], 0.01)
            x[x > xmax] = xmax
            x[x < xmin] = xmin
        else:
            xmax = max(x[mask == 1])
            xmin = min(x[mask == 1])

    if to_mask:
        x_norm[mask > 0] = (b - a) * ((x[mask > 0] - xmin) / (xmax - xmin)) + a
        x_norm = x_norm * mask
    else:
        x_norm = (b - a) * ((x - xmin) / (xmax - xmin)) + a
        x_norm[x_norm < 0] = 0
    return x_norm