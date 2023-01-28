import numpy as np
from skimage.feature import canny
from skimage.filters import sobel_v
from skimage.morphology import binary_erosion, disk
from skimage.transform import hough_line, hough_line_peaks
from skimage import measure


def find_top_cc(img_bw, top_k: int):
    """
    Get the largest connected component
    :param img_bw: grayscale image
    :param top_k: number of the top connected component
    :return: mask of top-k largest connected components
    """
    labels = measure.label(img_bw, return_num=False)
    count = np.bincount(labels.flat, weights=img_bw.flat)
    sort_count_idx = np.argsort(count)[::-1]
    maxCC_with_bcg = np.zeros_like(labels)
    top_k = len(count)-1 if len(count) < top_k else top_k
    for k in range(top_k):
        maxCC_with_bcg += labels == sort_count_idx[k]
    return maxCC_with_bcg


def save_fig(mip, mip_c, origin, angle, dist, prefix, dir_npy):
    """
    Save a figure indicating the rotation or tilted angle of the head
    :param mip: 2D Numpy array of maximum intensity projection
    :param mip_c:  2D Numpy array of edge detection
    :param origin:
    :param angle: estimated angle
    :param dist:
    :param prefix: filename prefix
    :param dir_npy: a string indicating the saving folder for generated figures
    :return:
    """
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    hw_ratio = mip.shape[-2] / mip.shape[-1]
    fig, ax = plt.subplots(1, 1, num='head_angle', figsize=(3, 3 * hw_ratio))
    ax.imshow(mip, cmap='gray')
    ax.plot(origin, (y0, y1), '-r')
    print(angle)
    ax.set_xlim(origin)
    ax.set_ylim((mip_c.shape[0], 0))
    ax.set_axis_off()
    ax.text(10, 10, f'{angle * (180 / np.pi):.1f}', color='g', fontsize=15)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.figure('head_angle')
    plt.savefig(f'{dir_npy}/../head_{prefix}_angle.png', dpi=200)
    plt.close('head_angle')


def estimate_rotation_angle(img, view='axial', dir_npy=None):
    """
    Estimate the head rotation (in axial view) or tilted (in coronal view) angle
    :param img:  3D Numpy array of maximum intensity projection through time
    :param view: either 'axial' or 'sagittal'
    :param dir_npy: a string indicating the saving folder for generated figures
    :return: estimated angle (in degree) and distance
    """
    if view == 'coronal':
        prefix, axis = 'tilted', 0
        mip = img.max(axis=axis)
        h, w = mip.shape
        mask = np.zeros_like(mip)
        mask[:int(h*.6), int(w / 2) - 30: int(w / 2) + 30] = 1
    else:
        prefix, axis = 'rotated', 1
        mip = img[:, :int(img.shape[1] / 2)].max(axis=axis)
        h, w = mip.shape
        mask = np.zeros_like(mip)
        # mask[:int(h / (4 / 3)), int(w / 2) - 30: int(w / 2) + 30] = 1
        mask[:, int(w / 2) - 30: int(w / 2) + 30] = 1

    # Edge detection
    mip_c = canny(mip * mask, 2, mip.max() * .1, mip.max() * .2)
    mip_c = sobel_v(mip_c * mask) > .5
    mip_c = mip_c * binary_erosion(mask, disk(2))  # Remove the vertical edge caused by the mask
    mip_c = find_top_cc(mip_c, 2)  # Keep only the largest connected component ~ the longest line

    # Angle estimation
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(mip_c, theta=tested_angles)

    origin = np.array((0, mip_c.shape[1]))
    angles = []
    dists = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        angles.append(angle), dists.append(dist),

    angle, dist = angles[np.argmin(np.abs(angles))], np.mean(dists)

    if dir_npy is not None:
        save_fig(mip, mip_c, origin, angle, dist, prefix, dir_npy)

    return angle * 180 / np.pi, dist
