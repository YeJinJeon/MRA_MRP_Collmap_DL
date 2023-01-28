import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening

from skimage import measure
from skimage.filters import rank
from skimage.measure import label
from skimage.util import img_as_ubyte
from skimage.morphology import disk, remove_small_holes, remove_small_objects, cube

from utils.commons.misc import timer
from utils.commons.image import Image4D, Image3D
from utils.commons.misc import savitzky_golay
from utils.dce_mra.correct_head_angle import find_top_cc
from utils.commons.misc import sc_mt, s_mt, show, _maximize


def find_max_cc(img_bw):
    """
    Get the largest connected component
    :param img_bw: grayscale image
    :return: mask of largest connected component
    """
    labels = measure.label(img_bw, return_num=False)
    count = np.bincount(labels.flat, weights=img_bw.flat)
    maxCC_withbcg = labels == np.argmax(count)
    return maxCC_withbcg


def remove_non_skull(sub, skull_mask, peak_general):
    """

    :param sub:
    :param skull_mask:
    :param peak_general:
    :return:
    """
    skull_mask = skull_mask.astype('bool')
    for i, _m in enumerate(skull_mask):
        if _m.sum() == 0:
            continue
        skull_mask[i] = binary_opening(skull_mask[i], disk(2))
        labels = measure.label(_m, return_num=False)  # binary_opening(_m)
        n = np.unique(labels)
        label_range = range(1, (len(n))) if len(n) > 1 else range(len(n))
        for j in label_range:
            avg_signal = sub[:, i, labels == j].mean(axis=1)
            if avg_signal.argmax() == 0:
                continue
            avg_signal = (avg_signal - avg_signal.mean()) / avg_signal.std()
            avg_signal = savitzky_golay(avg_signal, 5, 2)
            peaks, _ = find_peaks(avg_signal)

            remove = False
            if len(peaks) < 5:
                remove = True if np.abs(avg_signal.argmax() - peak_general) <= 10 else False

            if remove:
                skull_mask[i][labels == j] = 0
    skull_mask = [remove_small_objects(sk, 1e2) for sk in skull_mask]

    return np.array(skull_mask, dtype='uint8')


def remove_holes(_img):
    return remove_small_holes(binary_closing(_img, disk(9)), 2e3)


def extract_brain_per_slice(image, head_mask, thr):
    """

    :param image:
    :param head_mask:
    :param thr:
    :return:
    """
    if head_mask.sum() == 0:
        return head_mask, 0
    img_mean = image[head_mask > 0].mean()
    image = img_as_ubyte(image)

    ph, pw = 10, 5
    image = np.pad(image, ((ph, ph), (pw, pw)))
    head_mask = np.pad(head_mask, ((ph, ph), (pw, pw)))

    if img_mean < thr:
        denoised = rank.median(image, disk(15))
        markers = rank.gradient(denoised, disk(9)) < 20
        markers = binary_closing(markers, disk(5))
    else:
        markers = head_mask
        head_mask = binary_erosion(head_mask, iterations=12)
    return find_max_cc(markers * head_mask)[ph:-ph, pw:-pw], img_mean


@timer
def extract_brain_v0(img, head_mask):
    """

    :param img:
    :param head_mask:
    :return:
    """
    border_idx = 7
    head_mask[:border_idx] = 0
    head_mask[-border_idx:] = 0
    tg = img.min(axis=0)
    tg /= tg.max()
    thr = tg[border_idx][head_mask[border_idx] > 0].mean()
    brain_masks, img_means = [], []

    for _image, _mask in zip(tg, head_mask):
        brain_mask, img_mu = extract_brain_per_slice(_image, _mask, thr)
        brain_masks.append(brain_mask)
        img_means.append(img_mu)

    brain_masks = np.array(brain_masks, dtype='uint8')
    dif_mask = head_mask - brain_masks
    first_section = np.min(np.argmin(np.array(img_means) > thr))
    brain_masks[:first_section - 1][tg[:first_section - 1] > tg[dif_mask > 0].mean() * 1.5] = 0
    return brain_masks


@timer
def get_foreground(img, num_worker=4):
    """

    :param img:
    :param num_worker: number of workers used in concurrent morphological operations. num_worker=4 was shown to provide
    least processing time.
    :return:
    """
    ph, pw = 50, 50

    img_mean = img.mean(axis=0)
    img_mean = np.pad(img_mean, ((0, 0), (ph, ph), (pw, pw)))

    mu1 = img_mean.mean() * 1
    foreground = img_mean > mu1
    with ThreadPoolExecutor(num_worker) as executor:
        results = executor.map(remove_holes, foreground)
    foreground = np.array([fg for fg in results])

    # foreground = binary_erosion(foreground, cube(9))

    foreground = foreground[:, ph:-ph, pw:-pw].astype('uint8')
    return foreground


def _strip_skull(img):
    """

    :param img:
    :return:
    """
    """STAGE 1: SKULL SEGMENTATION"""
    img_max = img.max(axis=0)
    skull = []  # m2 is the skull mask
    for i, x in enumerate(img_max):
        thr = x[x > 0].mean() * 1
        m = x > thr
        skull.append(find_max_cc(m))
    for i, (sk, _img_min) in enumerate(zip(skull, img_max)):
        skull[i] = _img_min > (_img_min[sk].mean() * 1)

    """STAGE 2: NON-SKULL REMOVAL"""
    skull = np.array(skull)
    sub = img - img[0]
    peak_general = sub[:, (skull == 0) * (img[0] > 0)].mean(axis=1).argmax()
    skull = remove_non_skull(sub, skull, peak_general)
    skull = binary_closing(skull, cube(3))
    return skull.astype('uint8')


def _remove_skull(foreground, skull):
    """

    :param foreground:
    :param skull:
    :return:
    """
    final_mask = []
    for fg, sk in zip(foreground, skull):
        fg -= sk
        final_mask.append(find_max_cc(fg))
    final_mask = np.array(final_mask, dtype='uint8')
    return final_mask


def strip_skull_v1(img, return_fg=False):
    """"""
    """STEP 1: BACKGROUND REMOVAL"""
    foreground = get_foreground(img)
    img = img * foreground

    """STEP 2: SKULL SEGMENTATION IN CORONAL VIEW"""
    skull_c = _strip_skull(img)

    """STEP 3: SKULL SEGMENTATION IN AXIAL VIEW"""
    skull_a = _strip_skull(img.transpose([0, 2, 1, 3]))
    skull_a = skull_a.transpose([1, 0, 2])

    """STEP 4: COMBINE SKULL SEGMENTATION"""
    skull = (skull_c + skull_a) > 0

    """STEP 5: REMOVE SKULL"""
    foreground = _remove_skull(foreground, skull)
    img = img * foreground

    if return_fg:
        return img, foreground

    return img


@timer
def strip_skull(img, return_fg=False):
    """"""
    """STEP 1: BACKGROUND REMOVAL"""
    foreground = get_foreground(img)
    # plt.imshow(montage(img.mean(axis=0)), cmap='gray')
    # plt.contour(montage(foreground))
    # plt.show()
    # exit()
    img = img * foreground

    """STEP 2: SKULL SEGMENTATION IN AXIAL VIEW"""
    skull = _strip_skull(img)
    # skull = _strip_skull(img.transpose([0, 2, 1, 3]))
    # skull = skull.transpose([1, 0, 2])

    """STEP 3: REMOVE SKULL"""
    foreground = _remove_skull(foreground, skull)
    img = img * foreground

    if return_fg:
        return img, foreground

    return img


def inv(_mask: np.ndarray):
    """
    Inverse binary map
    :param _mask: Numpy array
    :return:
    """
    return np.abs(1 - _mask).astype('uint8')


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


def close_mask(_mask: np.ndarray, pad_size: int, structure=None):
    """
    Pad and erode the 3D-mask using 3D operation without removing first and last slides and lower regions
    :param structure:
    :param _mask: 3D Numpy array
    :param pad_size: size of erosion (and padding as well)
    :return:
    """
    _mask = np.pad(_mask, pad_size, constant_values=1)
    _mask = np.array([binary_closing(_m, structure=structure, iterations=pad_size) for _m in _mask])
    _mask = _mask[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size].astype('uint8')
    return _mask


def erode(image: Image4D, pad_size: int = 10):
    """
    Wrapper function for 'erode_mask' function
    Pad and erode the 3D-mask using 3D operation without removing first and last slides and lower regions
    :param image: Image object
    :param pad_size: size of erosion (and padding as well)
    :return: Image object after eroded
    """
    image.mask = erode_mask(image.mask, pad_size)

    # Post-process
    image.axial()
    for i, m in enumerate(image.mask):
        if i > image.mask.shape[0] - 20:
            image.mask[i] = remove_small_objects(binary_closing(image.mask[i].astype('bool'), disk(3)), 1e2)
            image.mask[i] = remove_small_holes(image.mask[i].astype('bool'), 2e3)
    image.remove_background()
    return image


def erode_head_back(image: Image4D, pad_size: int = 2):
    """
    Remove non-brain area at the back of the head by erosion with mask in sagittal view
    :param image: Image object
    :param pad_size: int
    :return: Image object
    TODO: Rename the function
    """
    image.sagittal()
    target_array = image.max
    _, h, w = target_array.shape
    non_brain_mask = np.zeros_like(target_array, dtype='uint8')
    non_brain_mask[:, :, int(w * .7):] = 1
    non_brain_mask *= image.mask
    non_brain_mask = erode_mask(non_brain_mask, pad_size)
    image.mask[:, :, int(w * .7) + pad_size:] = non_brain_mask[:, :, int(w * .7) + pad_size:]

    # non_brain_mask[image.min < 60] = 0
    # non_brain_mask2 = np.zeros_like(non_brain_mask)
    # non_brain_mask2[target_array < 30] = 1
    # non_brain_mask2 *= non_brain_mask
    # non_brain_mask2 = np.array([binary_closing(nbm, iterations=1) for nbm in non_brain_mask2])
    # non_brain_mask2 = close_mask(non_brain_mask2, 2)
    image.coronal()
    return image


def localize_bright_non_brain_per_slide(slide, offset_fraction):
    """

    :param slide: slide index
    :param offset_fraction:
    :return:
    """
    tmp = np.zeros_like(slide) + slide
    index = np.where(tmp > 0)
    unique_row, unique_col = np.unique(index[0]), np.unique(index[1])
    center_idx = np.array([np.median(unique_row), np.median(unique_col)], dtype='int')
    h, w = unique_row.max() - unique_row.min(), unique_col.max() - unique_col.min()

    tmp[tmp < 40] = 0
    # Threshold image to localize bright area
    thr = np.percentile(tmp[index], 50)
    labels = label(tmp > thr)
    for i_label in np.unique(labels):
        idx = np.where(labels == i_label)
        if (idx[0].mean() < (center_idx[0] - h / offset_fraction)) & (np.abs(idx[1].mean() - center_idx[1]) < (w / 2)):
            labels[labels == i_label] = 0

    # Morphological operations to localize and refine bright non-brain area
    non_brain_mask = labels > 0
    non_brain_mask = find_top_cc(non_brain_mask, 10)
    non_brain_mask = binary_dilation(non_brain_mask, iterations=2)
    non_brain_mask = remove_small_objects(non_brain_mask, 5e2)

    # import pylab as plt
    # plt.imshow(tmp, cmap='gray')
    # plt.contour(non_brain_mask)
    # plt.show()

    return non_brain_mask


def remove_bright_non_brain(image: Image4D):
    """
    Find and remove the bright non-brain area in coronal view
    :param image: Image object
    :return: Image object
    """
    image.coronal()
    non_brain = []
    target_array = image.min

    # Detect bright non-brain area (coarse)
    border_idx = 7  # The number of first and last slides discarded
    for i, sl in enumerate(target_array):
        if i < border_idx:
            non_brain.append(np.ones_like(sl))
        elif i > 40:
            non_brain.append(np.zeros_like(sl))
        else:
            non_brain.append(localize_bright_non_brain_per_slide(sl, i))
    non_brain = close_mask(np.array(non_brain), 10)
    image.mask = inv(non_brain) * image.mask

    """Post-process"""
    # Keep only the largest connected component and remove holes
    image.axial()
    target_array = image.min
    for i, _ in enumerate(image.mask):
        # image.mask[i] = remove_small_objects(image.mask[i].astype('bool'), 1e2)
        image.mask[i][target_array[i] <= 12] = 0
        image.mask[i] = remove_small_holes(find_max_cc(image.mask[i]), 2e2)
    image.sagittal()
    for i, _ in enumerate(image.mask):
        if image.mask[i].sum() == 0:
            continue
        image.mask[i] = remove_small_holes(find_max_cc(image.mask[i]), 2e2)

    # Refine the mask / remove redundant part at the lower region
    image.coronal()
    upper_mask = np.ones_like(image.mask[0])
    upper_mask[int(upper_mask.shape[0] * .6):] = 0
    for i, _ in enumerate(image.mask):
        if np.logical_or(image.mask[i].sum() == 0, i > image.shape[1] / 2):
            continue
        check_mask = find_max_cc((image.mask[i] + upper_mask) > 0)
        image.mask[i] *= check_mask
        image.mask[i] = remove_small_holes(image.mask[i].astype('bool'), 5e2)

    # Smoothen the mask
    image.sagittal()
    image.mask = close_mask(image.mask, 3)
    for i, _ in enumerate(image.mask):
        if image.mask[i].sum() == 0:
            continue
        image.mask[i] = remove_small_holes(find_max_cc(image.mask[i]), 1e3)

    # Remove non-brain area at the back of the head
    image = erode_head_back(image, 2)

    # Preserve the major blood vessels
    image.coronal()
    sub = (image.arr - image.arr[0]).max(axis=0)
    vessel_mask = (sub > 200).astype('uint8')
    image.mask = ((image.mask + vessel_mask) > 0).astype('uint8')
    image.mask = close_mask(image.mask, 2)

    # Re-adding the major blood vessels, since binary_closing may add nasal vessels, which are not necessary
    image.sagittal()
    vessel_mask = (sub > 300).astype('uint8').transpose([2, 1, 0])
    image.mask = np.array([find_max_cc(_m) for _m in image.mask])
    image.mask = ((image.mask + vessel_mask) > 0).astype('uint8')

    image.coronal()
    return image


def consistent_io(func):
    """A decorator function to improve robustness of 'extract_brain' function regarding input/output"""

    def wrapper(*args, **kwargs):
        """"""
        image = args[0]
        return_array = False
        if isinstance(image, np.ndarray):
            return_array = True
            image = Image4D(image)
        image = func(image, *args[1:], **kwargs)
        if return_array:
            return image.arr
        return image

    return wrapper


def image_input(func):
    """A decorator function to ensure the first function input is an Image object"""

    def wrapper(*args, **kwargs):
        """"""
        image = args[0]
        if isinstance(image, np.ndarray):
            image = Image4D(image) if image.ndim == 4 else Image3D(image)
        return func(image, *args[1:], **kwargs)

    return wrapper


@timer
@consistent_io
def extract_brain(image, mask_foreground: np.ndarray = None, remove_background: bool = True):
    """
    Remove non brain area (fat, eyes, muscle, nasal & sinus signal, skull, scalp, etc.)
    :param image: 4D Numpy array or Image4D object
    :param mask_foreground:
    :param remove_background: whether to mask the input image with the mask
    :return:
    """
    image.mask = get_foreground(image.arr) if mask_foreground is None else mask_foreground

    image = erode(image, 8)
    image = remove_bright_non_brain(image)

    if remove_background:
        image.remove_background()

    return image


@image_input
def find_major_vein_idx(sub, mask_foreground: np.ndarray = None, remove_background: bool = True,
                        visualize=False):
    """
    Return the slide index of the major blood vessel (in sagittal view)
    :param sub: subtraction image (= image.arr - image.arr[0])
    4D Numpy array or Image4D object. image has to be created in the coronal view.
    :param mask_foreground:
    :param remove_background: whether to mask the input image with the mask
    :param visualize: show the slide selected
    :return:
    """
    sub.mask = get_foreground(sub.arr) if mask_foreground is None else mask_foreground
    sub.sagittal()
    tg = sub.mean if sub.arr.ndim == 4 else sub.arr

    z, h, w = tg.shape
    thr = np.percentile(tg[sub.mask == 1], 99)
    vessel_mask = tg > thr
    vessel_mask[:, int(h / 2):] = 0
    vessel_mask = close_mask(vessel_mask, 3)
    vessel_mask = np.array([find_max_cc(vm) for vm in vessel_mask], dtype='uint8')
    slide_idx = np.argmax(vessel_mask.sum(axis=(1, 2)))

    if visualize:
        mask_single_slide = np.zeros_like(vessel_mask)
        mask_single_slide[slide_idx] = 1
        s_mt(tg)
        sc_mt(vessel_mask * mask_single_slide)
        plt.show()
    return slide_idx


def create_nasal_mask(image: Image3D, mask_foreground, pixel_width=1):
    """

    :param image:
    :param mask_foreground:
    :param pixel_width:
    :return:
    """
    image.sagittal()

    # Construct a nasal mask (by anatomy ~ assume the targeted image is standard)
    tg_image = image.mean if len(image.shape) == 4 else image.arr
    _, h, w = tg_image.shape
    nasal_mask = np.ones_like((tg_image[0]))
    ratio_tri = .4
    ratio_rec = .45, .3
    nm_tri = np.tril(nasal_mask, -(int(h * ratio_tri)))
    nm_rec = np.zeros_like(nasal_mask)
    nm_rec[int(h * ratio_rec[0]):, :int(w * ratio_rec[1])] = 1
    nasal_mask = nm_tri * nm_rec

    # The mask includes only regions with high average signal (across time) within the anatomical mask
    l_thr, u_thr = np.percentile(tg_image, 70), np.percentile(tg_image, 100)
    nasal = nasal_mask * (np.logical_and(tg_image > l_thr, tg_image < u_thr))

    # Assume that the nasal cavity is 4 cm wide (~ [40/pixel_width] pixels for a voxel size of 1 x 1 x 1 mm)
    nasal_width = 45 / pixel_width
    major_vein_idx = find_major_vein_idx(image, mask_foreground)
    nasal[:major_vein_idx - int(nasal_width / 2.5), :-40] = 0
    nasal[major_vein_idx + int(nasal_width / 2.5):, :-40] = 0
    nasal[:major_vein_idx - int(nasal_width / 2), :-15] = 0
    nasal[major_vein_idx + int(nasal_width / 2):, :-15] = 0
    nasal[:major_vein_idx - int(nasal_width / 1.5), :] = 0
    nasal[major_vein_idx + int(nasal_width / 1.5):, :] = 0

    nasal[tg_image < np.percentile(tg_image, 95)] = 0
    image.mask = nasal

    # Make the mask rounded in axial view
    image.axial()
    # image.mask = binary_dilation(image.mask, cube(5))
    image.mask = [binary_dilation(m, disk(5)) for m in close_mask(image.mask, 5)]

    # Always return the image in coronal view
    image.coronal()
    return image.mask


@image_input
def create_angio_mask(image, is_sub=False, mask_foreground: np.ndarray = None, pixel_width: float = 1.,
                      visualize=False):
    """
    Remove non brain area (fat, eyes, muscle, nasal & sinus signal, skull, scalp, etc.)
    :param visualize:
    :param image: 4D Numpy array or Image4D object
    :param is_sub: whether the input image is the subtraction image
    :param mask_foreground:
    :param pixel_width: width of a pixel (in cm)
    :return: 3D Numpy array - the nasal mask used for MRA/MRV images
    """
    image.mask = get_foreground(image.arr) if mask_foreground is None else mask_foreground
    image.remove_background()

    # The mask will be created in the sagittal view using the subtraction image
    if is_sub:
        sub = image
    else:
        sub = image.arr - image.arr[0]
        sub = Image4D(sub)

    sub.mask = create_nasal_mask(sub, image.mask)

    # Remove eyes regions
    sub.sagittal()
    for i, ns in enumerate(sub.mask):
        if ns.sum() > 0:
            sub.mask[i] = find_max_cc(ns)

    # Post process the nasal mask
    sub.coronal()
    pad_size = 4
    sub.mask = np.pad(sub.mask, pad_size)
    for i, ns in enumerate(sub.mask):
        if ns.sum() > 0:
            sub.mask[i] = binary_opening(binary_closing(remove_small_objects(ns.astype('bool'), 40), iterations=2),
                                         iterations=4)
            # sub.mask[i] = find_max_cc(sub.mask[i])
        if np.all(sub.mask[i]):  # avoid having the whole slide with values 1
            sub.mask[i] *= 0

    # Keep only one blob in axial view
    sub.axial()
    mask_mip = sub.mask.max(axis=0)
    labels = label(mask_mip)
    if labels.max() > 1:  # there are multiple blobs in axial view
        expected_nasal_loc = np.zeros_like(mask_mip)
        h, w = expected_nasal_loc.shape
        expected_nasal_loc[:int(h*.15), int(w/2-25):int(w/2+25)] = 1  # nasal mask should be overlapped with this mask
        expected_overlapping_area = 30  # 30 is a random threshold
        for l in range(1, labels.max() + 1):
            if np.logical_and(labels == l, expected_nasal_loc).sum() < expected_overlapping_area:
                mask_mip[labels == l] = 0
        sub.mask *= mask_mip

    sub.mask = sub.mask[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

    sub.coronal()
    # return sub.mask

    if visualize:
        s_mt(sub.mean)
        sc_mt(sub.mask)
        plt.show()

    return mask_foreground * inv(sub.mask)

    # _, ax = plt.subplots(1, 3, num=1)
    # mips = []
    # nasal_masks = []
    # for k, view in enumerate(['coronal', 'sagittal', 'axial']):
    #     eval(f'sub.{view}()')
    #     mips.append(sub.max.max(axis=0))
    #     nasal_masks.append(sub.mask.max(axis=0))
    #     ax[k].imshow(mips[-1], cmap='gray')
    #     ax[k].contour(nasal_masks[-1])
    #     ax[k].axis('off')
    # _maximize()
    # plt.show()
    # return image


@image_input
def create_collateral_mask(image):
    mask_foreground = image.arr > 0

    nasal = create_nasal_mask(image, mask_foreground)

    image.mask = nasal
    # image.coronal()
    #
    # s_mt(image.arr)
    # sc_mt(image.mask)
    # show()
    mask = np.clip(mask_foreground * inv(nasal) + (nasal * .05), 0, 1)
    return mask


def try_angio_mask():
    for ii in (range(9, 10)):
        arr = np.load(f'../../img{ii}.npy')
        img = Image4D(arr)
        mask_angio = create_angio_mask(img, pixel_width=0.78, visualize=True)


def try_collateral_mask():
    arr = np.load(f'../../ColDel.npy')
    mask_collateral = create_collateral_mask(arr)
    image = Image3D(arr)
    image.mask = mask_collateral
    image.sagittal()
    s_mt(image.arr)
    sc_mt(image.mask)
    show()


if __name__ == '__main__':
    import pylab as plt
    try_collateral_mask()

    # for ii in range(1, 9):
    #     print(ii)
    #     arr = np.load(f'../../img{ii}.npy')
    #     img = Image4D(arr)
    #     img = extract_brain(img, remove_background=False)
    #
    #     # img.coronal()
    #     img.sagittal()
    #     s_mt(img.mean)
    #     sc_mt(img.mask)
    #     show()
    # plt.savefig(f'../../img{ii}.png')
    # plt.close()
    show()
