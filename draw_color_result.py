import os
import numpy as np
import pylab as plt
from skimage.util import montage

from numba import njit
from skimage import exposure

def norm_range(x, a, b):
    return (b - a) * ((x - x.min()) / (x.max() - x.min())) + a

def linear_decay(x, x_max, x_min):
    return (x_min - x_max) * x + x_max

def exp_decay(x, alpha=1.5, beta=10):
    # z = np.linspace(0, 1, n)
    y = alpha ** (-beta * x) + 1
    y = norm_range(y, 1, alpha)
    return y

def decayed_root(x, max_root=1.2, method='exp'):
    if method == 'exp':
        root_nums = exp_decay(x, max_root, 10)
    else:
        root_nums = linear_decay(x, max_root, 1)
    y = np.power(x, 1 / root_nums)
    return y

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

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

def get_color():
    """
    :return: a colormap jet
    """
    from pylab import cm
    cm_jet = np.reshape(np.concatenate([cm.jet(i) for i in range(255)], axis=0), (255, 4))
    return np.vstack((np.array([0, 0, 0]), cm_jet[:, :3]))


JET = get_color()


class ColorBar:
    def __init__(self, height, width):
        """"""
        bar_tmp = np.zeros((101, 10))
        for i in range(100):
            bar_tmp[i] = 99 - i + 1
        # bar position
        self.end_bar_x = int(width - 3)
        self.start_bar_x = int(self.end_bar_x - 10 + 1)
        self.start_bar_y = int(np.floor((height - 101) / 2))
        self.end_bar_y = int(self.start_bar_y + 101 - 1)
        # Index Image generation
        self.bar = bar_tmp * 2.551  # colorbar scale from 0 to 255
        try:
            self.numimg = np.load('%s/mr_collateral_numing.npy' % resource_path('extra_data')) / 255
        except:
            self.numimg = np.load(f'{os.path.abspath(os.curdir)}/extra_data/mr_collateral_numing.npy') / 255

    def insert_num_board(self, rgbIMG, maxIMG, maxWWL):
        """
        :param indIMG: a 2D image
        :return: add min - max numbers to the colorbar
        """
        # insert min number
        for ch in range(rgbIMG.shape[-1]):
            rgbIMG[self.end_bar_y:self.end_bar_y + 9, self.end_bar_x - 6:self.end_bar_x, ch] = self.numimg[..., 0, 0]

        # insert max number
        max_num_str = str(int(maxIMG * maxWWL * 255))
        str_length = len(max_num_str)
        num_board = np.zeros((9, str_length * 6, 3))
        for i in range(str_length):
            selected_num = int(max_num_str[i])
            num_board[:, i * 6:(i + 1) * 6, 0] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 1] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 2] = self.numimg[:, :, 0, selected_num]
        rgbIMG[self.start_bar_y - 9:self.start_bar_y, self.end_bar_x - str_length * 6 + 1:self.end_bar_x + 1] = \
            num_board
        return rgbIMG

    def insert_color_bar(self, indIMG):
        """
        :param indIMG: a 2D image
        :return: an image with a color bar on the right side
        """
        # insert bar
        indIMG = indIMG[0] if indIMG.ndim > 2 else indIMG
        indIMG[self.start_bar_y:self.end_bar_y + 1, self.start_bar_x:self.end_bar_x + 1] = self.bar
        return indIMG

def mr_collateral_gen_color_image(inIMG, outlier, mask=None, rescaling_first=True, color_bar=None, WWL=None,
                                  from_deep_learning=False, decay_root=False):
    """

    :param WWL:
    :param from_deep_learning:
    :param inIMG:
    :param outlier:
    :param mask:
    :param rescaling_first:
    :param color_bar:
    :param decay_root:
    :return:
    """
    # if mask is None:
    #     mask = np.ones_like(inIMG)
    outlier = (outlier / 2, outlier / 2) if not isinstance(outlier, tuple) else outlier
    minIMG, maxIMG = inIMG.min(), max(inIMG.max(), 1)
    if WWL is None:  # for DCE
        if rescaling_first:
            # Image Signal Normalization
            nIMG = inIMG / maxIMG
            # nIMG = nIMG ** (1 / 2)
            outlier_low = (outlier[0] / 100)  # / 2
            outlier_high = 1 - (outlier[1] / 100)  # / 2)
            WWL = stretch_lim(nIMG[mask > 0], outlier_low, outlier_high)  # auto window width / level
            minWWL, maxWWL = WWL.min(), WWL.max()
            # Rescaled Image
            nIMG = nIMG[0] if nIMG.ndim > 2 else nIMG
            rsIMG = exposure.rescale_intensity(nIMG, tuple(WWL)) if not from_deep_learning else nIMG
            if decay_root:
                rsIMG = decayed_root(rsIMG, 1.1)  # decayed_discontinued_2d(rsIMG)
        else:
            rsIMG = inIMG / maxIMG
            maxWWL = 1
    else:
        minWWL, maxWWL = WWL
        rsIMG = inIMG

    indIMG = (rsIMG * 255).astype('uint8') * mask.astype('uint8')
    indIMG = color_bar.insert_color_bar(indIMG) if color_bar else indIMG
    labels = np.unique(indIMG)
    rgb = np.zeros((indIMG.shape + (3,)))
    for label in labels:
        loc = np.where(indIMG == label)
        rgb[loc[0], loc[1], :] = JET[label]
    # TODO: I cannot understand why the intensity of the bar is higher than the brain intensity, regarding 'indIMG',
    #  but seems to be the equal regarding 'rgb'
    rgb = color_bar.insert_num_board(rgbIMG=rgb, maxIMG=maxIMG, maxWWL=maxWWL) if color_bar else rgb
    # plt.imshow(rgb)
    return rgb

def save_fig(pred, gt, dir_in, prefix=None, color=False):
    if color:
        fig, ax = plt.subplots(1, 2, num='result', figsize=(20, 10))
        fig.suptitle(prefix)
        ax[0].imshow(montage(pred[:-2, :, :], grid_shape=(5, 5), multichannel= True))
        ax[1].imshow(montage(gt[:-2, :, :], grid_shape=(5, 5),  multichannel= True))
        plt.figure('result')
        plt.savefig(f'{dir_in}/result_{prefix}.png', dpi=150)
        plt.close('result')
    else:
        fig, ax = plt.subplots(1, 2, num='data', figsize=(20, 10))
        fig.suptitle(prefix)
        ax[0].imshow(montage(pred[:-2, :, :], grid_shape=(5, 5)), cmap="gray")
        ax[1].imshow(montage(gt[:-2, :, :], grid_shape=(5, 5)),  cmap="gray")
        plt.figure('data')
        plt.savefig(f'{dir_in}/data_{prefix}.png', dpi=150)
        plt.close('data')

if __name__ == "__main__":
    # data_dir = '/data1/yejin/mra_npy/CMC_DATA/Abnormal_No597/2019Y/20190126_KU_6310333/DMRA_source_DCE_Collateral_crop_py/NpyFiles'
    # input = np.load(data_dir + '/IMG_n01.npy')
    # mask_4d = np.load(data_dir + '/mask_4d.npy')[0]
    
    result_dir = '/data1/yejin/compu/3d_ppmdnn/result/20190126_KU_6310333'
    save_dir = '/data1/yejin/compu/3d_ppmdnn/evaluation_metrics'
    predict = np.load(result_dir +'/predict.npy')
    gt = np.load(result_dir +'/gt.npy')
    mask = np.load(result_dir +'/mask.npy')

    insert_bar = True
    outliers_c_collateral = [(.01, 2), ] + [(4, 6), ] + [(3, 9.5), ] + [(2, 11.5), ] * 2
    color_bar = ColorBar(*predict.shape[-2:]) if insert_bar else None
    rescaling_first = True
    
    phasemap_result = []
    gt_result = []

    for ind, phase in enumerate(['art', 'cap', 'even', 'lven', 'del']):

        phasemap_result = []
        gt_result = []

        for sl in range(predict.shape[1]):

            # save_trans_ref_dicom
            adj_sl = [max(sl - j, 0) for j in range(5)] + [min(sl + j, predict.shape[1] - 1) for j in range(1, 5)]
            _, idx = np.unique(adj_sl, return_index=True)
            adj_sl = [adj_sl[_idx] for _idx in np.sort(idx)]

            phase_map_target = predict[ind, sl, :, :]
            gt_target = gt[ind, sl, :, :]

            # save_color_dcm
            col_sf = 10
            phase_map_target = phase_map_target * col_sf
            gt_target = gt_target * col_sf

            predict_color_slice = mr_collateral_gen_color_image(phase_map_target, outliers_c_collateral[ind], mask[0,sl], rescaling_first, color_bar, from_deep_learning=True)
            gt_color_slice = mr_collateral_gen_color_image(gt_target, outliers_c_collateral[ind], mask[0,sl], rescaling_first, color_bar, from_deep_learning=True)
            phasemap_result.append(predict_color_slice)
            gt_result.append(gt_color_slice)

        phasemap_array = np.array(phasemap_result)
        gt_array = np.array(gt_result)
        # save color
        save_fig(phasemap_array, gt_array, save_dir, prefix=phase, color=True)
        # save gray
        save_fig(predict[ind], gt[ind], save_dir, prefix=phase)
    
