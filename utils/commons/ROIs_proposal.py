import numpy as np
import pylab as plt
from copy import deepcopy

from skimage.draw import ellipse  # ellipse
from scipy.signal import find_peaks, peak_widths

from utils.commons.error_logger import Logger


# from utils.misc import savitzky_golay


class ROIsProposal:
    def __init__(self, num_interval_x=9, num_interval_y=9, top_k=15, upper_time_point=60, cap_len=3,
                 mask_art_range_cols=(None, None), mask_art_range_rows=(None, 6),
                 mask_ven_range_cols=(4, 5), mask_ven_range_rows=(6, None),
                 slice_range=((55, 77), (60, 120)), earliest_peak=5, peak_width_min=1, prefix=''):  # ((5, 9), (7, 14))
        """
        :param upper_time_point: peak signal after this time point does not count
        :param top_k: specify the number of recommended regions kept by the system
        :param num_interval_y:
        :param num_interval_x:
        """
        self.logger = Logger('log_ROIsProposal_failure')
        self.slice_range = slice_range
        self.prefix = prefix
        self.earliest_peak = earliest_peak
        self.peak_width_min = peak_width_min
        self.num_interval_x, self.num_interval_y = num_interval_x, num_interval_y
        self.top_k = top_k
        self.upper_time_point = upper_time_point
        self.cap_len = cap_len
        self.ROIs_proposed_mask, self.ROIs_best_loc = (None,) * 2
        self.ROIs_all_sorted_loc = [None, None]
        self.marr = mask_art_range_rows
        self.marc = mask_art_range_cols
        self.mvrr = mask_ven_range_rows
        self.mvrc = mask_ven_range_cols

    @staticmethod
    def evaluate_ROIs(SUB, centroid, slice_range, ax=None, upper_time_limit=35, gamma=.85, art_peak=None,
                      min_phase_difference=3, peak_width_min=1):
        """
        Evaluate and find the best ROI (among different shapes and sizes) for a region
        :param slice_range:
        :param ax:
        :param SUB:
        :param region:
        :return:
        """
        # centroid = region.centroid
        best_score, best_rr, best_cc, best_peak, best_peak_val, best_sl = 0, None, None, None, None, None
        if (centroid[0] < slice_range[0]) or (centroid[0] > slice_range[1]):
            return best_score, best_rr, best_cc, best_peak, best_sl

        shape_fns = {
            # 'c': (circle, [3, 4, 5]),
            # 'e': (ellipse, [(3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4)]),
            'c': (ellipse, [(3, 4)])
        }  # circle and ellipse
        if ax is not None:
            best_y, best_peak, best_rh, best_peak_width = (None,) * 4
        for (_, item) in shape_fns.items():
            shape_fn = item[0]
            radii = item[1]
            for radius in radii:
                if isinstance(radius, tuple):
                    rr, cc = shape_fn(centroid[1], centroid[2], radius[0], radius[1])
                else:
                    rr, cc = shape_fn(centroid[1], centroid[2], radius)
                rr[rr >= SUB.shape[2]] = SUB.shape[2] - 1
                cc[cc >= SUB.shape[3]] = SUB.shape[3] - 1
                y = SUB[:, int(round(centroid[0])), rr, cc].mean(axis=(1,))
                # y = savitzky_golay(y, window_size=11)
                peaks, _ = find_peaks(y)
                if len(peaks) == 0:
                    continue
                peak = np.array([peaks[y[peaks].argmax()]])
                if art_peak is not None:
                    if (peak - art_peak) < min_phase_difference:
                        continue
                # penalized if the time series contains near peaks
                penalty = 50 if np.any(find_peaks(y[peak[0]:])[0] < 3) else 0

                if (peak >= upper_time_limit) or (y[peak] < y.max()) or np.any(
                        y[peak[0] + 5:] > (y[peak[0]] * gamma)):  # Extra conditions to check the peak validity
                    return best_score, best_rr, best_cc, best_peak, best_peak_val, best_sl
                results_half = peak_widths(y, peak, rel_height=.4)
                peak_width = results_half[3][0] - results_half[2][0]
                if peak_width < peak_width_min:
                    continue

                dist_penalty = 1 if art_peak is None else min_phase_difference/(peak[0] - art_peak[0])
                score = 10 * y[peak[0]] / peak_width - penalty
                score *= dist_penalty

                if score > best_score:
                    best_score, best_rr, best_cc, best_peak, best_peak_val, best_sl = \
                        score, rr, cc, peak, y[peak][0], int(round(centroid[0]))
                    if ax is not None:
                        best_y, best_rh, best_peak_width = y, results_half, peak_width

                # if best_peak == -1:
                #     fig, ax = plt.subplots(1, 1)
                #     ax.plot(y)
                #     ax.set_ylim([0, 350])
                #     ax.plot(best_peak, y[best_peak], "x")
                #     ax.hlines(*results_half[1:], color="C3")
                #     ax.text(results_half[3][0] + 1, results_half[1][0] + .05,
                #             '%.2f' % (peak_width,))
                #     ax.text(10, 50, '%.1f' % best_score)
                #     ax.text(best_peak[0] - 1, y[best_peak] - .08, best_peak[0])
                #     ax = None
                #     print(score)
                #     plt.show()

        if (ax is not None) and (best_peak is not None):
            ax.plot(best_y)
            ax.set_ylim([0, 350])
            ax.plot(best_peak, best_y[best_peak], "x")
            ax.hlines(*best_rh[1:], color="C3")
            ax.text(best_rh[3][0] + 1, best_rh[1][0] + .05,
                    '%.2f' % (peak_width,))
            ax.text(10, 50, '%.1f' % best_score)
            ax.text(best_peak[0] - 1, best_y[best_peak] - .08, best_peak[0])
            ax.axis('off')
            # if art_peak is None:
            #     _, ax1 = plt.subplots(1, 1)
            #     ax1.plot(best_y)
            #     ax1.set_ylim([0, 350])
            #     ax1.plot(best_peak, best_y[best_peak], "x")
            #     ax1.hlines(*best_rh[1:], color="C3")
            #     ax1.text(best_rh[3][0] + 1, best_rh[1][0] + .05,
            #             '%.2f' % (peak_width,))
            #     ax1.text(10, 50, '%.1f' % best_score)
            #     ax1.text(best_peak[0] - 1, best_y[best_peak] - .08, best_peak[0])
            #     # ax1.axis('off')
            #     plt.savefig(f'xx{np.random.choice(3)}')
            #     plt.close(_)

        return best_score, best_rr, best_cc, best_peak, best_peak_val, best_sl

    def handle_failed_cases(self, window, path, error=None):
        """"""
        self.logger.log_patient_id(path)

        # window[f'_{self.prefix}_auto_roi_'].Disabled = True
        # window[f'_{self.prefix}_auto_roi_'].update(disabled=True)
        window[f'_{self.prefix}_roi_art_'].update(disabled=False)
        window[f'_{self.prefix}_roi_vein_'].update(disabled=False)
        ms = 'Unable to automatically find ROIs. Please select the ROIs manually.'
        ms += f' [{error}]' if error else ''
        return window

    @staticmethod
    def find_earliest_peaks(peaks, _, earliest_peak, *args):
        """"""
        peaks = np.array(peaks).T[0]
        peaks[peaks < earliest_peak] = 99
        if len(np.unique(peaks)) > 1:
            upper_bound = np.unique(peaks)[1]  # only consider the first two earliest peak time)
        else:
            upper_bound = np.unique(peaks)[0] 
        return np.where(peaks <= upper_bound)[0]

    @staticmethod
    def find_peaks_after_art(peaks, peaks_val, _, art_peak, art_peak_val, min_cap_len=3, max_cap_len=3):
        """"""
        peaks = np.array(peaks).T[0]
        peaks_f = np.where((peaks >= (art_peak + min_cap_len)) & (peaks <= (art_peak + max_cap_len)))[0]
        peaks_f = [i for i in peaks_f if peaks_val[i] >= (art_peak_val * .7)]
        if len(peaks_f) == 0:
            return np.where(peaks >= (art_peak + min_cap_len))[0]
        return peaks_f

    @staticmethod
    def filter_and_sort(a_list, f_idx, scores):
        """"""
        scores = [scores[k] for k in f_idx]
        a_list = [a_list[k] for k in f_idx]
        ranks = np.argsort(scores)[::-1]
        return [a_list[k] for k in ranks]

    def propose_ROIs(self, SUB, mask, visualize=False, npy_dir=None):
        """
        :param visualize:
        :param mask:
        :param SUB:
        :return:
        """

        SUB = SUB[:self.upper_time_point]
        ny, nx = SUB.shape[-2:]
        x = np.linspace(0, nx - 1, self.num_interval_x + 1).astype(int)
        y = np.linspace(0, ny - 1, self.num_interval_y + 1).astype(int)
        mask_art = np.zeros((ny, nx))
        range_x = (
            x[self.marc[0]] if self.marc[0] is not None else None,
            x[self.marc[1]] if self.marc[1] is not None else None
        )
        range_y = (
            y[self.marr[0]] if self.marr[0] is not None else None,
            y[self.marr[1]] if self.marr[1] is not None else None
        )
        mask_art[range_y[0]:range_y[1], range_x[0]:range_x[1]] = 1
        range_x = (
            x[self.mvrc[0]] if self.mvrc[0] is not None else None,
            x[self.mvrc[1]] if self.mvrc[1] is not None else None
        )
        range_y = (
            y[self.mvrr[0]] if self.mvrr[0] is not None else None,
            y[self.mvrr[1]] if self.mvrr[1] is not None else None
        )
        mask_ven = np.zeros((ny, nx))
        mask_ven[range_y[0]:range_y[1], range_x[0]:range_x[1]] = 1

        thr_mask = np.zeros(((2,) + SUB.shape[1:]))
        for i_phase, slr in enumerate(self.slice_range):
            thr_mask[i_phase, slr[0]:slr[1]] = 1

        # from functools import partial
        # mt = partial(montage, padding_width=1)
        #
        # tp = SUB.mean(axis=(1, 2, 3)).argmax()
        # im = SUB[tp] * mask[0]
        # im[mask[0] == 0] = im.min()
        # ma = np.repeat(mask_art[np.newaxis], len(mask[0]), axis=0) * thr_mask[0]
        # mv = np.repeat(mask_ven[np.newaxis], len(mask[0]), axis=0) * thr_mask[1]
        # im = mt(im)
        # plt.figure(1, figsize=(5, 5*(im.shape[0]/im.shape[1])))
        # plt.imshow(im, cmap='gray')
        # plt.contour(mt(ma), colors='r', linewidths=.1)
        # plt.contour(mt(mv), colors='blue', linewidths=.1)
        # plt.axis('off')
        # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        # plt.savefig('a', dpi=200)

        # TODO: This needs to be tested further
        xmin, xmax = 0, SUB.shape[3] - 5
        ymin, ymax = 0, SUB.shape[2] - 5
        xi = np.linspace(int(xmin), int(xmax), int((xmax - xmin) / 2))
        yi = np.linspace(int(ymin), int(ymax), int((ymax - ymin) / 2))
        xi, yi = np.meshgrid(xi, yi)
        grid = np.zeros(SUB.shape[2:])
        grid[yi.astype(int), xi.astype(int)] = 1
        grid = grid[np.newaxis]

        peak1_time = SUB.max(axis=1)[:, mask_art > 0].mean(axis=1).argmax()
        peak2_time = SUB.max(axis=1)[:, mask_ven > 0].mean(axis=1).argmax()
        thr_mask[0] = SUB[peak1_time - 4:peak1_time + 2].mean(axis=0) * (mask_art * mask * thr_mask[0]) * grid
        thr_mask[1] = SUB[peak2_time - 2:peak2_time + 4].mean(axis=0) * (mask_ven * mask * thr_mask[1]) * grid
        # thr1, thr2 = np.percentile(thr_mask[0][thr_mask[0] > 0], 99.5), \
        #              np.percentile(thr_mask[1][thr_mask[1] > 0], 99.5)

        # plt.figure(2, figsize=(5, 5*(im.shape[0]/im.shape[1])))
        # plt.imshow(im, cmap='gray')
        # plt.contour(mt(thr_mask[0]), colors='r', linewidths=.1)
        # plt.contour(mt(thr_mask[1]), colors='blue', linewidths=.1)
        # plt.axis('off')
        # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        # plt.savefig('b', dpi=200)

        pre_thr_mask = deepcopy(thr_mask)

        thr1, thr2 = np.percentile(thr_mask[0][thr_mask[0] > 0], 95), \
                     np.percentile(thr_mask[1][thr_mask[1] > 0], 95)  # 95 99
        thr_mask[0] = thr_mask[0] > thr1
        thr_mask[1] = thr_mask[1] > thr2

        # plt.figure(3, figsize=(5, 5*(im.shape[0]/im.shape[1])))
        # plt.imshow(im, cmap='gray')
        # plt.contour(mt(thr_mask[0]), colors='r', linewidths=.1)
        # plt.contour(mt(thr_mask[1]), colors='blue', linewidths=.1)
        # plt.axis('off')
        # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        # plt.savefig('c', dpi=200)
        # plt.close('all')

        ROIs_proposed_mask = np.zeros_like(thr_mask)
        ROIs_best_loc = [None, None]

        """Phase 1: ROIs evaluation"""
        art_peak, art_peak_val = None, None
        for i_phase, _ in enumerate(thr_mask):
            tm = np.asarray(np.where(thr_mask[i_phase] == 1)).T
            scores, rrs, ccs, peaks, peaks_val, sls = [], [], [], [], [], []
            if visualize:
                _, axes = plt.subplots(5, int(np.ceil(len(tm) / 5)), figsize=(10, 8), num=10)
                axes = axes.flatten()
            else:
                axes = (None,) * len(tm)

            # rm_idx = []
            for gamma in [.85, .9, 1]:  # Reduce the constraint if the evaluator fails to get any ROIs
                for i, centroid in enumerate(tm):
                    score, rr, cc, peak, peak_val, sl = self.evaluate_ROIs(SUB, centroid, self.slice_range[i_phase],
                                                                           axes[i], gamma=gamma, art_peak=art_peak,
                                                                           peak_width_min=self.peak_width_min)

                    if peak is None:
                        # rm_idx.append(i)
                        continue
                    scores.append(score), rrs.append(rr), ccs.append(cc), peaks.append(peak), sls.append(sl)
                    peaks_val.append(peak_val)
                if len(peaks) > 0:
                    break
            if len(peaks) == 0:
                return

            """Phase 2: Filtering and sorting ROIs"""
            # Sort the proposed ROIs from highest score to lowest score
            f_func = self.find_peaks_after_art if i_phase == 1 else self.find_earliest_peaks
            f_idx = []
            cap_len = self.cap_len
            self.earliest_peak = int(np.mean(peaks) - 3) if self.earliest_peak == -1 else self.earliest_peak
            while (len(f_idx) == 0) and (cap_len > 0):
                f_idx = f_func(peaks, peaks_val, self.earliest_peak, art_peak, art_peak_val, cap_len)
                if len(f_idx) == 0:
                    cap_len -= 1
            f_idx = f_func(peaks, peaks_val, self.earliest_peak, art_peak, art_peak_val, cap_len)
            if len(f_idx) == 0:
                f_idx = np.arange(len(tm))

            print([p[0] for p in peaks])
            rrs = self.filter_and_sort(rrs, f_idx, scores)
            ccs = self.filter_and_sort(ccs, f_idx, scores)
            sls = self.filter_and_sort(sls, f_idx, scores)
            peaks_val = self.filter_and_sort(peaks_val, f_idx, scores)
            peaks = self.filter_and_sort(peaks, f_idx, scores)
            print([p[0] for p in peaks])

            art_peak = peaks[0]
            art_peak_val = peaks_val[0]

            self.ROIs_all_sorted_loc[i_phase] = [sls, rrs, ccs, peaks]
            for j in range(len(rrs)):
                if j == self.top_k:
                    break
                # centroid = region.centroid
                rr, cc, sl = rrs[j], ccs[j], sls[j]
                ROIs_proposed_mask[i_phase, sl, rr, cc] = len(rrs) - j

                if j == 0:
                    ROIs_best_loc[i_phase] = [sl, rr, cc]
                    art_peak = peaks[j]
            if visualize:
                plt.show()

        self.ROIs_proposed_mask = ROIs_proposed_mask
        # TODO: FIX THE ISSUES OCCURED WHEN auto_roi failed
        self.ROIs_best_loc = ROIs_best_loc


class ROIsProposalDSC(ROIsProposal):
    def __init__(self, num_interval_x=9, num_interval_y=9, top_k=15, upper_time_point=60, cap_len=3,
                 mask_art_range_cols=(None, None), mask_art_range_rows=(None, 5),
                 mask_ven_range_cols=(4, 5), mask_ven_range_rows=(6, None),
                 slice_range=((5, 9), (7, 14)), earliest_peak=14, peak_width_min=-1):
        super().__init__(num_interval_x, num_interval_y, top_k, upper_time_point, cap_len,
                         mask_art_range_cols, mask_art_range_rows,
                         mask_ven_range_cols, mask_ven_range_rows,
                         slice_range, prefix='dsc', earliest_peak=-1, peak_width_min=peak_width_min)


class ROIsProposalDCE(ROIsProposal):
    def __init__(self, num_interval_x=9, num_interval_y=9, top_k=15, upper_time_point=60, cap_len=3,
                 mask_art_range_cols=(2, 7), mask_art_range_rows=(4, 6),
                 mask_ven_range_cols=(2, 8), mask_ven_range_rows=(None, 3),
                 slice_range=((50, 90), (40, 140)), earliest_peak=0, peak_width_min=2):
        super().__init__(num_interval_x, num_interval_y, top_k, upper_time_point, cap_len,
                         mask_art_range_cols, mask_art_range_rows,
                         mask_ven_range_cols, mask_ven_range_rows,
                         slice_range, prefix='dce', earliest_peak=-1, peak_width_min=peak_width_min)


def get_rois_proposal(prefix):
    if prefix == 'dce':
        return ROIsProposalDCE
    return ROIsProposalDSC


if __name__ == "__main__":
    _sub, _mask = np.load(r'D:\ASUS\Research\MRA_GUI/SUB_1.npy'), \
                  np.load(r'D:\ASUS\Research\MRA_GUI/mask_1.npy')
    roi_proposals = ROIsProposalDCE()
    roi_proposals.propose_ROIs(_sub, _mask, visualize=False)

    # COPY THESE TO THE propose_ROIs method to save input for debugging
    # np.save('SUB', SUB)
    # np.save('mask', SUB)
    # exit()
