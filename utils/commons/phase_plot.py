import numpy as np
import matplotlib
import pylab as plt


class PhasePlot:
    def __init__(self, time_points_keys, text_size_scale=1., text_size=12, prefix='dsc'):
        self.text_size = text_size * text_size_scale
        self.phases = {
            f'_{prefix}_roi_art_': {'color': 'r', 'legend': 'Arterial Peak Phase', 'linewidth': 2,
                                    'fontdict': {'family': 'consolas', 'color': 'red', 'weight': 'normal',
                                                 'size': self.text_size},
                                    'loc': (.57, .9),
                                    },
            f'_{prefix}_roi_vein_': {'color': 'b', 'legend': 'Venous Peak Phase', 'linewidth': 2,
                                     'fontdict': {'family': 'consolas', 'color': 'blue', 'weight': 'normal',
                                                  'size': self.text_size},
                                     'loc': (.57, .8),
                                     },
        }
        self.set_tight_layout = False
        self.aspan_list, self.axv_list, self.tx_list = (None,) * 3
        self.time_points_keys = time_points_keys

    def compute_time_points(self, art_roi, ven_roi, values):
        """A placeholder method. Please see the actual implementation of each modality in the subclass"""
        art_start, art_peak, even_peak, lven_end, even_end = (0,) * 5
        time_points = []
        return art_start, art_peak, even_peak, lven_end, even_end, time_points

    def display_phase(self, ax, art_roi, ven_roi, values,
                      alpha=.5, span_top=.95, asp_margin=.2, linewidth=1, linestyle='-.', msdp=None):
        """
        Display phase information when ROIs of both phase available
        :param values:
        :param ax:
        :param art_roi: DscROI instance of Arterial phase
        :param ven_roi: DscROI instance of Venous phase
        :param alpha:
        :param span_top:
        :param asp_margin:
        :param linewidth:
        :param linestyle:
        :return:
        """
        # Validate the phase of the ROIs, swap if necessary
        phase_names = [*self.phases]
        if art_roi.phase != phase_names[0]:
            tmp = art_roi
            art_roi = ven_roi
            ven_roi = tmp

        art_start, art_peak, even_peak, lven_end, even_end, time_points = self.compute_time_points(art_roi, ven_roi,
                                                                                                   values)
        delay_end = min(art_roi.mean_values.shape[0], lven_end + (lven_end - even_peak))

        """Annotate phase peaks"""
        # Translate time points to the right by one to match with MATLAB version
        art_start, art_peak, even_peak, even_end, lven_end, delay_end = \
            art_start + 1, art_peak + 1, even_peak + 1, even_end + 1, lven_end + 1, delay_end + 1
        msdp = art_peak if msdp is None else msdp + 1  # maximum signal difference phase
        # AXVSPAN
        if self.aspan_list:
            [aspan.remove() for aspan in self.aspan_list]
            del self.aspan_list
        self.aspan_list = [ax.axvspan(art_start - asp_margin, art_peak + asp_margin * 2, 1 - span_top, span_top,
                                      facecolor='salmon',
                                      alpha=alpha - .1),  # art
                           ax.axvspan(art_peak + asp_margin * 3, even_peak - asp_margin * 2, 1 - span_top, span_top,
                                      facecolor='lightgreen', alpha=.35),  # cap
                           ax.axvspan(even_peak - asp_margin, even_end + asp_margin * 2, 1 - span_top, span_top,
                                      facecolor='blue',
                                      alpha=.2),  # even
                           ax.axvspan(even_end + asp_margin * 3, lven_end + asp_margin * 2, 1 - span_top, span_top,
                                      facecolor='blue', alpha=.2),  # lven
                           ax.axvspan(lven_end + asp_margin * 3, delay_end + asp_margin * 2, 1 - span_top, span_top,
                                      facecolor='moccasin', alpha=alpha),  # del
                           ax.axvspan(1, art_start - asp_margin * 2, 1 - span_top, span_top, facecolor='grey',
                                      alpha=alpha - .3),  # pre
                           ax.axvspan(delay_end + asp_margin * 3, len(art_roi.mean_values), 1 - span_top, span_top,
                                      facecolor='grey', alpha=alpha - .3)]  # post

        # AXVLINE
        if self.axv_list:
            [axv.remove() for axv in self.axv_list]
            del self.axv_list
        self.axv_list = [
            ax.axvline(x=art_start, color='r', linewidth=linewidth, linestyle=linestyle),  # art start
            ax.axvline(x=art_peak, color='r', linewidth=linewidth, linestyle=linestyle),  # art end
            ax.axvline(x=even_peak, color='blue', linewidth=linewidth, linestyle=linestyle),  # even start
            ax.axvline(x=even_end, color='blue', linewidth=linewidth, linestyle=linestyle),  # even end
            ax.axvline(x=lven_end, color='blue', linewidth=linewidth, linestyle=linestyle),  # lven end
            ax.axvline(x=delay_end, color='sienna', linewidth=linewidth, linestyle=linestyle),  # delay end
            ax.axvline(x=msdp, color='gray', linewidth=2, linestyle='--'),  # lven end
        ]

        # Add legend
        if self.tx_list:
            [tx.remove() for tx in self.tx_list]
            del self.tx_list
        self.tx_list = []
        for roi in [art_roi, ven_roi]:
            pp, pv = roi.peak_position + 1, roi.peak_value  # +1 to match with MATLAB
            tmp = self.phases[roi.phase]
            self.tx_list.append(
                ax.text(ax.get_xlim()[1] * tmp['loc'][0],
                        ax.get_ylim()[1] * tmp['loc'][1],
                        '%s: %d\n(%.3f)' % (tmp['legend'], pp, pv),
                        fontdict=tmp['fontdict'],
                        ))
        return time_points


class PhasePlotDSC(PhasePlot):
    def __init__(self, time_points_keys, text_size_scale=1., prefix='dsc'):
        super().__init__(time_points_keys, text_size_scale=text_size_scale)

    def compute_time_points(self, art_roi, ven_roi, values):
        """"""
        time_points = None if values['_dsc_auto_roi_phase_'] else [float(int(values[tpk])) - 1 for tpk in
                                                                   self.time_points_keys]

        if time_points is None:
            art_peak = float(art_roi.peak_position)
            even_peak = float(ven_roi.peak_position)
            art_start = max(art_peak - 5, 0)
            lven_end = even_peak + 6
            time_points = [art_start, art_peak, even_peak, lven_end]
        else:
            art_start, art_peak, even_peak, lven_end = time_points
        even_end = int(round((lven_end + even_peak) / 2.))

        return art_start, art_peak, even_peak, lven_end, even_end, time_points


class PhasePlotDCE(PhasePlot):
    def __init__(self, time_points_keys, text_size_scale=1.):
        super().__init__(time_points_keys, text_size_scale=text_size_scale, prefix='dce')

    def compute_time_points(self, art_roi, ven_roi, values):
        """"""
        time_points = None if values['_dce_auto_roi_phase_'] else [float(int(values[tpk]) - 1) for tpk in
                                                                   self.time_points_keys]

        art_start = 1
        lven_end = len(ven_roi.mean_values) - 1
        if time_points is None:
            art_peak = float(art_roi.peak_position)
            even_peak = float(ven_roi.peak_position)
            min_art_len = min(3, int(art_peak))  # new line
            for i in range(min_art_len, int(art_peak)):
                slop = art_roi.mean_values[int(art_peak) - i] - art_roi.mean_values[int(art_peak) - (i + 1)]
                if slop < 20:
                    art_start = art_peak - i
                    break
            if art_peak <= 5:
                art_start = 2

            if (even_peak - art_peak) < 4:
                min_ven_len, max_ven_len = 6, 7
            else:
                min_ven_len, max_ven_len = 7, 8

            for count, i in enumerate(range(int(even_peak) + min_ven_len, len(ven_roi.mean_values) - 1)):
                if count == (max_ven_len - min_ven_len + 1):
                    lven_end = i - 1
                    break
                slop = ven_roi.mean_values[i + 1] - ven_roi.mean_values[i]
                if slop > -8:
                    lven_end = i - 1  # -1 is preferred by the doctor
                    break
            lven_end = min(max(lven_end, even_peak + min_ven_len), even_peak + max_ven_len)
            art_start -= 1 if art_start == art_peak else 0  # made up, not true
            time_points = [art_start, art_peak, even_peak, lven_end]
        else:
            art_start, art_peak, even_peak, lven_end = time_points

        even_end = even_peak + np.floor((lven_end - even_peak) / 2)

        return art_start, art_peak, even_peak, lven_end, even_end, time_points


def get_phase_plot(prefix):
    if prefix == 'dce':
        return PhasePlotDCE
    return PhasePlotDSC


if __name__ == "__main__":
    ## THIS MAIN IS OBSOLETE. NEED MODIFICATIONS.

    matplotlib.use('TkAgg')
    import pylab as plt
    from utils.dsc_mrp.load_npy import load_npy
    from utils.commons.ROIs_selection import RoiSelector
    from utils.commons.show_slice_for_ROI import SliceViewerDSC as SliceViewer

    IMG, mask, hdr, _ = load_npy(
        dir_in='F:\Minh\projects\MRA\matlab_from_prof/BP510 2016-10-27-raw/PWI_DSC_Collateral/')

    minIMG = IMG.min(axis=0, keepdims=True)
    SUB = -IMG + IMG[0]
    tMIP = SUB.max(axis=0, keepdims=True)

    dsc_roi = {}
    slice_idx = 14

    for dsc_current_roi_phase in phases.keys():
        slice_viewer = SliceViewer(tMIP, minIMG, figsize=10)
        slice_viewer.show_slice(minIMG, slice_idx)
        roi_selector = RoiSelector('Ellipse')
        slice_viewer.fig.canvas.mpl_connect('key_press_event', roi_selector.toggle_selector)

        plt.show()
        if roi_selector.clicked:
            dsc_roi[dsc_current_roi_phase] = DscROI(roi_selector.gen_bin_roi(IMG[0, slice_idx].shape), slice_idx, SUB,
                                                    dsc_current_roi_phase)
        roi_selector.clicked = False

    fig, ax = plt.subplots(1, 1, num=4, figsize=(8, 8))
    for dsc_current_roi_phase in phases.keys():
        dsc_roi[dsc_current_roi_phase].plot(ax)

    # aspan_list, axv_list, tx_list = display_phase(ax, *[dsc_roi[p] for p in
    #                                                     [*dsc_roi]], )  # Be careful with the input order
    # plt.show()
    # print()
