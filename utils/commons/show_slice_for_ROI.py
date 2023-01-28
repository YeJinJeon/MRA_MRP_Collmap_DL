# import pylab as plt
import tools.pyplot as plt
from tools import PySimpleGUI as sg
from utils.commons.misc import draw_figure

font_sl = {'family': 'consolas',
           'color': 'lightgreen',
           'weight': 'normal',
           'size': 15,
           }
font_title = {'family': 'consolas',
              'color': 'yellow',
              'weight': 'bold',
              'size': 15,
              }


class SliceViewer:
    """
    With matplotlib widgets, the update method works best if you create an artist object and adjust its value
    """

    def __init__(self, tMIP, minIMG=None, figsize=6, slice_idx=0, ROIs_proposed_mask=None,
                 vmax_coef=.8, fig_titles=(), prefix='', location=(0, 0)):
        hw_ratio = tMIP.shape[-2] / tMIP.shape[-1]
        self.prefix = prefix
        plt.switch_backend('TkAgg')
        self.fig, self.ax = plt.subplots(1, 1, num=fig_titles[0],
                                         figsize=(figsize, figsize * hw_ratio),
                                         facecolor='black')
        self.t = self.ax.text(int(tMIP.shape[-1] / 2) - len(str(slice_idx)) - 10, 10, slice_idx, fontdict=font_sl)
        if minIMG is not None:
            self.ca = self.ax.imshow(minIMG[0, slice_idx - 1], cmap='gray')
        else:
            self.ca = self.ax.imshow(tMIP[0, slice_idx - 1], cmap='gray')
        self.ax.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        self.ROIs_proposed_mask = ROIs_proposed_mask
        self.cs = []
        self.contour_colors = ['r', 'b']
        self.add_contours(slice_idx)
        self.max_tMIP = tMIP.max() * vmax_coef
        self.ca.set_clim(0, self.max_tMIP)
        self.t.set_text(slice_idx)
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (320 + location[0], 120 + location[1]))

    def display_window(self):
        self.cv_w = sg.Window('ROI Selection', layout=[[self.cv_layout]]).Finalize()
        self.fig_agg = draw_figure(self.cv_w[self.cv_key].TKCanvas, self.fig)

    def reset_contours(self):
        for cs in self.cs:
            if cs is not None:
                [c.remove() for c in cs.collections]
        self.cs = []

    def add_contours(self, slice_idx):
        if self.ROIs_proposed_mask is not None:
            for ROIs_proposed_mask, contour_colors in zip(self.ROIs_proposed_mask, self.contour_colors):
                if ROIs_proposed_mask[slice_idx - 1].sum() > 0:
                    self.cs.append(self.ax.contour(ROIs_proposed_mask[slice_idx - 1],
                                                   colors=contour_colors, linewidths=.3))

    def show_slice(self, minIMG, slice_idx):
        self.reset_contours()
        self.add_contours(slice_idx)
        self.ca.set_data(minIMG[0, slice_idx - 1])
        self.ca.set_clim(0, self.max_tMIP)
        self.t.set_text(slice_idx)

    def mpl_connect(self, process, values, window):
        """"""
        k = f'_{self.prefix}_slice_pos_'
        press = [None] * 2

        def on_scroll(_event):
            update(int(values[k]) + 1 if _event.button == 'up' else int(values[k]) - 1)

        def on_key_pressed(_event):
            key_dict = {
                'left': -10,
                'right': 10,
                'down': -1,
                'up': 1,
            }
            if _event.key not in key_dict.keys():
                return
            update(int(values[k]) + key_dict[_event.key])

        def update(new_slice_idx):
            new_slice_idx = 1 if new_slice_idx < 0 else new_slice_idx
            new_slice_idx = window[k].Values[-1] if new_slice_idx > window[k].Values[-1] else new_slice_idx
            window[k].update(new_slice_idx)
            values[k] = new_slice_idx
            process(k, values, window)

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', on_key_pressed)  # same keyword


class SliceViewerDSC(SliceViewer):
    def __init__(self, tMIP, minIMG, figsize=6, slice_idx=0, ROIs_proposed_mask=None, location=(0, 0)):
        from layouts.figure_titles import DSC_FIG_TITLES as FIG_TITLES
        super().__init__(tMIP, minIMG, figsize=figsize, slice_idx=slice_idx,
                         ROIs_proposed_mask=ROIs_proposed_mask, fig_titles=FIG_TITLES, prefix='dsc',
                         location=location)


class SliceViewerDCE(SliceViewer):
    def __init__(self, tMIP, figsize=6, slice_idx=0, ROIs_proposed_mask=None, location=(0, 0)):
        from layouts.figure_titles import DCE_FIG_TITLES as FIG_TITLES
        super().__init__(tMIP, figsize=figsize, slice_idx=slice_idx,
                         ROIs_proposed_mask=ROIs_proposed_mask, vmax_coef=.3,
                         fig_titles=FIG_TITLES, prefix='dce', location=location)


class ROIsReviewer:
    """
    With matplotlib widgets, the update method works best if you create an artist object and adjust its value
    """

    def __init__(self, tMIP, minIMG=None, ROIs=dict, figsize=(float, float), fig_titles=(), vmax_coef=.8,):
        """

        :param tMIP:
        :param minIMG:
        :param ROIs: dict dsc_roi
        :param figsize:
        """
        self.fig, self.ax = plt.subplots(1, 2, num=fig_titles[-1],
                                         figsize=(figsize[0], figsize[1]),
                                         facecolor='black')
        self.t_sl, self.t_title, self.ca = [], [], []
        self.titles = ['Arterial', 'Venous']
        self.colors = ['r', 'b']
        self.max_tMIP = tMIP.max() * vmax_coef
        for i, ROI in enumerate(ROIs):
            slice_idx = ROIs[ROI].slice_idx
            self.ax[i].axis('off')
            self.t_sl.append(
                self.ax[i].text(10, 25, slice_idx, fontdict=font_sl))
            self.t_title.append(self.ax[i].text(int(tMIP.shape[-1] / 2) - len(self.titles[i]) - 15,
                                                int(tMIP.shape[-2] - 7), self.titles[i], fontdict=font_title))
            if minIMG is not None:
                self.ca.append(self.ax[i].imshow(minIMG[0, slice_idx - 1], cmap='gray'))
            else:
                self.ca.append(self.ax[i].imshow(tMIP[0, slice_idx - 1], cmap='gray'))
            self.cs = self.ax[i].contour(ROIs[ROI].roi, colors=self.colors[i])
            self.ca[i].set_clim(0, self.max_tMIP)
            self.t_sl[i].set_text(slice_idx)
            self.t_title[i].set_text(self.titles[i])
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        # plt.tight_layout()

    def show_slice(self, minIMG, ROIs):
        for i, ROI in enumerate(ROIs):
            slice_idx = ROIs[ROI].slice_idx
            self.ca[i].set_data(minIMG[0, slice_idx - 1])
            [c.remove() for c in self.cs.collections]
            self.ax[i].contour(ROIs[ROI].roi)
            self.ca[i].set_clim(0, self.max_tMIP)
            self.t_sl[i].set_text(slice_idx)


class ROIsReviewerDSC(ROIsReviewer):
    cv_name = '_dsc_cv_roi_reviewer_'

    def __init__(self, tMIP, minIMG, ROIs, figsize=(6, 12)):
        from layouts.figure_titles import DSC_FIG_TITLES as FIG_TITLES
        super().__init__(tMIP, minIMG, ROIs, figsize=figsize, fig_titles=FIG_TITLES,)


class ROIsReviewerDCE(ROIsReviewer):
    cv_name = '_dce_cv_roi_reviewer_'

    def __init__(self, tMIP, _, ROIs, figsize=(6, 12)):
        from layouts.figure_titles import DCE_FIG_TITLES as FIG_TITLES
        super().__init__(tMIP, None, ROIs=ROIs, figsize=figsize, fig_titles=FIG_TITLES,
                         vmax_coef=.3)


def get_slice_viewer(prefix):
    if prefix == 'dce':
        return SliceViewerDCE
    return SliceViewerDSC


def get_rois_reviewer(prefix):
    if prefix == 'dce':
        return ROIsReviewerDCE
    return ROIsReviewerDSC
