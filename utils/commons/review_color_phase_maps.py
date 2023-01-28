import re
import numpy as np
import pylab as plt
from tools import PySimpleGUI as sg
from matplotlib import ticker

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PHASES = ['Arterial', 'Capillary', 'Early Venous', 'Late Venous', 'Delay']
PHASES_KEYS = ['art', 'cap', 'even', 'lven', 'del']


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def update_individual(event, values, pm, axes, fig_canvas_aggs, w):
    """"""
    phase_idx = int(re.findall('[0-9]', event)[0])
    if '_slider_' in event:
        sl_idx = int(values[event] - 1) if isinstance(values, dict) else int(values - 1)
        axes[phase_idx].set_data(pm[phase_idx, sl_idx])
        w[event](sl_idx + 1)
        w.write_event_value(f'_dce_draw_review_', phase_idx)


class ReviewColorPhaseMaps:
    """"""

    def __init__(self, phase_maps, image_type='DSC', main_window_loc: tuple = None):
        """

        :param pm: size of (No of Phase, Depth, Height Width, Channel(3))
        :param image_type:
        """
        self.image_type = image_type.upper()
        self.layout, self.w = (None,) * 2
        self.canvas_keys, self.slider_keys, self.checkbox_keys = [], [], []
        self.fig_names, self.figs, self.axes, self.fig_canvas_aggs = [], [], [], []
        self.in_sync = {}
        phase_maps = self.make_dict(phase_maps)
        phase_maps = dict(sorted(phase_maps.items()))
        self.create_fig(phase_maps)
        self.create_layout(phase_maps)
        self.display_window(main_window_loc)
        for i, (pm, _) in phase_maps.items():
            self.in_sync[i] = []
            self.update(f'_r{i}_select_all_', {f'_r{i}_select_all_': True}, pm)

    @staticmethod
    def make_dict(phase_map):
        """Convert phase map into dictionary if it is currently in Numpy format"""
        phase_maps = {}
        if isinstance(phase_map, np.ndarray):
            phase_maps[0] = (phase_map, '')
        else:
            phase_maps = phase_map
        return phase_maps

    def create_fig(self, phase_maps: dict):
        """"""
        for i, (pm, _) in phase_maps.items():
            for j in range(pm.shape[0]):
                self.fig_names.append(f'{self.image_type}{i}: {PHASES[j]}')
                hw_ratio = pm.shape[-3] / pm.shape[-2]
                plt.figure(self.fig_names[-1], figsize=(3.5, 3.5 * hw_ratio))
                self.axes.append(plt.imshow(pm[j, int(np.floor(pm.shape[1] / 2))], ))
                plt.axis('off')

                # Setup axis
                ax = plt.gca()
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                plt.tight_layout(pad=0)
                self.figs.append(plt.gcf())

    def create_layout(self, phase_maps: dict):
        """"""
        self.layout = []
        for i, (pm, label) in phase_maps.items():
            frames = []
            for j in range(len(PHASES)):
                _, _, figure_w, figure_h = self.figs[i * len(PHASES) + j].bbox.bounds
                self.canvas_keys.append(f'_r{i}_{j}_fig_{PHASES_KEYS[j]}_')
                self.slider_keys.append(f'_r{i}_{j}_slider_{PHASES_KEYS[j]}_')
                self.checkbox_keys.append(f'_r{i}_{j}_checkbox_{PHASES_KEYS[j]}_')
                col = [
                    [sg.Checkbox(PHASES[j], key=self.checkbox_keys[-1], enable_events=True)],
                    [sg.Canvas(size=(figure_w, figure_h), key=self.canvas_keys[-1])],
                    [sg.Slider((1, pm.shape[1]), resolution=1, orientation='h', key=self.slider_keys[-1],
                               enable_events=True, default_value=10)],
                ]
                frames += [sg.Column(col, element_justification='center'), ]
            self.layout.append([sg.Checkbox('Select all', key=f'_r{i}_select_all_', enable_events=True,
                                            default=True)])
            self.layout.append([sg.T(label)])
            self.layout.append(frames)

    def display_window(self, main_window_loc=None):
        self.w = sg.Window(f'{self.image_type}: Preview Collateral Images', self.layout, alpha_channel=0,
                           keep_on_top=True).Finalize()

        # Set location of the preview window (based on the current location of the main window)
        current_loc = self.w.CurrentLocation()
        if main_window_loc is not None:
            new_loc = list((current_loc[0] + main_window_loc[0], current_loc[1]))
            self.w.move(*new_loc)
        self.w.set_alpha(1)

        for i, fig in enumerate(self.figs):
            self.fig_canvas_aggs.append(draw_figure(self.w[self.canvas_keys[i]].TKCanvas, fig))

    def update(self, event, values, pm):
        """"""
        if event == '__TIMEOUT__':
            return
        if event == '_dce_draw_review_':
            self.fig_canvas_aggs[values[event]].draw()
            return
        row_idx = int(re.findall('_r\d+_*', event)[0][2:-1])
        in_sync = self.in_sync[row_idx]
        n = len(PHASES)
        if isinstance(pm, dict):
            pm = pm[row_idx][0]
        if '_select_all_' in event:
            if values[event]:
                for i in range(n):
                    if i not in in_sync:
                        in_sync.append(i)
                        self.w[self.checkbox_keys[row_idx * n + i]](True)
                phase_idx = in_sync[0] if in_sync else 0
            else:
                self.in_sync[row_idx] = []
                [self.w[self.checkbox_keys[row_idx * n + i]](False) for i in range(len(PHASES))]
                return
        else:
            phase_idx = int(re.findall('_\d+_*', event)[0][1:-1])  # int(re.findall('[0-9]', event)[0])
            if '_checkbox_' in event:
                self.w[f'_r{row_idx}_select_all_'](False)
                if values[event]:
                    in_sync.append(phase_idx)
                else:
                    in_sync.remove(phase_idx) if phase_idx in in_sync else None

        if phase_idx in in_sync:
            _value = values[self.slider_keys[in_sync[0]]] if '_checkbox_' in event else values[event]

            for i in in_sync:
                _event = self.slider_keys[row_idx * n + i]
                self.update_individual(_event, _value, pm)
        else:
            self.update_individual(event, values, pm)

    def update_individual(self, event, values, pm):
        """"""
        row_idx = int(re.findall('_r\d+_*', event)[0][2:-1])
        phase_idx = int(re.findall('_\d+_*', event)[0][1:-1])
        n = len(PHASES)
        if '_slider_' in event:
            sl_idx = int(values[event] - 1) if isinstance(values, dict) else int(values - 1)
            self.axes[row_idx * n + phase_idx].set_data(pm[phase_idx, sl_idx])
            self.fig_canvas_aggs[row_idx * n + phase_idx].draw()
            self.w[event](sl_idx + 1)


if __name__ == "__main__":
    # phase_map = np.load(
    #     r'F:\Minh\projects\MRA\matlab_from_prof\BP455 2018-10-09\PWI_DSC_Collateral\NpyFiles\pm_color.npy')
    _phase_map = np.load(
        r'F:/Minh/projects/MRA/matlab_from_prof/BP455 2018-10-09/DMRA_DCE_Collateral_crop\NpyFiles\pm_color.npy')

    review_pm = ReviewColorPhaseMaps(_phase_map)
    review_pm.update('_select_all_', True, _phase_map)
    while True:
        __event, __values = review_pm.w.read(timeout=10)
        if __event is None:
            break
        review_pm.update(__event, __values, _phase_map)
    review_pm.w.close()
