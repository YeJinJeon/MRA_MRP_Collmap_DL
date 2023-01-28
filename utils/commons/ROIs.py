import numpy as np


class ROI:
    """Handle ROI data and plots"""

    def __init__(self, roi, slice_idx, SUB, phase, prefix='dsc'):
        self.ax_length = SUB.shape[0] + 10
        self.roi = roi
        self.slice_idx = slice_idx
        self.phases = {
            f'_{prefix}_roi_art_': {'color': 'r', 'legend': 'Arterial Peak Phase',
                                    'fontdict': {'family': 'consolas', 'color': 'red', 'weight': 'normal', 'size': 14},
                                    'loc': (.57, .9),
                                    },
            f'_{prefix}_roi_vein_': {'color': 'b', 'legend': 'Venous Peak Phase',
                                     'fontdict': {'family': 'consolas', 'color': 'blue', 'weight': 'normal',
                                                  'size': 14},
                                     'loc': (.57, .8),
                                     },
        }
        self.line_color = self.phases[phase]['color']
        self.phase = phase
        self.mean_values, self.peak_value, self.min_value, self.peak_position = (None,) * 4
        self.get_stats(SUB)
        self.annotate = None

    def get_stats(self, SUB):
        """Compute min/max and argmin/argmax of ROI signal"""
        self.mean_values = np.array(
            [_SUB[self.roi > 0].mean() for _SUB in SUB[:, self.slice_idx - 1]])
        self.peak_value = self.mean_values.max()
        self.min_value = self.mean_values.min()
        self.peak_position = self.mean_values.argmax()

    def plot(self, ax):
        """Plot ROI signal through time"""
        ax.plot(np.arange(1, len(self.mean_values) + 1), self.mean_values, '.-%s' % (self.line_color,), linewidth=1)
        ax.get_lines()[-1].set_label(self.phase)  # name each line
        self.adjust_axis(ax)
        # self.set_aspect(ax)
        # ax.set_aspect('auto')

    def set_aspect(self, ax, ratio=(3 / 5)):
        y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
        aspect = min((x_len * ratio) / y_len, 1 / ((x_len * ratio) / y_len))
        ax.set_aspect(aspect)

    def update(self, roi, SUB, ax, slice_idx):
        """Update the current plot when both phases exist"""
        self.slice_idx = slice_idx
        self.roi = roi
        self.get_stats(SUB)
        for line in ax.get_lines():
            if line.get_label() == self.phase:
                line.set_ydata(self.mean_values)
        self.adjust_axis(ax)

        # ax.set_aspect('auto')

    def adjust_axis(self, ax):
        """Adjust axis min/max and show legend"""
        ax.set_xlim([0, self.ax_length])
        ax_all_ymin = []
        ax_all_ymax = []
        for line in ax.get_lines():
            if line.get_label() in self.phases.keys():
                ax_all_ymin.append(line.get_ydata().min())
                ax_all_ymax.append(line.get_ydata().max())
        ax_ymin = min(ax_all_ymin)
        ax_ymax = max(ax_all_ymax)
        ax.set_ylim([ax_ymin - abs(ax_ymin) * 0.1, ax_ymax + abs(ax_ymax) * 0.1])
        if self.annotate:
            self.annotate.remove()
        self.annotate = ax.annotate('',
                                    xy=(self.peak_position + .5, self.peak_value),
                                    xytext=(self.peak_position + 3, self.peak_value),
                                    arrowprops=dict(edgecolor=self.phases[self.phase]['color'], arrowstyle='->',
                                                    shrinkA=0.01),
                                    )


class DscROI(ROI):
    """Handle ROI data and plots"""

    def __init__(self, roi, slice_idx, SUB, phase):
        super().__init__(roi, slice_idx, SUB, phase, 'dsc')
        self.ax_length = SUB.shape[0] + 10


class DceROI(ROI):
    """Handle ROI data and plots"""

    def __init__(self, roi, slice_idx, SUB, phase):
        super().__init__(roi, slice_idx, SUB, phase, 'dce')
        self.ax_length = SUB.shape[0] + 5


def get_roi(prefix):
    if prefix == 'dce':
        return DceROI
    return DscROI
