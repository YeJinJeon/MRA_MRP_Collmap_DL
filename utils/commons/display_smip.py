import numpy as np
import pylab as plt
from matplotlib import ticker
from skimage.util import montage

font = {'family': 'consolas',
        'color': 'lightgreen',
        'weight': 'normal',
        'size': 10,
        }


def find_num_rows_cols(num_image, fig_size):
    # Solve the quadratic equation ax**2 + bx + c = 0
    # find two solutions
    prod = num_image
    div = fig_size[0] / fig_size[1]
    rows = np.sqrt(prod / div)
    cols = rows * div
    rows, cols = int(round(rows)), int(round(cols))
    while np.prod((rows, cols)) < num_image:
        if rows <= cols:
            cols += 1
        else:
            rows += 1
    return rows, cols


class Smip:
    """Subtraction maximum intensity projection"""

    def __init__(self, sub: np.ndarray, prefix='DCE', fig_sz=(float, float), perc_outlier=5):
        """

        :param sub: Subtraction image; 4D Numpy array (CDHW)
        :param prefix:
        :param fig_sz:
        :param perc_outlier: signal intensity outside the range (perc_outlier/2, 100-perc_outlier/2) will be cutoff.
        if perc_outlier is None, the default/current perc_outlier will be used to compute display_range
        """
        self.prefix = prefix
        self.fig_size = fig_sz
        self._cv_name = '_cv_smip_'
        self._perc_outlier = perc_outlier
        self._smip = self.compute_smip(sub)
        self.ax, self.fig, self._display_range = (None,) * 3

        self.create_fig()

    @property
    def cv_name(self):
        return f'_{self.prefix.lower()}' + self._cv_name

    @property
    def fig_name(self):
        suffix = f' with Outliers Elimination' if self.perc_outlier > 0 else ''
        return f'{self.prefix} - SMIP{suffix}'

    @property
    def shape(self):
        return self._smip.shape

    @property
    def smip(self):
        return self._smip

    @property
    def perc_outlier(self):
        return self._perc_outlier

    @property
    def display_range(self):
        return self._display_range

    @display_range.getter
    def display_range(self):
        if self._display_range is None:
            self._display_range = np.percentile(self._smip, (self.perc_outlier / 2, 100 - self.perc_outlier / 2))
        return self._display_range

    @display_range.setter
    def display_range(self, display_range: tuple = None):
        self._display_range = display_range

    @perc_outlier.setter
    def perc_outlier(self, perc_outlier=None):
        self._perc_outlier = perc_outlier if perc_outlier is not None else self._perc_outlier

    def compute_smip(self, sub: np.ndarray):
        """"""
        smip = sub.max(axis=1)
        if self.prefix.lower() == 'dsc':
            return smip
        smip = (smip - smip.min()) / smip.max()
        return smip

    def init_fig(self, shape: (int, int)):
        """Initialize figure"""
        h, w = shape
        self.fig_size = list(self.fig_size)
        if h < w:
            self.fig_size[1] = min(self.fig_size[1], self.fig_size[0] * (h / w))
        else:
            self.fig_size[0] = min(self.fig_size[0], self.fig_size[1] * (w / h))
        plt.figure(self.fig_name, frameon=False, facecolor='black', edgecolor='black',
                   figsize=(self.fig_size[0], self.fig_size[1]))

    def update_contrast(self, perc_outlier=None):
        """ Update the contrast of the Smip image according to the perc_outlier
        :param perc_outlier: signal intensity outside the range (perc_outlier/2, 100-perc_outlier/2) will be cutoff
        if perc_outlier is None, the default/current perc_outlier will be used to compute display_range
        """
        self.perc_outlier = perc_outlier
        for im in self.ax.get_images():
            im.set_clim(self.display_range)

    def create_fig(self):
        """Create a figure for the subtraction maximum intensity projection"""

        montage_row, montage_col = find_num_rows_cols(self.shape[0], self.fig_size)
        smip_montage = montage(self._smip, grid_shape=(montage_row, montage_col), fill=self._smip.min())

        # Initialize the figure
        self.init_fig(smip_montage.shape)

        # Display sMIP
        plt.imshow(smip_montage, cmap='gray', vmin=self._smip.min(), vmax=self._smip.max())
        plt.axis('off')

        # Display slice index on the the top-left corner of each image
        shape = self.shape
        [plt.text(10 + self.shape[-1] * j, 25 + shape[-2] * i,
                  int(i * montage_col + j) + 1 if ((i * montage_col + j) < shape[0]) else None, fontdict=font)
         for j in range(montage_col) for i in range(montage_col)]

        # Setup axis
        self.ax = plt.gca()
        self.ax.xaxis.set_major_locator(ticker.NullLocator())
        self.ax.yaxis.set_major_locator(ticker.NullLocator())
        plt.tight_layout(pad=0)
        self.update_contrast()
        self.fig = plt.gcf()


if __name__ == "__main__":
    pass
