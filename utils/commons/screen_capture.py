from tools import PySimpleGUI as sg
from PIL import ImageGrab
import pylab as plt
import numpy as np
from utils.dce_mra.load_dicom import CropImage


class CropScreen(CropImage):
    """Interactive Crop the screen capture by ImageGrab"""

    def __init__(self, save_dir=None):
        super(CropScreen, self).__init__()
        self.cropped_img = None
        self.fignum = 'Screen-shot'
        self.save_dir = save_dir

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def comp_fig_ratio(_img):
        return _img.shape[0] / _img.shape[1]

    def show_image(self, _img, sz=20):
        fig_ratio = self.comp_fig_ratio(_img)
        fig, ax = plt.subplots(1, 1, num=f'Screen-shot', figsize=(sz, sz * fig_ratio))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(_img)
        ax.axis('off')
        return fig, ax

    def _crop_image(self, _img=None, fs_sz=20):
        """"""
        _img = np.array(ImageGrab.grab(all_screens=True)) if _img is None else _img
        fig, ax = self.show_image(_img, sz=fs_sz)
        fig.canvas.mpl_connect('button_press_event', self.toggle_selector)
        # win = plt.gcf().canvas.manager.window
        # win.overrideredirect(1)
        self.get_selector(ax)
        self.pause()
        # plt.pause(1e-3)
        plt.close(fig)

        if (self.ec is not None) and (self.er is not None):
            self.cropped_img = _img[self.ec[1]:self.er[1], self.ec[0]:self.er[0]]

        if (self.save_dir is not None) and (self.cropped_img is not None):
            plt.imsave(f'{self.save_dir}/screen-shot.png', self.cropped_img)

        self.pressed = False  # Reset state

    def crop_image(self, _img=None, fs_sz=20):
        """A wrapper function (also a walk-around function) to stop the code from hanging when the user close the figure by click x-button"""
        layout = [[sg.T('')]]
        w_dummy = sg.Window('_dummy_', layout, keep_on_top=False, alpha_channel=0)
        w_dummy.read(0)
        self._crop_image(fs_sz=20)
        w_dummy.close()

    def show_cropped_image(self, _img=None, sz=15):
        self.cropped_img = _img if _img is not None else self.cropped_img
        if self.cropped_img is None:
            return
        self.show_image(self.cropped_img, sz=sz)


if __name__ == "__main__":
    cr = CropScreen(save_dir=r'C:\Users\Minh\Documents/')
    cr.crop_image(fs_sz=20)
    cr.show_cropped_image(sz=12)
    plt.show()
