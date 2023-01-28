import os

import numpy as np

from utils.commons.misc import s_mt, sc_mt

# View-Axis dictionary for 3D image
axes_dict = {
    'coronal': [0, 1, 2],
    'axial': [1, 0, 2],
    'sagittal': [2, 1, 0],
}


class Image:
    """An abstract class for Image4D and Image3D"""

    def __init__(self, arr=None, dir_in=None):
        """

        :param arr: 4D or 3D Numpy array in coronal view
        :param dir_in: a string indicates the Numpy folder containing the Numpy files
        """
        self.dir_in = dir_in
        self._arr = arr
        self._mask = None
        self._view = 'coronal'
        self._axis = [0, 1, 2]
        self._class_prefix = ''

    class _Decorators:
        @classmethod
        def match_view(cls, decorated):
            """Match the view of array and mask"""

            def wrapper(cls_img, array, **kwargs):
                array = np.array(array, dtype='uint8') if isinstance(array, list) else array
                start_dim = 1 if len(cls_img.shape) == 4 else 0
                if array.shape != cls_img.shape[start_dim:]:
                    if cls_img.ndim > len(array.shape):
                        axis = [ax - 1 for ax in cls_img.axis[1:]]
                    else:
                        axis = [ax for ax in cls_img.axis]
                    array = array.transpose(axis)
                decorated(cls_img, array)

            return wrapper

        @classmethod
        def change_view(cls, decorated):
            """Reset to coronal view then perform transpose for the target view"""

            def wrapper(cls_img, **kwargs):
                cls_img.transpose()  # Reverse to the default view, aka 'coronal'
                cls_img.view = decorated.__name__
                if cls_img.view != 'coronal':
                    cls_img.transpose()

            return wrapper

        @classmethod
        def update_axis_info(cls, set_view):
            """Update axis list whenever the view changes"""

            def wrapper(cls_img, view, **kwargs):
                set_view(cls_img, view)
                axis = axes_dict[cls_img.view]
                axis = [0, ] + [ax + 1 for ax in axis] if cls_img.ndim == 4 else axis
                cls_img.axis = axis

            return wrapper

    def load_npy(self, filename):
        full_filename = f'{self.dir_in}/{filename}.npy'
        if os.path.isfile(full_filename):
            return np.load(full_filename)
        else:
            raise FileNotFoundError(f'{filename} was not created.')

    def transpose(self):
        """
        Transpose arr and mask (if available)
        """
        self._arr = self._arr.transpose(self._axis)
        if self._mask is not None:
            self.mask = self._mask  # match view

    @_Decorators.change_view
    def axial(self):
        """Transpose the array and mask so that they are in axial view"""

    @_Decorators.change_view
    def sagittal(self):
        """Transpose the array and mask so that they are in sagittal view"""

    @_Decorators.change_view
    def coronal(self):
        """Transpose the array and mask so that they are in coronal view"""

    @property
    def view(self):
        return self._view

    @view.setter
    @_Decorators.update_axis_info
    def view(self, view):
        self._view = view

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, array=None):
        if not isinstance(array, np.ndarray):
            raise TypeError('Array type must be Numpy Array')
        else:
            self._arr = array

    @arr.getter
    def arr(self):
        if self._arr is None:
            return self.load_npy('IMG')
        return self._arr

    @property
    def min(self):
        return self._arr.min(axis=0)

    @property
    def max(self):
        return self._arr.max(axis=0)

    @property
    def mean(self):
        return self._arr.mean(axis=0)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    @_Decorators.match_view
    def mask(self, array=None):
        if not isinstance(array, np.ndarray):
            raise TypeError('Mask type must be Numpy Array')
        else:
            self._mask = array

    @mask.getter
    def mask(self):
        if self._mask is None:
            return self.load_npy('mask')
        return self._mask

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = axis

    @axis.getter
    def axis(self):
        return self._axis

    def show(self, cm='gray', show_now=False, v_range: tuple = (None, None)):
        s_mt(self._arr, cm=cm, to_show=show_now, v_range=v_range)

    def show_contour(self, show_now=False):
        if self._mask is not None:
            sc_mt(self._mask, to_show=show_now)
        else:
            raise NotImplementedError

    def remove_background(self):
        self._arr *= self.mask

    def __str__(self):
        return f'Image{self._class_prefix}'

    def __repr__(self):
        return f'Image{self._class_prefix} instance'


class ImageClass(Image):
    """"""


class Image3D(Image):
    """A class for handling 3D images"""

    def __init__(self, arr, dir_in=None):
        super(Image3D, self).__init__(arr, dir_in)
        self._class_prefix = '3D'


class Image4D(Image):
    """A class for handling 4D MRA and DSC images"""

    def __init__(self, arr, dir_in=None):
        super(Image4D, self).__init__(arr, dir_in)
        self._class_prefix = '4D'
        self._axis = [0, 1, 2, 3]

    @property
    def sub(self):
        return self._arr - self._arr[0]
