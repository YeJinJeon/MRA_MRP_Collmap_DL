import os
import time
import ctypes
import functools
from multiprocessing import cpu_count

import numpy as np
import pylab as plt
from skimage.util import montage
from scipy.ndimage import rotate
from skimage.transform import resize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

D_TYPE = 'float32'  # 'float64'
NUM_WORKERS = cpu_count() - 1
INTERPOLATE_ORDER = 1  # 3


def get_tray():
    tray = sg.SystemTray()
    return tray


def get_window_size():
    from layouts.phase_map_layout_with_canvas_v4 import SCREEN_SIZE_DEFAULT
    window_size = sg.Window.get_screen_size() if os.name == 'nt' else SCREEN_SIZE_DEFAULT
    return window_size


def set_dpi_awareness():
    # Query DPI Awareness (Windows 10 and 8)
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))

    # Set DPI Awareness  (Windows 10 and 8)
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    # the argument is the awareness level, which can be 0, 1 or 2:
    # for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)
    return errorCode


def estimate_center_location(window=None, window_size_ratio=.5, width=None, height=None, return_size=False):
    """

    :param height:
    :param width:
    :param return_size:
    :param window:
    :param window_size_ratio: size ratio versus the main window
    :return:
    """
    if os.name == 'nt':
        current_location = window.CurrentLocation()
        screen_size = window.get_screen_dimensions()
        w_size = [int(x * window_size_ratio) for x in screen_size] if window_size_ratio > 0 else (width, height)
        w_location = [
            screen_size[0] // 2 - w_size[0] // 2 + current_location[0],
            (screen_size[1] // 2 - w_size[1] // 2) // 2 + current_location[1],
        ]
    else:
        w_location = (None, None)
        w_size = (int(1700 * window.ratio[0]), int(800 * window.ratio[1]))
    if return_size:
        return w_location, w_size
    return w_location


def upscale(img: np.ndarray, scale=3):
    """Scale 3D/4D input image"""
    if img.ndim == 4:
        new_shape = (img.shape[0], img.shape[1] * scale, img.shape[2] * scale, img.shape[3])
    else:
        new_shape = (img.shape[0], img.shape[1] * scale, img.shape[2] * scale)
    return resize(img, new_shape, clip=False, preserve_range=True, anti_aliasing=False)


def calculate_window_size(window=None, ratio=.5):
    """

    :param window: main window
    :param ratio: size ratio versus the main window
    :return:
    """
    if os.name == 'nt':
        window_session_size = [int(x * ratio) for x in window.get_screen_dimensions()]
    else:
        window_session_size = (int(1700 * window.ratio), int(800 * window.ratio))
    return window_session_size


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.get_tk_widget().configure(bg='black')
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def savitzky_golay(y, window_size=11, order=1, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_wh(w, hw_ratio=.65):
    """Get width and height"""
    return w, w * hw_ratio


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def my_rotate(pm, ang, i):
    return rotate(pm[i], angle=ang, axes=(0, 1), order=INTERPOLATE_ORDER, reshape=False, prefilter=False)


def interpolate(img, new_size, i):
    new_img = resize(img[i], new_size, order=INTERPOLATE_ORDER, anti_aliasing=False, clip=False, preserve_range=True)
    img[i] = None
    return new_img


def _maximize():
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # works fine on Windows!


def show(maximize=False, exit_after=True):
    _maximize() if maximize else None
    plt.show()
    if exit_after:
        exit()


def s_mt(x: np.ndarray, cm='gray', to_show=False, v_range: tuple = (None, None)):
    plt.imshow(montage(x, multichannel=x.ndim == 4), cmap=cm, vmin=v_range[0], vmax=v_range[1])

    _maximize()
    if to_show:
        show()


def sc_mt(x, to_show=False, color='g'):
    plt.contour(montage(x), colors=color, linewidths=.3)
    if to_show:
        show()

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
