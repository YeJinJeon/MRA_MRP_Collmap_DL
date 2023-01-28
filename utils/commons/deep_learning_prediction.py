from tools import PySimpleGUI as sg
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from mxnet import nd, gpu
from networks import DRNN
import numpy as np
import time
import sys
import importlib


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def get_wait_layout():
    """"""
    layout = [
        [sg.T('Generate collateral using the deep learning model.')],
        [sg.T('Please wait...')],
    ]
    return layout


def crop_center(img, size_x, size_y, center=None, translate=None, get_center=False):
    """2D center crop for ND image
        translate: [translate_y, translate_x]
    """
    if translate is None:
        translate = [0, 0]
    if isinstance(translate, int):
        translate = [translate, translate]
    elif isinstance(translate, list) and translate.__len__() == 1:
        translate.append(translate[0])
    y, x = img.shape[-2:]
    # create new variable "new_center" to stop the program from changing the input variable
    if center is None:
        new_center = [y // 2, x // 2]
    else:
        new_center = [center[i] for i in range(len(center))]
    new_center[0], new_center[1] = new_center[0] + translate[0], new_center[1] + translate[1]
    start_x = new_center[1] - size_x // 2
    start_y = new_center[0] - size_y // 2
    if get_center:
        return img[..., start_y: start_y + size_y, start_x: start_x + size_x], new_center
    else:
        return img[..., start_y: start_y + size_y, start_x: start_x + size_x]


class Model:
    def __init__(self):
        """"""
        self.dir_model = resource_path('models')
        self.dir_model_alternative = os.path.abspath('models')
        self.resumed_epoch = 179
        self.gpu_id = 0
        self.num_fpg = 8
        self.kernel_size = 3
        self.num_stage = 4
        self.num_unit = 3
        self.units = [6, 12, 24, 48]
        self.init_channels = 8
        self.act_type = 'relu'
        self.use_batchnorm = False
        self.not_use_rescaled = False
        self.last_act_type = 'tanh'
        self.crop_size = 224
        self.num_classes = 5

        """Run prediction on the test set (without running training)"""
        self.ctx = gpu(self.gpu_id)
        # net_module = importlib.import_module('networks.' + self.network_name)
        net_opts = DRNN.Init(net_opts=self)
        self.net = DRNN.Network(net_opts)

        try:
            self.net.load_parameters('%s/drnn_DSC_ijcars.params' % (self.dir_model,), ctx=self.ctx)
        except:
            self.net.load_parameters('%s/drnn_DSC_ijcars.params' % (self.dir_model_alternative,), ctx=self.ctx)
        self.net.hybridize()

    def predict(self, x, mask=None, folder=None):
        w = sg.Window(title='Collateral Phase Map Prediction', layout=get_wait_layout())
        w.Read(timeout=10)
        tic = time.time()

        if (x.shape[-2] != self.crop_size) or (x.shape[-1] != self.crop_size):
            x, mask = crop_center(x, self.crop_size, self.crop_size), \
                      crop_center(mask, self.crop_size, self.crop_size)
        y = self.net(nd.array(x[np.newaxis], ctx=self.ctx)).asnumpy()[0]  # remove the first dimension (N=1)
        y = y * mask if mask is not None else y  # Mask the prediction if possible
        # Note: both mask and y are currently have 4 dimensions

        if folder:
            dir_out_file = '%s/../DSC_DRNN/NumpyFiles' % (folder, )
            if not os.path.exists(dir_out_file):
                os.makedirs(dir_out_file)
            filename = "%s/phase_map.npy" % (dir_out_file,)
            filename_mask = "%s/phase_map_mask.npy" % (dir_out_file,)
            np.save(filename, y)
            np.save(filename_mask, mask)

            message = 'Done!\n   Total time: %.2f\n   Files are saved in %s' % (time.time() - tic, dir_out_file)
            sg.PopupNoButtons(message, title='DSC-MRP -- Collateral Prediction',
                              non_blocking=True, auto_close=True, auto_close_duration=2)
            print(message)

        w.Close()
        return y, mask


# if __name__ == "__main__":
#     img = np.load(
#         r'D:\workspace\Copied_from_C\Workspace\BrainKUCMC\April2019\2019 CMC-contrast extra\Ingenia CX (Philips)-Contrast-Normal\BP402 2019-01-05\PWI_DSC_Collateral\MatFiles\IMG_n01.npy')
#     model = Model()
#     model.dir_model = '../models'
#     model.predict(img)
