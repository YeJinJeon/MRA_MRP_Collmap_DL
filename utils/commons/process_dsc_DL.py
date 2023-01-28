import numpy as np
import pylab as plt
from glob import glob
from os.path import abspath, join
from tools import PySimpleGUI as sg
from utils.commons import misc
from utils.commons.review_color_phase_maps import ReviewColorPhaseMaps
import utils.dsc_mrp.load_npy as load_npy
from utils.dsc_mrp.load_dicom import load
from utils.commons.deep_learning_prediction import Model
from utils.commons.gen_dicoms import mr_collateral_dsc_gen_dicom


class DSC_MRP_DL:
    """"""

    def __init__(self):
        """"""
        print('Creating the deep learning model...')
        self.model = Model()
        self.npy_folder, self.IMG, self.mask, self.hdr, self.current_pm, self.rpm = (None,) * 6
        self.w_rpm_active = False
        print('Done!')

    def reset_phase_maps_reviewer(self):
        """"""
        self.w_rpm_active = False
        if self.current_pm is not None:
            self.current_pm = None  # Delete the phase maps generate by '_dsc_gen_dicoms_'
            if self.rpm is not None:
                self.rpm.w.close()
                self.rpm = None
        # Load the current_pm in the new folder (if exists)
        pth = join(f'{self.npy_folder}', 'pm_DL.npy')
        if glob(pth):
            self.current_pm = np.load(pth)

    def process(self, event, values, window):
        """

        :param window:
        :param event:
        :param values:
        :return:
        """
        if (event == '_dsc_open_dicom_dl_') and (values['_dsc_open_dicom_dl_'] != ''):
            self.rpm.w.close() if self.rpm is not None else None
            self.IMG, self.mask, self.hdr, tmp_npy_folder = load(values['_dsc_open_dicom_dl_'])
            if self.IMG is not None:
                self.npy_folder = tmp_npy_folder
            self.reset_phase_maps_reviewer()
            misc.reset_bbit(window[event])

        if (event == '_dsc_open_npy_dl_') and (values['_dsc_open_npy_dl_'] != ''):
            self.IMG, self.mask, self.hdr, tmp_npy_folder = load_npy.load_npy(values['_dsc_open_npy_dl_'])
            print(f'Loading Numpy Files in {tmp_npy_folder}...')
            if self.IMG is not None:
                self.npy_folder = tmp_npy_folder
                print('Done')
            else:
                ms = 'Numpy files not found!'
                print(ms)
                sg.Popup(ms, keep_on_top=True, title='Message')
            self.reset_phase_maps_reviewer()
            misc.reset_bbit(window[event])

        if event == '_gci_dl_':
            if self.IMG is None:
                sg.Popup('Please load images first!', keep_on_top=True, title='Message')
            else:
                print('Predicting collateral...')
                pred, self.mask = self.model.predict(self.IMG, mask=self.mask, folder=self.npy_folder)

                print('Generate DICOM files...')
                save_dir_path = abspath('%s/../DSC_DRNN/' % self.npy_folder)
                self.reset_phase_maps_reviewer()
                self.current_pm = mr_collateral_dsc_gen_dicom(save_dir_path, self.hdr, pred, self.mask, suffix='',
                                                              rescaling_first=True, from_deep_learning=True)
            np.save(join(f'{self.npy_folder}', 'pm_DL.npy'), self.current_pm)

        if event == '_review_phase_maps_':
            if self.current_pm is None:
                sg.PopupError('Please generate the phase maps first!', keep_on_top=True, title='Message')
                return

            self.rpm = ReviewColorPhaseMaps(self.current_pm, 'DSC-DL')
            self.w_rpm_active = True
            [plt.close(fig) for fig in self.rpm.figs]
