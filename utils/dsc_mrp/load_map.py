from concurrent.futures import ThreadPoolExecutor
import os, glob
import time
import gc
import numpy as np
import pydicom
from pydicom import dcmread
import SimpleITK as sitk
import pylab as plt

def load_itk(_filename):
    
    _itkimage = sitk.ReadImage(_filename)
    scan = sitk.GetArrayFromImage(_itkimage)
    origin = np.array(list(reversed(_itkimage.GetOrigin())))
    spacing = np.array(list(reversed(_itkimage.GetSpacing())))
    
    return scan, origin, spacing,_itkimage

def load_phase_map_dsc(base_input, prefix, save_npy=False):

    art_folder = []
    cap_folder = []
    even_folder= []
    lven_folder = []
    del_folder = []

    
    saved_vars = ['art', 'cap', 'even', 'lven', 'delay']
    dir_phase_map = glob.glob(base_input + '/DSC_*')[0]

    if prefix == "old":
        nos = len(glob.glob(f"{dir_phase_map}/DSC_Collateral_Arterial_0*.dcm"))
        start_nos, end_nos = 1, nos+1
    else:
        nos = len(glob.glob(f"{dir_phase_map}/Gray_0_Art_DSC_0*.dcm"))
        start_nos, end_nos = 0, nos

    dir_npy = '%s/NpyFiles' % (base_input)
    if not os.path.exists(dir_npy):
        print("NpyFiles is not generated")
        return

    def art_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "old":
            return glob.glob(f"{dir_phase_map}/DSC_Collateral_Arterial_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_0_Art_DSC_0{i}.dcm")[0]
    def cap_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "old":
            return glob.glob(f"{dir_phase_map}/DSC_Collateral_Capillary_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_1_Cap_DSC_0{i}.dcm")[0]
    def even_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "old":
            return glob.glob(f"{dir_phase_map}/DSC_Collateral_Early_Venous_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_2_EVen_DSC_0{i}.dcm")[0]
    def lven_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "old":
            return glob.glob(f"{dir_phase_map}/DSC_Collateral_Late_Venous_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_3_LVen_DSC_0{i}.dcm")[0]
    def del_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "old":
            return glob.glob(f"{dir_phase_map}/DSC_Collateral_Delay_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_4_Del_DSC_0{i}.dcm")[0]
 
    for i in range(start_nos, end_nos): ######################################################
        art_folder.append(art_base(i, prefix))
        cap_folder.append(cap_base(i, prefix))
        even_folder.append(even_base(i, prefix))
        lven_folder.append(lven_base(i, prefix))
        del_folder.append(del_base(i, prefix))

    art_datasets = []
    cap_datasets = []
    even_datasets = []
    lven_datasets = []
    del_datasets = []
    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, art_folder)):
            art_datasets.append(result)
    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, cap_folder)):
            cap_datasets.append(result)
    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, even_folder)):
            even_datasets.append(result)
    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, lven_folder)):
            lven_datasets.append(result)
    with ThreadPoolExecutor() as executor:
        for (i, result) in enumerate(executor.map(dcmread, del_folder)):
            del_datasets.append(result)

    art = np.array([e.pixel_array for e in art_datasets])[np.newaxis]
    cap = np.array([e.pixel_array for e in cap_datasets])[np.newaxis]
    even = np.array([e.pixel_array for e in even_datasets])[np.newaxis]
    lven = np.array([e.pixel_array for e in lven_datasets])[np.newaxis]
    delay = np.array([e.pixel_array for e in del_datasets])[np.newaxis]

    phase_map = np.concatenate([art, cap, even, lven, delay])


    if save_npy:
        for var in saved_vars:
            np.save(f"{dir_npy}/{var}.npy", eval(var))

    return phase_map

if __name__ == "__main__":

    base_input = "/media/yejin/새 볼륨/mra/BrainMRA_Nov2021_Anonymized/KU_DATA/Normal/2016Y/20160625_KU_80086955/PWI_DSC_Collateral_py"
    
    phase_map, scan, origin, spacing, itkimage = load_phase_map_dsc(base_input, "dsc")
    print()
