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

def load_phase_map(base_input, prefix, save_npy=False):

    print('\n\nLoading Phasemap files from %s\nPlease wait...' % base_input)
    tic0 = time.time()
    art_folder = []
    cap_folder = []
    even_folder= []
    lven_folder = []
    del_folder = []

    if prefix == "dce":
        saved_phase_vars = ['art', 'cap', 'even', 'lven', 'delay']
        saved_property_vars= ['tck', 'dis', 'nos', 'sdi', 'rot_angle', 'r_slices']
        dir_phase_map = glob.glob(base_input + '/SF*')
        # if there are more than one collateral map directory - choose max number of slices(nos)
        if len(dir_phase_map) > 1:
            nos_list = [int(dir.split('/')[-1].split('_')[-3][-2:]) for dir in dir_phase_map]
            idx = np.argmax(nos_list)
            dir_phase_map = dir_phase_map[idx]
            nos = nos_list[idx]
        else:
            dir_phase_map = glob.glob(base_input + '/SF*')[0]
            nos = int(dir_phase_map.split('/')[-1].split('_')[-3][-2:])
    else:  # dsc
        saved_vars = ['art', 'cap', 'even', 'lven', 'delay']
        dir_phase_map = glob.glob(base_input + '/DSC_*')[0]
        nos = len(glob.glob(f"{dir_phase_map}/Gray_0_Art_DSC_0*.dcm"))

    def art_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "dce":
            return glob.glob(f"{dir_phase_map}/*_Col_Gr_0_Art_DCE_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_0_Art_DSC_0{i}.dcm")[0]
    def cap_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "dce":
            return glob.glob(f"{dir_phase_map}/*_Col_Gr_1_Cap_DCE_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_1_Cap_DSC_0{i}.dcm")[0]
    def even_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "dce":
            return glob.glob(f"{dir_phase_map}/*_Col_Gr_2_EVen_DCE_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_2_EVen_DSC_0{i}.dcm")[0]
    def lven_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "dce":
            return glob.glob(f"{dir_phase_map}/*_Col_Gr_3_LVen_DCE_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_3_LVen_DSC_0{i}.dcm")[0]
    def del_base(i, prefix):
        if i < 10:
            i = f"0{i}"
        if prefix == "dce":
            return glob.glob(f"{dir_phase_map}/*_Col_Gr_4_Del_DCE_0{i}.dcm")[0]
        else:
            return glob.glob(f"{dir_phase_map}/Gray_4_Del_DSC_0{i}.dcm")[0]
 
    for i in range(nos): ######################################################
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

    # get slice info
    if prefix == "dce":
        slice_info = dir_phase_map.split('/')[-1].split('_')
        rot_angle = int(slice_info[-1][3:])
        sdi = int(slice_info[-2][2:])
        r_slices = int(slice_info[-3][3:])
        dis = int(slice_info[-4][3:])
        tck = int(slice_info[-5][3:])
        nos = int(art.shape[2])
        
    phase_maps = [art, cap, even, lven, delay]
    phase_properties = [tck, dis, r_slices, sdi, rot_angle, nos]
 
    dir_npy = '%s/NpyFiles' % (base_input)
    if save_npy:
        for var in saved_vars:
            np.save(f"{dir_npy}/{var}.npy", eval(var))
    message = 'Done! Total time: %.2f s' % (time.time() - tic0)
    print(message)

    return phase_maps, phase_properties

if __name__ == "__main__":

    base_input = "/media/yejin/새 볼륨/mra/BrainMRA_Nov2021_Anonymized/KU_DATA/Normal/2016Y/20160625_KU_80086955/PWI_DSC_Collateral_py"
    
    phase_map, scan, origin, spacing, itkimage = load_phase_map(base_input, "dsc")
    print()
