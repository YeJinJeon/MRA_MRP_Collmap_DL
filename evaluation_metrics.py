import numpy as np
import torch
import os
import glob
from sklearn.metrics import mean_absolute_error as MAE
import pathlib
from writer import *
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from skimage.util import montage
from gen_dicoms_noHeader import mr_collateral_dce_gen_dicoms
    
def R_Squared(pred, gt):
    """https://github.com/pytorch/pytorch/issues/1254 참고해서 고치기"""
    pred_diff = pred - pred.mean() #vector - scalar = vector.shape
    gt_diff = gt - gt.mean()
    #sum_mul_diff = torch.sum(pred_diff * gt_diff) #vector * vector = vector
    r_squared = torch.sum(pred_diff * gt_diff) / torch.sqrt(torch.sum(pred_diff**2) * torch.sum(gt_diff**2))
    return r_squared.item()

def TM(pred, gt):
    """input : vector"""
    euclidian_dist = torch.dist(pred, gt) #scalar
    pred = pred.view(1, -1)
    gt = pred.view(-1, 1)
    product = torch.matmul(pred, gt) #scalar
    tm = product / (product + euclidian_dist ** 2)
    return tm.item()

def SSIM(pred, gt):
    """input : vector"""
    #dynamic range of pred and truth: 20 * log10(max-min)
    #dr = 20*torch.log10(torch.max(pred.max(),gt.max()) - torch.min(pred.min(), gt.min()))
    dr = 1.8
    c1 = (0.01*dr)**2
    c2 = (0.03*dr)**2
    cov = torch.sum((pred - pred.mean()) * (gt - gt.mean())) / pred.view(-1, 1).shape[0]

    ssim = ((2*pred.mean()*gt.mean()+c1) * (2*cov+c2)) / \
        ((pred.mean().pow(2) + gt.mean().pow(2) + c1) * (pred.var() + gt.var() + c2))
    return ssim.item()

def save_color_fig(pred, gt, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ind, phase in enumerate(['Art', 'Cap', 'EVen', 'LVen', 'Del']):
        fig, ax = plt.subplots(1, 2, num='result', figsize=(20, 10))
        fig.suptitle(phase)
        ax[0].imshow(montage(pred[ind][:-2, :, :], grid_shape=(5, 5), multichannel= True))
        ax[1].imshow(montage(gt[ind][:-2, :, :], grid_shape=(5, 5), multichannel= True))
        plt.figure('result')
        plt.savefig(f'{save_path}/result_{phase}.png', dpi=150)
        plt.close('result')


class Metrics_Evaluation:

    def __init__(self, predict_save_dir, evaluation_metrics_save_dir, save_fig=False):
        self.predict_save_dir = predict_save_dir
        self.result_list = glob.glob(f"{self.predict_save_dir}/*/*/*Y/*")
        # self.result_list = glob.glob(f"{self.predict_save_dir}/CMC_DATA/Abnormal_No597/2019Y/20190126_KU_6310333")
        self.evaluation_metrics_save_dir = evaluation_metrics_save_dir
        pathlib.Path(evaluation_metrics_save_dir).mkdir(parents=True, exist_ok=True)
        self.writers = [DscMrpCsvWriter(), DscMrpExcelWriter()]
        self.save_fig = save_fig

    def evaluate(self):
        r_squared_list = []
        MAE_list = []
        TM_list = []
        SSIM_list = []
        for result_dir in self.result_list:
            patient_info = '/'.join(result_dir.split('/')[-4:]) # CMC_DATA/Abnormal_No597/2019Y/20190126_KU_631033
            print(patient_info)
            try:
                r_squared_s = []
                MAEs = []
                TMs = []
                SSIMs = []
                pred_array_list = []
                label_array_list = []

                _mask = np.load((result_dir + "/" + "mask.npy"))
                mask = torch.from_numpy(_mask)
                
                for i in range(5):
                    _pred = np.load(result_dir + "/" + "predict.npy")[i]
                    _label = np.load( result_dir + "/" + "gt.npy")[i]
                    pred_array_list.append(_pred)
                    label_array_list.append(_label)
                    # calculate evaluation metrics
                    pre_predict = torch.from_numpy(_pred).type(torch.FloatTensor)[mask[0] > 0].numpy()
                    pre_label = torch.from_numpy(_label).type(torch.FloatTensor)[mask[0] > 0].numpy()
                    predict = (pre_predict - min(pre_predict)) / (max(pre_predict) - min(pre_predict))
                    label = (pre_label - min(pre_label)) / (max(pre_label) - min(pre_label))
                    r_squared_s.append(R_Squared(torch.from_numpy(predict), torch.from_numpy(label)))
                    MAEs.append(MAE(torch.from_numpy(label), torch.from_numpy(predict)))
                    TMs.append(TM(torch.from_numpy(predict), torch.from_numpy(label)))
                    SSIMs.append(SSIM(torch.from_numpy(predict), torch.from_numpy(label)))
                r_squared_list.append(r_squared_s)
                MAE_list.append(MAEs)
                TM_list.append(TMs)
                SSIM_list.append(SSIMs)
                # save color collateral maps
                if self.save_fig == True:
                    pred_array = np.array(pred_array_list)
                    # label_array = np.array(label_array_list)
                    # color_gt = mr_collateral_dce_gen_dicoms(os.path.join(self.evaluation_metrics_save_dir, patient_info), None, label_array, _mask, suffix='', rescaling_first=False, not_dl=True)
                    color_pred = mr_collateral_dce_gen_dicoms(os.path.join(self.evaluation_metrics_save_dir, patient_info), None, pred_array, _mask, suffix='', rescaling_first=True, not_dl=False)
                    # save_color_fig(color_pred, color_gt, os.path.join(self.evaluation_metrics_save_dir, patient_info))
            except Exception as e:
                print(result_dir)
                print(e)
                continue
        data = {"R-Squared": r_squared_list, "MAE": MAE_list, "TM": TM_list, "SSIM": SSIM_list}
        r_squared_arr = np.array(r_squared_list)
        mae_arr = np.array(MAE_list)
        tm_arr = np.array(TM_list)
        ssim_arr = np.array(SSIM_list)
        np.save(f"{self.evaluation_metrics_save_dir}/r_squared.npy", r_squared_arr)
        np.save(f"{self.evaluation_metrics_save_dir}/mae.npy", mae_arr)
        np.save(f"{self.evaluation_metrics_save_dir}/tm.npy", tm_arr)
        np.save(f"{self.evaluation_metrics_save_dir}/ssim.npy", ssim_arr)
        for writer in self.writers:
            if type(writer) is DscMrpCsvWriter:
                cf = {"data": data, "output_file": f"{self.evaluation_metrics_save_dir}/result.txt"}
            elif type(writer) is DscMrpExcelWriter:
                cf = {"data": data, "output_file": f"{self.evaluation_metrics_save_dir}/result.xlsx"}
            writer.write(cf)

if __name__ == '__main__':

    import argparse
    from datetime import datetime

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--root_save_dir", type=str, required=True) 
    parser.add_argument('-p', "--predict_save_dir", type=str, required=True) 
    parser.add_argument('-s', "--save", type=bool, default=False)
    args = vars(parser.parse_args())
    predict_folder = args['predict_save_dir'].split("/")[-1]

    # save directory 
    # root_save_dir = "/data1/yejin/compu/mra_v2/3d_unet_tgm_trm_without_sliceattn/result"
    # predict_save_dir = "/data1/yejin/compu/mra_v2/3d_unet_tgm_trm_without_sliceattn/result/model_224_2022-08-03-12-10-17_seed42" #################################
    # predict_folder = predict_save_dir.split("/")[-1]

    # evaluate
    evaluation_metrics_save_dir = os.path.join(args['root_save_dir'], 'evaluation_metrics', predict_folder)
    evaluation_metrics = Metrics_Evaluation(args['predict_save_dir'], evaluation_metrics_save_dir, args['save'])
    evaluation_metrics.evaluate()