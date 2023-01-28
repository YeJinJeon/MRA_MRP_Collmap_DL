import torch
import numpy as np
import pathlib
import os
import gc
from constants import ModelMode
import matplotlib.pyplot as plt
from skimage.util import montage
import csv
# import torchvision.transforms as transforms
# from data_augmentation.center_crop import CenterCrop
from gen_dicoms_noHeader import mr_collateral_dsc_gen_dicoms
from utils.dsc_mrp.load_dicom import load_dsc
from utils.dsc_mrp.load_map import load_phase_map_dsc
from utils.dsc_mrp.preprocess import preprocess_dsc

def save_color_fig(pred, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ind, phase in enumerate(['Art', 'Cap', 'EVen', 'LVen', 'Del']):
        fig, ax = plt.subplots(1, 1, num='result', figsize=(20, 10))
        fig.suptitle(phase)
        ax.imshow(montage(pred[ind][:-2, :, :], grid_shape=(5, 5), multichannel= True))
        plt.figure('result')
        plt.savefig(f'{save_path}/result_{phase}.png', dpi=150)
        plt.close('result')

def findfile(model_path, model_ckpt):
    pretrain_path = model_path + '/pretrain'
    pretrain_dirs = os.listdir(pretrain_path)
    for dirname in pretrain_dirs:
        files = os.listdir(os.path.join(pretrain_path, dirname))
        if model_ckpt in files: 
            return os.path.join(pretrain_path, dirname, model_ckpt)

def get_file_names(dataset_file):
    data_dirs = []
    with open(dataset_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            data_dirs.append(row[0])
    return data_dirs
    
class Predict:

    def __init__(self, network_architecture, best_model_dir, test_generator, predict_save_dir, device,
                 model_parallel=False, model_mode=ModelMode.ThreeDimensions):
        if model_parallel:
            self.network_architecture = network_architecture
        else:
            self.network_architecture = network_architecture.to(device)
        self.checkpoint = torch.load(best_model_dir, map_location=device)
        self.network_architecture.load_state_dict(self.checkpoint['model_state_dict'])
        self.data_dirs = get_file_names("./dataset/dasan_dsc_data.csv")
        self.predict_save_dir = predict_save_dir
        self.model_mode = model_mode
        pathlib.Path(self.predict_save_dir).mkdir(parents=True, exist_ok=True)
        self.device = device

    def predict(self):
        print(self.predict_save_dir)

        # for index, (inputs, labels, masks, filenames, hdrs) in enumerate(self.test_generator):
        #     # folder = '/'.join(filenames[0].split('/')[-7:-3])
        #     folder = filenames[0].split('/')[-2]
        #     print(folder)
        for index, datapath in enumerate(self.data_dirs):
            folder = datapath.split('/')[-2]
            phasemap_folder = datapath.split("/")[-1]
            map_version = "new" if phasemap_folder in ["PWI_DSC_Collateral_py", "PWI_source_DSC_Collateral_py"] else "old"
            imgs, masks, hdrs = load_dsc(datapath)
            phases = load_phase_map_dsc(datapath, map_version)
            image, label, mask = preprocess_dsc(imgs, phases, masks)
            
            inputs = torch.tensor(image, dtype=torch.float64)[None, :]
            labels = torch.tensor(label, dtype=torch.float64)[None, :]
            masks = torch.tensor(mask)[None, :]
           
            inputs, labels, mask = inputs.to(self.device), \
                                    labels.to(self.device), \
                                    masks.to(self.device)
            self.network_architecture.eval()
            reg_predicts, _ = self.network_architecture(inputs) #[1, 5, 27, 224, 224]
            del inputs

            directory = f'{self.predict_save_dir}/{folder}'
            if not os.path.isdir(directory):
                os.makedirs(directory)
            _pred_array = torch.mul(reg_predicts, mask)[0].cpu().detach().numpy()
            _mask = mask[0].cpu().detach().numpy()
            color_pred = mr_collateral_dsc_gen_dicoms(directory, hdrs, _pred_array, _mask, suffix='DL', rescaling_first=True, insert_bar=True, from_deep_learning=True) # slice norm
            save_color_fig(color_pred, directory)
            
            # np.save(f'{self.predict_save_dir}/{folder}/gt.npy', torch.mul(labels, mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
            del labels
            # np.save(f'{self.predict_save_dir}/{folder}/predict.npy',
            #         torch.mul(reg_predicts, mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
            del reg_predicts
            # np.save(f'{self.predict_save_dir}/{folder}/mask.npy', mask[0].cpu().detach().numpy())
            del mask

            gc.collect()
            torch.cuda.empty_cache()
            
if __name__ == '__main__':

    import argparse
    from mra_dataset import MraDataset, MraLoader
    from torch.utils import data
    from network_architecture.efficientnet_backbone_drnn_f2_loss_only import EfficientnetBackboneDrnnF2LossOnly
    from network_architecture.efficientnet_backbone_drnn_sdd_loss import EfficientnetBackboneDrnnSddLoss
    from network_architecture.original_drnn import OriginalDrnn
    from network_architecture.unet_3d.u_net import UNet
    from network_architecture.unetplusplus_3d.unet_plus_plus import UnetPlusPlus
    from network_architecture.unet_backbone_tgm_trm import UnetBackboneTgmTrm
    from evaluation_metrics import Metrics_Evaluation
    from constants import DatasetType
    from datetime import datetime

    # predict
    root_dir = "/media/yejin/새 볼륨/compu/mrp_v3/mrod_phasewise"

    ### DATA
    csv_file = "./dataset/dasan_dsc_data.csv"

    generator_config = {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 4
    }
    test_dataset = MraDataset(csv_file, "dsc")
    test_generator = MraLoader(test_dataset, **generator_config)

    ### MODEL
    model_parallel = True
    device_0 = torch.device(f"cuda:{0}")
    device_1 = torch.device(f"cuda:{1}")
    model_name = root_dir.split('/')[-1]
    if "mrod" in model_name:
        net = EfficientnetBackboneDrnnSddLoss(40, 12, 0.5, 15, device_0, device_0, device_1, device_1, "efficientnet-b0")
    elif model_name == "drnn":
        net = OriginalDrnn(40, 4, 0.5, device_0)

    root_save_dir = root_dir + "/result"
    model_checkpoint = "model_192.pth"
    best_model_dir = findfile(root_dir, model_checkpoint)
    file_name = model_checkpoint.split(".")[0] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    predict_save_dir = os.path.join(root_save_dir, file_name)
    predict_save_dir = predict_save_dir + f"_seed{42}"
    predict = Predict(net, best_model_dir, test_generator,
                        predict_save_dir, device_0, model_parallel)
    predict.predict()
    print("Done Prediction")

    # # evaluate
    # predict_folder = predict_save_dir.split("/")[-1]
    # evaluation_metrics_save_dir = os.path.join(root_save_dir, 'evaluation_metrics', predict_folder)
    # evaluation_metrics = Metrics_Evaluation(predict_save_dir, evaluation_metrics_save_dir, save_fig=True)
    # evaluation_metrics.evaluate()
    # print("Done Evaluation")

    # # construct the argument parser and parse the arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', "--model_dir", type=str, required=True) # /data1/yejin/compu/mra_v2/3d_unet"
    # parser.add_argument('-ckp', "--model_ckpt", type=str, required=True) # model_300.pth
    # parser.add_argument('-d', "--device", type=int, default=0)
    # parser.add_argument('-s', "--save", type=bool, default=False)
    # args = vars(parser.parse_args())

    # ### DATA
    # csv_file = "./dataset/split_dce_sample.csv"

    # generator_config = {
    #             "batch_size": 1,
    #             "shuffle": False,
    #             "num_workers": 4
    # }
    # test_dataset = DceMraDataset(csv_file)
    # test_generator = data.DataLoader(test_dataset, **generator_config)

    # ### MODEL
    # model_parallel = False
    # device = torch.device(f"cuda:{args['device']}")
    # model_name = args['model_dir'].split('/')[-1]
    # if model_name == "3d_unet":
    #     net = UNet(40, 5, device)
    # elif model_name == "3d_unet_tgm_trm":
    #     net = UnetBackboneTgmTrm(40, 5, 20, device, 64)
    # elif model_name == "3d_unet_tgm_trm_without_sliceattn":
    #     net = UnetBackboneTgmTrm(40, 5, 0, device, 64)
    # elif model_name == "mrod":
    #     net = EfficientnetBackboneDrnnF2LossOnly(40, 12, 0.5, 10, device, device, device, device, "efficientnet-b0")
    # elif model_name == "drnn":
    #     net = OriginalDrnn(40, 4, 0.5, device)
    # best_model_dir = findfile(args['model_dir'], args['model_ckpt'])

    # # predict
    # root_save_dir = args['model_dir'] + "/result"
    # checkpoint = best_model_dir.split("/")[-1][:-4]
    # file_name = checkpoint + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # predict_save_dir = os.path.join(root_save_dir, file_name)
    # predict_save_dir = predict_save_dir + f"_seed{42}"
    # predict = Predict(net, best_model_dir, test_generator,
    #                     predict_save_dir, device, model_parallel)
    # predict.predict()
    # print("Done Prediction")

    # # evaluate
    # predict_folder = predict_save_dir.split("/")[-1]
    # evaluation_metrics_save_dir = os.path.join(root_save_dir, 'evaluation_metrics', predict_folder)
    # evaluation_metrics = Metrics_Evaluation(predict_save_dir, evaluation_metrics_save_dir, args['save'])
    # evaluation_metrics.evaluate()
    # print("Done Evaluation")


