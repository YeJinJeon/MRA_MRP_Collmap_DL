from constants import *
import torchvision.transforms as transforms
from data_augmentation.center_crop import CenterCrop
from data_augmentation.horizontal_flip import HorizontalFlip
from training_config.base_dce_mra_config import BaseDceMraConfig
import torch


class _Config(BaseDceMraConfig):

    def __init__(self, old_root_dir="/media/yejin/새 볼륨/mra/BrainMRA_Nov2021_Anonymized",
                 new_root_dir="/data1/yejin/mra_npy",
                 csv_file="./dataset/split_dce_mra_dataset.csv"):
        super(_Config, self).__init__(old_root_dir, new_root_dir, csv_file)
        self.model_parallel = True
        self.pretrain = '/data1/yejin/compu/3d_unet++/pretrain/UnetPlusPlus_1AveragePhaseLoss_Adam_inputs_labels_labels_weight_for_each_phase_mask_2022-05-02-21-12-04_seed42/model_92.pth'
        self.summary_writer_folder_dir = "/data1/yejin/compu/_unet++/runs"
        self.log_file_dir = "/data1/yejin/compu/3d_unet++/log_file"
        self.model_save_dir = "/data1/yejin/compu/3d_unet++/pretrain"
        self.predict_save_dir = "/data1/yejin/compu/3d_unet++/result"
        self.evaluation_metrics_save_dir = "/data1/yejin/compu/3d_unet++/evaluation_metrics"

        self.network_architecture = {
            "file": "network_architecture/unetplusplus_3d/unet_plus_plus",
            "parameters": {
                "input_channels": 40,
                "output_channels": 5,
                "dev_0": torch.device("cuda:3"),
                "dev_1": torch.device("cuda:3"),
            }
        }
        self.loss = {
            "loss1": {
                "file": "criteria/average_phase_loss",
            }
        }
        setattr(self, DatasetTypeString.TRAIN, {
            "dataset": {
                "file": "dce_mra_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.TRAIN,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_rs_intensity_n.npy',
                                   'labels_weight_for_each_phase': 'phase_maps_rs_intensity_n_wm_50_bins_for_each_phase.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 16
            }
        })


config = _Config().__dict__