from constants import *
import torchvision.transforms as transforms
from data_augmentation.center_crop import CenterCrop
from data_augmentation.horizontal_flip import HorizontalFlip
from training_config.base_dce_mra_config import BaseDceMraConfig
import torch


class _Config(BaseDceMraConfig):

    def __init__(self, old_root_dir="/data1/yejin/mra_npy_new",
                 new_root_dir="/data1/yejin/mra_npy_v3",
                 csv_file="./dataset/split_dce_mra_dataset.csv"):
        super(_Config, self).__init__(old_root_dir, new_root_dir, csv_file)

        self.log_file_dir = "/data1/yejin/compu/mra_v3/3d_unet_tgm_trm/log_file"
        self.model_save_dir = "/data1/yejin/compu/mra_v3/3d_unet_tgm_trm/pretrain"
        self.predict_save_dir = "/data1/yejin/compu/mra_v3/3d_unet_tgm_trm/result"
        self.evaluation_metrics_save_dir = "/data1/yejin/compu/mra_v3/3d_unet_tgm_trm/evaluation_metrics"
        self.summary_writer_folder_dir = "/data1/yejin/compu/mra_v3/3d_unet_tgm_trm/runs"

        # self.pretrain = '/data1/yejin/compu/mra_new/3d_unet/pretrain/UNet_1AveragePhaseLoss_Adam_inputs_labels_labels_weight_for_each_phase_mask_2022-06-24-22-34-05_seed42/model_100.pth'

        self.network_architecture = {
            "file": "network_architecture/unet_backbone_tgm_trm",
            "parameters": {
                "n_channels": 40,
                "n_classes": 5,
                "n_slices": 20, 
                "device": torch.device("cuda:3"),
                "rank" : 64
            }
        }
        self.loss = {
            "loss1": {
                "file": "criteria/average_phase_loss",
                "parameters": {
                    "device": torch.device("cuda:3")
                }
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
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 4
            }
        })
        setattr(self, DatasetTypeString.VAL, {
            "dataset": {
                "file": "dce_mra_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.VAL,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'labels_weight': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 4
            }
        })


config = _Config().__dict__
