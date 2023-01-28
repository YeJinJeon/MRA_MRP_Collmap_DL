import torch
from torch.utils import data
from torch.autograd import Variable
try:from torch.utils.tensorboard import SummaryWriter
except:from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.utils import import_file, convert_str_from_underscore_to_camel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import re
import numpy as np
import pathlib
import os
from dce_predict import Predict
from evaluation_metrics import Metrics_Evaluation
import os
import logging
from constants import *
import abc


class TrainingModelWrapper:

    def __init__(self, epochs=190, model_save_dir="", config_file="config.py",
                 train_summary_writer_folder="training_loss",
                 val_summary_writer_folder="validation_loss", seed=None, device=None):
        self.epochs = epochs
        self.model_save_dir = model_save_dir
        self.config_file = config_file
        self.train_summary_writer_folder = train_summary_writer_folder
        self.val_summary_writer_folder = val_summary_writer_folder
        self.seed = seed
        self.device = device
        self.config = import_file(config_file).config
        self._load_class_attributes(self.config)
        self.board_writer = SummaryWriter(f"{self.summary_writer_folder_dir}")
        self.logging = logging.basicConfig(filename=f"{self.log_file_dir}/log.txt", filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    def _init_fn(self):
        np.random.seed(int(1))

    def _load_class_attributes(self, config):
        for key, value in config.items():
            if key == "model_parallel":
                setattr(self, key, value)
            elif key == "network_architecture":
                self._load_network_architecture(value)
            elif key == "loss":
                self._load_loss(key, value)
            elif key == "val_loss":
                self._load_loss(key, value)
            elif key == "optimizer":
                self._load_optimizer(value)
            elif key == "lr_scheduler":
                self._load_lr_scheduler(value)
            elif key == DatasetTypeString.TRAIN or key == DatasetTypeString.VAL or key == DatasetTypeString.TEST:
                self._load_data_generator(key, value)
            #elif key == "device":
            #    self.device = torch.device(f'cuda: {self.device}')
            else:
                if "dir" in key:
                    file_name = self.generate_file_name() + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    value = os.path.join(value, file_name)
                    if self.seed is not None:
                        value = value + f"_seed{self.seed}"
                    else:
                        value = value + "_nonseed"
                    self._mkdir_not_exist(value)
                setattr(self, key, value)

    def _load_network_architecture(self, config):
        if self.model_parallel:
            self.network_architecture = self._load_nest_config(**config)
        else:
            self.network_architecture = self._load_nest_config(**config).to(self.device)
        if self.pretrain:
            self.checkpoint = torch.load(self.pretrain, map_location=self.device)
            self.network_architecture.load_state_dict(self.checkpoint['model_state_dict'])
            model_name = self.pretrain.split("/")[-1]
            self.pretrain_number = int(re.search('[0-9]{2,3}', model_name).group(0))
        else:
            self.pretrain_number = 0
        

    def _load_loss(self, k, config):
        print(k)
        if k == "loss":
            loss_list = []
            for _, loss_config in config.items():
                loss = self._load_nest_config(**loss_config)
                # loss.set_device(self.device)
                loss_list.append(loss)
            self.lost_list = loss_list
        elif k == "val_loss":
            val_loss_list = []
            if config is None:
                self.val_loss_list = self.lost_list
            else:
                for _, loss_config in config.items():
                    val_loss = self._load_nest_config(**loss_config)
                    # val_loss.set_device(self.device)
                    val_loss_list.append(val_loss)
                self.val_loss_list = val_loss_list

    def _load_optimizer(self, config):
        optimizer_name = config['name']
        optimizer_parameters = config['parameters']
        init_setup = optimizer_parameters['init_setup']
        self.init_lr = init_setup['lr']
        params = [{'params': self.network_architecture.parameters(), 'lr': self.init_lr},
              {'params': self.lost_list[0].parameters(), 'lr': self.init_lr}] \
            if "UncertaintyLoss" in str(type(self.lost_list[0])) \
            else self.network_architecture.parameters()
        optimizer = getattr(optim, optimizer_name)(params, **init_setup)
        if self.pretrain:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        # for g in optimizer.param_groups:
        #     for key, value in optimizer_parameters.items():
        #         g[key] = value
        self.optimizer = optimizer

    def _load_lr_scheduler(self, config):
        if config:
            lr_scheduler_name = config['name']
            lr_scheduler_parameters = config['parameters']
            scheduler = getattr(lr_scheduler, lr_scheduler_name)(self.optimizer, **lr_scheduler_parameters)
            try:
                if self.pretrain:
                    scheduler.load_state_dict(self.checkpoint['scheduler'])
            except Exception as e:
                print(f"No scheduler checkpoint: {e}")
            self.scheduler = scheduler

    def _load_data_generator(self, data_generator_type, config):
        dataset_config = config.get('dataset')
        generator_config = config.get('generator')
        if data_generator_type == DatasetTypeString.TRAIN:
            training_dataset = self._load_nest_config(**dataset_config)
            self.training_generator = data.DataLoader(training_dataset, **generator_config)
        elif data_generator_type == DatasetTypeString.VAL:
            validation_dataset = self._load_nest_config(**dataset_config)
            self.validation_generator = data.DataLoader(validation_dataset, **generator_config)
        elif data_generator_type == DatasetTypeString.TEST:
            test_dataset = self._load_nest_config(**dataset_config)
            self.test_generator = data.DataLoader(test_dataset, **generator_config)

    def _load_nest_config(self, **kwargs):
        file_name = kwargs.get('file')
        parameters = kwargs.get('parameters')
        if "/" in file_name:
            cls_name = convert_str_from_underscore_to_camel(file_name.split("/")[-1])
        else:
            cls_name = convert_str_from_underscore_to_camel(file_name)
        if parameters is None:
            return getattr(import_file(file_name), cls_name)()
        return getattr(import_file(file_name), cls_name)(**parameters)

    def _mkdir_not_exist(self, dir):
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        # os.chmod("/media/yejin/새 볼륨", mode=0o777)
        # os.makedirs(dir, exist_ok=True)

    def train(self):
        lowest_validation_loss = 0
        self.best_model_dir = ""
        start = datetime.now()
        for epoch in range(self.epochs):
            self.network_architecture.train()
            try:
                if "StepLR" in str(type(self.scheduler)):
                    self.scheduler.step()
                    print(self.optimizer.param_groups[0]['lr'])
            except Exception as e:
                print(f"StepLR exception: {e}")
            epoch = epoch + self.pretrain_number + 1
            training_loss, regression_loss, ord_regression_loss = self.calculate_train_loss()
            training_loss /= len(self.training_generator)
            self.board_writer.add_scalar(f"{self.train_summary_writer_folder}/training_loss", training_loss, epoch)
            self.board_writer.add_scalar(f"{self.train_summary_writer_folder}/reg_loss", regression_loss/len(self.training_generator), epoch)
            self.board_writer.add_scalar(f"{self.train_summary_writer_folder}/ord_reg_loss", ord_regression_loss/len(self.training_generator), epoch)
            self.board_writer.add_scalar(self.train_summary_writer_folder, training_loss, epoch)
            if epoch % self.validation_leap == 0 or epoch == (self.epochs + self.pretrain_number):
                self.network_architecture.eval()
                validation_loss, regression_loss, ord_regression_loss = self.calculate_validate_loss()
                validation_loss /= (len(self.validation_generator))
                try:
                    if "ReduceLROnPlateau" in str(type(self.scheduler)):
                        self.scheduler.step(validation_loss)
                        print(self.optimizer.param_groups[0]['lr'])
                except Exception as e:
                    print(f"ReduceLROnPlateau exception: {e}")
                self._save_model(epoch, validation_loss)
                if lowest_validation_loss == 0 or validation_loss < lowest_validation_loss:
                    lowest_validation_loss = validation_loss
                    self.best_model_dir = f"{self.model_save_dir}/model_{epoch}.pth"
                    print(self.best_model_dir)
                now = datetime.now()
                print(f"{self.config_file}: [{(now - start).total_seconds()}] epoch: {epoch}. Validation Loss: {validation_loss}")
                self.board_writer.add_scalar(f"{self.val_summary_writer_folder}/validation_loss", validation_loss, epoch)
                self.board_writer.add_scalar(f"{self.val_summary_writer_folder}/reg_loss", regression_loss/len(self.validation_generator), epoch)
                self.board_writer.add_scalar(f"{self.val_summary_writer_folder}/class_loss", ord_regression_loss/len(self.validation_generator), epoch)

    @abc.abstractmethod
    def calculate_train_loss(self):
        return

    @abc.abstractmethod
    def calculate_validate_loss(self):
        return

    def predict(self):
        # self.best_model_dir = '/data1/yejin/compu/3d_ppmdnn/pretrain/EfficientnetBackboneDrnnF2LossOnly_1AveragePhaseLoss_Adam_inputs_labels_labels_weight_for_each_phase_mask_2022-04-27-22-03-00_seed42/model_28.pth'
        print(self.best_model_dir)
        if getattr(self, "model_mode", None) is not None:
            predict = Predict(self.network_architecture, self.best_model_dir, self.test_generator,
                              self.predict_save_dir, self.device, self.model_parallel, self.model_mode)
        else:
            predict = Predict(self.network_architecture, self.best_model_dir, self.test_generator,
                              self.predict_save_dir, self.device, self.model_parallel)
        predict.predict()

    def evaluate(self):
        evaluation_metrics = Metrics_Evaluation(self.predict_save_dir, self.evaluation_metrics_save_dir, self.device)
        evaluation_metrics.evaluate()

    def __repr__(self):
        dict_str = {key: value for key, value in self.__dict__.items() if key != "network_architecture"}
        return f"{dict_str}"

    def generate_file_name(self):
        network_architecture_name = self.network_architecture.__class__.__name__
        loss_with_weights = [f"{weight}{loss}" for weight, loss in zip(self.loss_weights, self.lost_list)]
        loss_with_weight_str = "_".join(map(str, loss_with_weights))
        optimizer_name = OptimizerType.ADAM if OptimizerType.ADAM in str(self.optimizer) else OptimizerType.SGD
        dataset_train_name = str(self.training_generator.dataset)
        return f"{network_architecture_name}_{loss_with_weight_str}_{optimizer_name}_{dataset_train_name}".replace(" ", "")

    def _save_model(self, epoch, validation_loss):
        loss_state_dict = self.val_loss_list[0].state_dict() if "UncertaintyLoss" in str(type(self.val_loss_list[0])) \
            else None
        torch.save({
                "epoch": epoch,
                "model_state_dict": self.network_architecture.state_dict(),
                "loss_state_dict": loss_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": validation_loss,
                "scheduler": self.scheduler.state_dict(),
            }, f"{self.model_save_dir}/model_{epoch}.pth"
        )



