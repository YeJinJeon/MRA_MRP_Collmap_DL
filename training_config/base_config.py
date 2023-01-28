class BaseConfig:
    def __init__(self):
        self.epochs = 200
        self.pretrain = False
        self.train_summary_writer_folder = "training_loss"
        self.val_summary_writer_folder = "validation_loss"
        self.validation_leap = 2
        self.model_parallel = False
        self.network_architecture = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.loss = None
        self.loss_weights = None
        self.val_loss = None
        self.val_loss_weights = None
        self.optimizer = None
        self.lr_scheduler = None
        self.summary_writer_folder_dir = "/data1/yejin/compu/3d_ppmdnn/runs"
        self.log_file_dir = "/data1/yejin/compu/3d_ppmdnn/log_file"
        self.model_save_dir = "/data1/yejin/compu/3d_ppmdnn/pretrain"
        self.predict_save_dir = "/data1/yejin/compu/3d_ppmdnn/result"
        self.evaluation_metrics_save_dir = "/data1/yejin/compu/3d_ppmdnn/evaluation_metrics"

if __name__ == '__main__':
    base_config = BaseConfig()
    print(base_config.__dict__)