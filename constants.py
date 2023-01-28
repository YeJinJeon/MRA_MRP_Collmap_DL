class OptimizerType:
    ADAM = "Adam"
    SGD = "SGD"

class OptimizerLrScheduler:
    StepLR = "StepLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"


class DatasetTypeString:
    TRAIN = "dataset_train"
    VAL = "dataset_val"
    TEST = "dataset_test"

class DatasetType:
    TRAIN = 0
    VAL = 1
    TEST = 2
    OUTLIERS = 3
    ALL = 4

class WrapperMode:
    WRAPPER5d = "training_wrapper_5d"
    WRAPPER4d = "training_wrapper_4d"

class ModelMode:
    TwoDimensions = "2D"
    ThreeDimensions = "3D"