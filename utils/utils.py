import importlib.util as im_util
import re
import os
import torch
import numpy as np
import random

def import_file(file_name):
    spec = im_util.spec_from_file_location(".", f"{file_name}.py")
    file = im_util.module_from_spec(spec)
    spec.loader.exec_module(file)
    return file


def convert_str_from_camel_to_underscore(input):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', input).lower()       


def convert_str_from_underscore_to_camel(input):
    input_l = [*map(lambda x: x.capitalize(),input.split("_"))]
    return "".join(input_l)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

