import csv
import os
from re import L
import numpy as np
import pandas as pd
import torch
import collections
from torch._six import string_classes
from torch.utils import data
import torchvision.transforms as transforms
from data_augmentation.center_crop import CenterCrop
from constants import DatasetType
from utils.dce_mra.load_dicom import load_dce
from utils.dsc_mrp.load_dicom import load_dsc
from utils.load_map import load_phase_map
# from utils.load_map_2 import load_phase_map
from utils.dsc_mrp.load_map import load_phase_map_dsc
from utils.dce_mra.preprocess import preprocess_dce
from utils.dsc_mrp.preprocess import preprocess_dsc


class MraDataset(data.Dataset):

    def __init__(self, csv_file, data_type, transform=None):
        self.csv_file = csv_file
        self.data_type = data_type
        self.src_dirs = []
        self.label_dirs = []
        self._get_file_names(self.csv_file)
        if data_type == "dsc":
            transform = transforms.Compose([CenterCrop(224)])
        self.transform = transform

    def _get_file_names(self, dataset_file):
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                self.src_dirs.append(row[0])
                self.label_dirs.append(row[0])
        return self.src_dirs, self.label_dirs
            
    def __len__(self):
        return len(self.src_dirs)

    def __getitem__(self, idx):
        file_name = self.label_dirs[idx]
        print(file_name)
        if self.data_type == "dce":
            imgs, _, hdrs = load_dce(self.src_dirs[idx], to_crop=True, auto_crop=True)
            phases, properties = load_phase_map(self.label_dirs[idx], prefix="dce")
            image, label, mask = preprocess_dce(imgs, phases, properties)
        elif self.data_type == "dsc":
            phasemap_folder = file_name.split("/")[-1]
            map_version = "new" if phasemap_folder in ["PWI_DSC_Collateral_py", "PWI_source_DSC_Collateral_py"] else "old"
            imgs, masks, hdrs = load_dsc(self.src_dirs[idx])
            phases = load_phase_map_dsc(self.label_dirs[idx], map_version)
            image, label, mask = preprocess_dsc(imgs, phases, masks)
        else:
            print("Datatype Error: wrong data type")
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            mask = self.transform(mask)
        outputs = [image, label, mask, file_name, hdrs]
        return tuple(outputs)


def _collate_fn(batch):
        
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        ilens_torch : B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([64000, 64000, 64000], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    elem = batch[0]
    elem_type = type(elem) # tuple
    
    #return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return _collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            for key in elem:
                for d in batch:
                    return elem_type({key: _collate_fn([d[key]])})
            # return elem_type({key: _collate_fn([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: _collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [_collate_fn(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([_collate_fn(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [_collate_fn(samples) for samples in transposed]
    else:
        return batch[0]


class MraLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(MraLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


if __name__ == '__main__':
    
    mra_dataset = MraDataset("./dataset/split_dce_sample.csv")
    generator_config = {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 4
    }
    test_generator = MraLoader(mra_dataset, **generator_config)
    

    for (input, label, mask, file_name, hdr) in test_generator:
        print(input.shape)
        print(label.shape)
        print(mask.shape)
        print(len(hdr))
