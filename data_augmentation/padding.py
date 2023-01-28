import torch.nn.functional as F
import torch
from data_augmentation.center_crop import TargetSize


class Padding(object):

    def __init__(self, target_size):
        assert isinstance(target_size, (int, TargetSize))
        self.target_size = target_size

    def __call__(self, items):
        outputs = []
        for item in items:
            item = torch.from_numpy(item)
            _, _, h, w = item.shape
            if isinstance(self.target_size, int):
                w_diff = (self.target_size - w) // 2
                h_diff = (self.target_size - h) // 2
                if self.target_size >= w:
                    w_diff_plus = w_diff + 1 if (self.target_size - w) % 2 != 0 else w_diff
                else:
                    w_diff_plus = w_diff - 1 if (self.target_size - w) % 2 != 0 else w_diff
                if self.target_size >= h:
                    h_diff_plus = h_diff + 1 if (self.target_size - h) % 2 != 0 else h_diff
                else:
                    h_diff_plus = h_diff - 1 if (self.target_size - h) % 2 != 0 else h_diff
                output_item = F.pad(item, (w_diff, w_diff_plus, h_diff, h_diff_plus))
                outputs.append(output_item.numpy())
            elif isinstance(self.target_size, TargetSize):
                if self.target_size.width >= w:
                    w_diff = (self.target_size.width - w) // 2
                    w_diff_plus = w_diff + 1 if (self.target_size.width - w) % 2 != 0 else w_diff
                else:
                    w_diff = -((w - self.target_size.width) // 2)
                    w_diff_plus = w_diff - 1 if (self.target_size.width - w) % 2 != 0 else w_diff
                if self.target_size.height >= h:
                    h_diff = (self.target_size.height - h) // 2
                    h_diff_plus = h_diff + 1 if (self.target_size.height - h) % 2 != 0 else h_diff
                else:
                    h_diff = -((h - self.target_size.height) // 2)
                    h_diff_plus = h_diff - 1 if (self.target_size.height - h) % 2 != 0 else h_diff
                output_item = F.pad(item, (w_diff, w_diff_plus, h_diff, h_diff_plus))
                outputs.append(output_item.numpy())
        return outputs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    raw_dce = np.load("/home/longlh/hard_2/CMC_KU_AI_New/BP1185 20200119/DMRA_DCE_Collateral_crop/MatFiles/IMG_n01.npy")
    n, d, h, w = raw_dce.shape
    padding = Padding(TargetSize(176, 256))
    raw_dce = raw_dce.reshape(1, n, d, h, w)
    raw_dce = padding(raw_dce)[0]
    print(raw_dce.shape)
    dy, dx = 176, 256
    nrows, ncols = 1, 1
    figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
    for i in range(n):
        os.makedirs(f"/home/longlh/Desktop/BP1185_20200119_bad_case_raw/timeseries_{(i+1)}")
        for j in range(d):
            f, axes = plt.subplots(1, figsize=figsize)
            f.subplots_adjust(0, 0, 1, 1, 0, 0)
            axes.axis("off")
            axes.imshow(raw_dce[i][j], cmap='gray')
            axes.set(xticks=[], yticks=[])
            plt.savefig(f"/home/longlh/Desktop/BP1185_20200119_bad_case_raw/timeseries_{(i+1)}/slice_{(j+1)}")
