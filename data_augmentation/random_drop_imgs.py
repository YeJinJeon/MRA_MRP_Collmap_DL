import random
import numpy as np

class RandomDropImgs(object):

    def __init__(self, target_size=19, dim=1):
        assert isinstance(target_size, (float, int))
        assert isinstance(dim, int)
        self.target_size = target_size
        self.dim = dim

    def __call__(self, items):
        if items[0].shape[self.dim] != self.target_size:
            rm_idx = random.randint(0, self.target_size)
            outputs = []
            for item in items:
                item = np.delete(item, rm_idx, axis=self.dim)
                item = item.copy()
                outputs.append(item)
            return outputs
        return items