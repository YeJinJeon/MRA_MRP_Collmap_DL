from dataclasses import dataclass

@dataclass(init=True)
class TargetSize:
    height: int
    width: int

class CenterCrop(object):

    def __init__(self, target_size):
        assert isinstance(target_size, (int, TargetSize))
        self.target_size = target_size

    def __call__(self, items):
        # outputs = []
        # for item in items:
        item = items.copy()
        channels, depth, height, width = item.shape
        if isinstance(self.target_size, int):
            start_x = width // 2 - (self.target_size // 2)
            start_y = height // 2 - (self.target_size // 2)
            outputs = item[:, :, start_y:start_y + self.target_size, start_x:start_x + self.target_size]
        elif isinstance(self.target_size, TargetSize):
            start_x = width // 2 - (self.target_size.width // 2)
            start_y = height // 2 - (self.target_size.height // 2)
            outputs = item[:, :, start_y:start_y + self.target_size.height, start_x:start_x + self.target_size.width]
        return outputs