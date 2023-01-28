import random


class HorizontalFlip(object):

    def __init__(self, prob=0.5):
        assert isinstance(prob, (float, int))
        self.prob = prob

    def __call__(self, items):
        if random.random() < self.prob:
            outputs = []
            for item in items:
                item = item[:, :, :, ::-1]
                item = item.copy()
                outputs.append(item)
            return outputs
        return items