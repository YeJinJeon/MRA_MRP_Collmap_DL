class GetNFirstChannels(object):

    def __init__(self, first_n_channels, dim):
        assert isinstance(first_n_channels, int)
        assert isinstance(dim, int)
        self.first_n_channels = first_n_channels
        self.dim = dim

    def __call__(self, items):
        # TODO: write code to another dim
        outputs = []
        for item in items:
            if self.dim == 0:
                if item.shape[0] > self.first_n_channels:
                    output_item = item[:self.first_n_channels, :, :, :]
                else:
                    output_item = item
                outputs.append(output_item)
        return outputs