import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, n_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4*growth_rate
        # self.bn1 = nn.BatchNorm3d(n_channels)
        # self.gn1 = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn1 = nn.GroupNorm(1, int(n_channels))
        self.conv1 = nn.Conv3d(n_channels, inter_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                               bias=False)
        # self.bn2 = nn.BatchNorm3d(inter_channels)
        # self.gn2 = nn.GroupNorm(int(inter_channels), int(inter_channels))
        self.gn2 = nn.GroupNorm(1, int(inter_channels))
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=(3, 3, 3),
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn1(x)))
        out = self.conv2(F.relu(self.gn2(out)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, n_channels, n_output_channels):
        super(Transition, self).__init__()
        # self.bn1 = nn.BatchNorm3d(n_channels)
        # self.gn = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn = nn.GroupNorm(1, int(n_channels))
        self.conv1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.avg_pool(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        # self.gn = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn = nn.GroupNorm(1, int(n_channels))
        # self.bn = nn.BatchNorm3d(n_channels)
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 8),
                               padding=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        out = self.conv3(F.relu(self.gn(out)))
        return out

# Original DRNN is the neural network architecture which is the same to paper's NN architecture.
class OriginalDrnn(nn.Module):
    # denblock_config = (6, 12, 24, 48)
    def __init__(self, n_channels, growth_rate, reduction, device):
        super(OriginalDrnn, self).__init__()
        self.device = device
        self.conv1 = nn.Conv3d(n_channels, 8, kernel_size=(3, 3, 3), padding=1)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        n_channels = 8
        # dense 1
        self.denseblock1 = self._make_dense(n_channels, growth_rate, 6)
        n_channels += 6 * growth_rate
        self.transition1 = Transition(n_channels, int(n_channels * reduction))
        n_channels = int(n_channels * reduction)
        # dense 2
        self.denseblock2 = self._make_dense(n_channels, growth_rate, 12)
        n_channels += 12 * growth_rate
        self.transition2 = Transition(n_channels, int(n_channels * reduction))
        n_channels = int(n_channels * reduction)
        # dense 3
        self.denseblock3 = self._make_dense(n_channels, growth_rate, 24)
        n_channels += 24 * growth_rate
        self.transition3 = Transition(n_channels, int(n_channels * reduction))
        n_channels = int(n_channels * reduction)
        # dense 4
        self.denseblock4 = self._make_dense(n_channels, growth_rate, 48)
        n_channels += 48 * growth_rate
        # self.transition4 = Transition(n_channels, n_channels * reduction) # n_channels = 256
        self.transpose_conv1 = nn.ConvTranspose3d(n_channels, int(n_channels * reduction), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        n_channels = int(n_channels * reduction) # n_channesl = 128
        # resblock
        self.resblock1 = ResidualBlock(int(n_channels * 2))
        n_channels = int(n_channels * 4) # n_channels = 512
        self.transpose_conv2 = nn.ConvTranspose3d(n_channels, int(n_channels / 8), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        n_channels = int(n_channels // 8) #n_channels = 64
        self.resblock2 = ResidualBlock(int(n_channels * 2))
        n_channels = int(n_channels * 4) #n_channels = 256
        self.transpose_conv3 = nn.ConvTranspose3d(n_channels, int(n_channels / 8), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        n_channels = int(n_channels // 8)  # n_channels = 32
        self.resblock3 = ResidualBlock(int(n_channels * 2))
        n_channels = int(n_channels * 4) # n_channels = 128
        self.transpose_conv4 = nn.ConvTranspose3d(n_channels, int(n_channels / 16), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        n_channels = int(n_channels // 16) # n_channels = 8
        self.resblock4 = ResidualBlock(int(n_channels * 2))
        # self.last_gn = nn.GroupNorm(32, 32)
        self.last_gn = nn.GroupNorm(1, 32)
        self.last_conv = nn.Conv3d(32, 5, kernel_size=(1, 1, 1))

    def _make_dense(self, n_channels, growth_rate, n_dense_layers):
        layers = []
        for i in range(int(n_dense_layers)):
            layers.append(Bottleneck(n_channels, growth_rate))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x = x.to(self.device, dtype=torch.float)
        out = self.conv1(x)
        encoder_layer_1 = out
        out = self.max_pool1(out)
        encoder_layer_2 = self.denseblock1(out)
        out = self.transition1(encoder_layer_2)
        encoder_layer_3 = self.denseblock2(out)
        out = self.transition2(encoder_layer_3)
        encoder_layer_4 = self.denseblock3(out)
        out = self.transition3(encoder_layer_4)
        out = self.denseblock4(out)
        # decoder
        out = self.transpose_conv1(out)
        out = torch.cat((encoder_layer_4, out), dim=1)
        res1 = self.resblock1(out)
        out = torch.cat((out, res1), dim=1)

        out = self.transpose_conv2(out)
        out = torch.cat((encoder_layer_3, out), dim=1)
        res2 = self.resblock2(out)
        out = torch.cat((out, res2), dim=1)

        out = self.transpose_conv3(out)
        out = torch.cat((encoder_layer_2, out), dim=1)
        res3 = self.resblock3(out)
        out = torch.cat((out, res3), dim=1)

        out = self.transpose_conv4(out)
        out = torch.cat((encoder_layer_1, out), dim=1)
        res4 = self.resblock4(out)
        out = torch.cat((out, res4), dim=1) #n_channels = 32

        out = self.last_conv(out)
        return (torch.tanh(out))


if __name__ == '__main__':
    a = torch.rand((1, 60, 19, 224, 224))
    net = OriginalDrnn(60, 4, 0.5)
    net(a)
