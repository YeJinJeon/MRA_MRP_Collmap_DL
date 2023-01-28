import torch
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(1, int(middle_channels))
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(1, int(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UnetPlusPlus(nn.Module):
    def __init__(self, input_channels, output_channels, dev_0, dev_1):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.dev_0 = dev_0
        self.dev_1 = dev_1
        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)).to(dev_0)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True).to(dev_0)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0]).to(dev_0)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1]).to(dev_0)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2]).to(dev_0)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3]).to(dev_0)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4]).to(dev_0)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0]).to(dev_0)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1]).to(dev_0)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2]).to(dev_1)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3]).to(dev_1)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0]).to(dev_0)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1]).to(dev_1)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2]).to(dev_1)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0]).to(dev_1)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1]).to(dev_1)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0]).to(dev_1)

        self.final = nn.Conv3d(nb_filter[0], output_channels, kernel_size=1).to(dev_1)


    def forward(self, input):

        input = input.to(self.dev_0, dtype=torch.float)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0.to(self.dev_1), self.up(x3_0.to(self.dev_0)).to(self.dev_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0.to(self.dev_1), x1_1.to(self.dev_1), self.up(x2_1.to(self.dev_0)).to(self.dev_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0.to(self.dev_1), x0_1.to(self.dev_1), x0_2.to(self.dev_1), self.up(x1_2.to(self.dev_0)).to(self.dev_1)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1).to(self.dev_1))
        x2_2 = self.conv2_2(torch.cat([x2_0.to(self.dev_1), x2_1, self.up(x3_1.to(self.dev_0)).to(self.dev_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0.to(self.dev_1), x1_1.to(self.dev_1), x1_2.to(self.dev_1), self.up(x2_2.to(self.dev_0)).to(self.dev_1)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0.to(self.dev_1), x0_1.to(self.dev_1), x0_2.to(self.dev_1), x0_3.to(self.dev_1), self.up(x1_3.to(self.dev_0)).to(self.dev_1)], 1))

        output = self.final(x0_4)
        return torch.tanh(output)

if __name__ == '__main__':
    # a = torch.rand(1, 60, 19, 224, 224)
    from utils.utils import count_parameters
    dev_0 = torch.device("cuda:0")
    dev_1 = torch.device("cuda:1")
    # dev_2 = torch.device("cuda:2")
    net = UnetPlusPlus(60, 5, dev_0, dev_1)
    # b = net(a)
    print(count_parameters(net))