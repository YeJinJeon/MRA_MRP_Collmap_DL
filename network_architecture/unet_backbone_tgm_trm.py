""" Full assembly of the parts to form the complete network """
import sys
sys.path.append('.')
from network_architecture.unet_3d.unet_parts import *
from network_architecture.unet_3d.reconet_head import ReconetHead
import torch.nn.functional as F
import torch

class UnetBackboneTgmTrm(nn.Module):
    def __init__(self, n_channels, n_classes, n_slices, device, rank, trilinear=True):
        super(UnetBackboneTgmTrm, self).__init__()
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_slices = n_slices
        self.device = device
        self.rank = rank
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.tgmtrm = ReconetHead(512, 256, 112, self.n_slices, self.rank, self.device)

        self.convdownup = nn.Sequential(
                            nn.Conv3d(512, 512, 3, dilation=1, padding=1, bias=False),
                            nn.GroupNorm(1, int(512)),
                            nn.ReLU(True))
                    
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if self.device:
            x = x.to(self.device, dtype=torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.tgmtrm(x5)
        x5 = self.convdownup(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return F.tanh(logits)

if __name__ == '__main__':

    from utils.utils import count_parameters
    a = torch.rand((1, 40, 20, 224, 224))
    net = UnetBackboneTgmTrm(40, 5, None, 64)
    print(count_parameters(net))
    _p = net(a)
    print(_p.shape)



