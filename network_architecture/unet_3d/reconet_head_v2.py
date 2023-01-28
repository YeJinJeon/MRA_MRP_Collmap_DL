"""
RecoNet - Tensor Generation and Reconstruction Module
https://github.com/CWanli
"""

import os
from termios import CS5
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torch.nn.functional as F
from torch.autograd import Variable

class TGMandTRM(nn.Module):
    def __init__(self, h, s, r, device):
        super(TGMandTRM, self).__init__()
        self.device = device
        self.rank = r
        self.ps = [1, 1, 1, 1]
        self.h = h
        self.s = s
        conv1_1, conv1_2, conv1_3, conv1_4 = self.ConvGeneration(self.rank, h, s)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3
        self.conv1_4 = conv1_4

        self.lam = nn.Parameter(torch.ones(self.rank, dtype=torch.float32, device=self.device))

        self.pool = nn.AdaptiveAvgPool3d(self.ps[0])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    
    def ConvGeneration(self, rank, h, s):
        conv1 = []
        n = 1
        for _ in range(0, rank):
                conv1.append(nn.Sequential(
                nn.Conv1d(256, 256 // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
                conv2.append(nn.Sequential(
                nn.Conv1d(s, s // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
                conv3.append(nn.Sequential(
                nn.Conv1d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        conv4 = []
        for _ in range(0, rank):
                conv4.append(nn.Sequential(
                nn.Conv1d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv4 = nn.ModuleList(conv4)

        return conv1, conv2, conv3, conv4

    def TukerReconstruction(self, batch_size, s, h, ps, feat, feat1, feat2, feat3):
        b = batch_size
        recon_C = feat.view(b, -1, ps)
        recon_S = feat1.view(b, ps, -1)
        recon_H = feat2.view(b, ps, -1)
        recon_W = feat3.view(b, ps * ps, -1)
        CSHW = torch.bmm(torch.bmm(torch.bmm(recon_C, recon_S).view(b, -1, ps * ps), recon_H).view(b, -1, ps * ps), recon_W).view(b, -1, s, h, h)
        return CSHW
       

    def forward(self, x):
        b, c, _, w, h = x.size()
        # input_x = x.clone()
        C = self.pool(x) #(1, 256, 1, 1, 1)
        S = self.pool(x.permute(0, 2, 3, 4, 1).contiguous()) #(1, 20, 1, 1, 1)
        H = self.pool(x.permute(0, 4, 1, 2, 3).contiguous()) #(1, 14, 1, 1, 1)
        W = self.pool(x.permute(0, 3, 4, 1, 2).contiguous()) #(1, 14, 1, 1, 1)
        # convert to 3d tensor for Conv1d
        C, S, H, W = C.view(1, -1 ,1), S.view(1, -1, 1), H.view(1, -1, 1), W.view(1, -1, 1)
        self.softmax_lam = F.softmax(self.lam,-1)
        local_lam = self.softmax_lam.clone().detach()
        tensor1 = torch.zeros_like(x, requires_grad=False).to(self.device)
        for i in range(0, self.rank):
            tensor1 = tensor1 + local_lam[0].view(1)*self.TukerReconstruction(b, self.s, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](S), self.conv1_3[i](H), self.conv1_4[i](W))
        out = torch.cat((x , F.relu_(x * tensor1)), 1)
        return out


class ReconetHead(nn.Module):
    def __init__(self, in_channels, out_channels, dim, slice_num, rank, dvc):
        super(ReconetHead, self).__init__()
        self.device = dvc
        h = dim // 8 #stride=8, h=4
        s = slice_num
        r = rank
        self.feat = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, dilation=1, padding=1, bias=False),
            nn.GroupNorm(1, int(out_channels)),
            nn.ReLU(inplace=True))

        self.decomp =TGMandTRM(h=h, s=s, r=r, device=self.device)

    def forward(self, x):
        feat = self.feat(x)
        outs = self.decomp(feat)
        return outs

class reconet(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, norm_layer=nn.BatchNorm3d, dim=512, slice_num=20, **kwargs):
        super(reconet, self).__init__()
        self.head = ReconetHead(2048, nclass, norm_layer, dim, slice_num)
        # if aux:
        #     self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, c, h, w = x.size()
        # _, c2, c3, c4 = self.base_forward(x)

        x = list(self.head(x))
        x[0] = upsample(x[0], (c,h,w), mode="trilinear")

        # if self.aux:
        #     auxout = self.auxlayer(c3)
        #     auxout = upsample(auxout, (h,w), **self._up_kwargs)
        #     x.append(auxout)

        return tuple(x)

if __name__ == "__main__":
    # in_channels, out_channels, norm_layer,  dim,  se_loss=True, up_kwargs=None
    # model = reconetHead(2048, 10, nn.BatchNorm3d, 512, 20, True, None).to(torch.device("cuda:0"))
    model = reconet(10, backbone='resnet50', root=None).to(torch.device("cuda:0"))
    a = torch.rand((1, 2048, 20, 64, 64)).to(torch.device("cuda:0"))
    result = model(a)
    print(result.shape)
    
    # ps = [1, 1, 1, 1]
    # pool = nn.AdaptiveAvgPool3d(ps[0])
    # x = torch.rand((1, 256, 20, 14, 14)).to(torch.device("cuda:0"))

    # b, c, s, w, h = x.size()
    # C = pool(x) #(1, 256, 1, 1, 1)
    # S = pool(x.permute(0, 2, 3, 4, 1).contiguous()) #(1, 20, 1, 1, 1)
    # H = pool(x.permute(0, 4, 1, 2, 3).contiguous()) #(1, 14, 1, 1, 1)
    # W = pool(x.permute(0, 3, 4, 1, 2).contiguous()) #(1, 14, 1, 1, 1)

    # # C = feat.view(b, -1, ps)
    # # H = feat2.view(b, ps, -1)
    # # W = feat3.view(b, ps * ps, -1)
    # # CHW = torch.bmm(torch.bmm(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)

    # # method 1
    # CS = torch.bmm(C.view(b, -1, 1), S.view(b, 1, -1)) #(1,256,20)
    # CSH = torch.bmm(CS.view(b, -1, 1), H.view(b, 1, -1)) #(1, 5120, 14)
    # CSHW_1 = torch.bmm(CSH.view(b, -1, 1), W.view(b, 1, -1)) #(1, 7168, 14)
    # CSHW_1 = CSHW_1.view(b, c, s, w, h)
    # print(CSHW_1.shape)


    # # method 2
    # CS = torch.mm(C.view(b, -1, 1), S.view(b, 1, -1)) #(1,256,20)
    # HW = torch.mm(H.view(b, -1, 1), W.view(b, 1, -1)) #(1,14,14)
    # CSHW_2 = torch.mm(CS.view(b, -1, 1), HW.view(b,1,-1)) #(1,5120, 196)
    # CSHW_2 = CSHW_2.view(b, c, s, w, h)
    # print(CSHW_2.shape)

    # print(torch.all(CSHW_1==CSHW_2))


    

