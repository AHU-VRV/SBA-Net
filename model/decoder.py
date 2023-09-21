import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import trans_fusion_cnn1
from modules import HFM, DoG_att


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Progressive upsampling
class aggregation_mul(nn.Module):
    def __init__(self, channel=32):
        super(aggregation_mul, self).__init__()
        self.mul3_4 = BasicConv2d(channel, channel, 3, padding=1)
        self.mul2_34 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1)
        )

    def forward(self, x2, x3, x4):
        x4 = self.upsample(x4)
        x3_4 = self.mul3_4(torch.mul(x3, x4))
        x3_4 = x3_4 + self.conv(x3_4)

        x3_4 = self.upsample(x3_4)
        x2_34 = self.mul2_34(torch.mul(x2, x3_4))
        x2_34 = x2_34 + self.conv(x2_34)

        return x2_34


# Model
class pvtv2HDNet6_aggmul(nn.Module):
    def __init__(self, channel=32):
        super(pvtv2HDNet6_aggmul, self).__init__()

        self.backbone = trans_fusion_cnn1()
        self.relu = nn.ReLU(True)

        self.hfm1 = HFM(32)
        self.rfb1_1 = BasicConv2d(64, channel, 1)
        self.dog2 = DoG_att(128, 128)
        self.rfb2_1 = BasicConv2d(128, channel, 1)
        self.dog3 = DoG_att(320, 320)
        self.rfb3_1 = BasicConv2d(320, channel, 1)
        self.rfb4_1 = BasicConv2d(512, channel, 1)

        # ---- Partial decoder ----
        self.agg3 = aggregation_mul(32)
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1)
        )
        self.conv1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Backbone
        pvtv2out = self.backbone(x)
        x1 = pvtv2out[0]
        x2 = pvtv2out[1]
        x3 = pvtv2out[2]
        x4 = pvtv2out[3]

        # SBA
        x1_rfb = self.rfb1_1(x1)
        x1_rfb = self.hfm1(x1_rfb)
        x2 = self.dog2(x2)
        x2_rfb = self.rfb2_1(x2)
        x3 = self.dog3(x3)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        ra5_feat = self.agg3(x2_rfb, x3_rfb, x4_rfb)
        ra5_feat = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear')

        ra5_feat = ra5_feat + self.conv(torch.mul(x1_rfb, ra5_feat))

        ra5_feat = self.conv1(ra5_feat)
        ra5_feat = F.interpolate(ra5_feat, scale_factor=2, mode='bilinear')

        return ra5_feat
