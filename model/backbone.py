import torch
from torch import nn
from model.pvtv2 import pvt_v2_b2


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Convolution and pooling
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Fusing CNN and Transformer features to get fused features.
# c_x：CNN features, t_x：Transformer features
class Fusion_mul(nn.Module):
    def __init__(self, channel):
        super(Fusion_mul, self).__init__()

        self.channel = channel

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1)
        )

    def forward(self, c_x, t_x):
        x = torch.mul(c_x, t_x)
        x = x + self.conv(x)

        return x


# Encoder
class trans_fusion_cnn1(nn.Module):
    def __init__(self):
        super(trans_fusion_cnn1, self).__init__()

        # ---- PVT v2 ----
        self.backbone = pvt_v2_b2()

        # # Load the pretrained weights of PVT v2
        path = "./pretrained/pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.relu = nn.ReLU(True)

        # ---- U-Net ----
        self.down1 = Down(3, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 320)
        self.down4 = Down(320, 512)

        # ---- Fusion ----
        self.fus1 = Fusion_mul(64)
        self.fus2 = Fusion_mul(128)
        self.fus3 = Fusion_mul(320)
        self.fus4 = Fusion_mul(512)

    def forward(self, x):
        ux1_ = self.down1(x)
        ux1 = self.maxpool(ux1_)

        B = x.shape[0]
        outs = []

        # TAC
        tx1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            tx1 = blk(tx1, H, W)
        tx1 = self.backbone.norm1(tx1)
        tx1 = tx1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x1 = self.fus1(ux1, tx1)
        outs.append(ux1_)

        # ---- 128 channels ----
        ux2 = self.down2(x1)
        tx2, H, W = self.backbone.patch_embed2(tx1)
        for i, blk in enumerate(self.backbone.block2):
            tx2 = blk(tx2, H, W)
        tx2 = self.backbone.norm2(tx2)
        tx2 = tx2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x2 = self.fus2(ux2, tx2)
        outs.append(x2)

        # ---- 320 channels ----
        ux3 = self.down3(x2)
        tx3, H, W = self.backbone.patch_embed3(tx2)
        for i, blk in enumerate(self.backbone.block3):
            tx3 = blk(tx3, H, W)
        tx3 = self.backbone.norm3(tx3)
        tx3 = tx3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = self.fus3(ux3, tx3)
        outs.append(x3)

        # ---- 512 channels ----
        ux4 = self.down4(x3)
        tx4, H, W = self.backbone.patch_embed4(tx3)
        for i, blk in enumerate(self.backbone.block4):
            tx4 = blk(tx4, H, W)
        tx4 = self.backbone.norm4(tx4)
        tx4 = tx4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x4 = self.fus4(ux4, tx4)
        outs.append(x4)

        return outs
