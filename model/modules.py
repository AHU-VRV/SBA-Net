import torch
import torch.nn as nn


# High-frequency boosting for more object details.
# input：Coarsest features.
class HFM(nn.Module):
    def __init__(self, num_channels):
        super(HFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()

        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)  # 1, 1, H, W

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()

        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        y = y * high_filter

        y = torch.fft.ifft2(y, s=(h, w)).float()
        y = x + y

        return y


# Sub-band based boosting
# input：Features of intermediate layers.
class DoG_att(nn.Module):
    def __init__(self, num_channels):
        super(DoG_att, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()
        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
        return g.unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()

        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter1 = self.make_gaussian(h_idx, w_idx, h, w, sigma=7)
        high_filter2 = self.make_gaussian(h_idx, w_idx, h, w, sigma=10)
        dog_filter = high_filter1 - high_filter2

        y = y * dog_filter

        y = torch.fft.ifft2(y, s=(h, w)).float()
        y = x + y

        return y
