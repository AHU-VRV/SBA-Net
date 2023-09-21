import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    segin(pred), edgein(pred_edge), segmask(gt), edgemask(gt_edge):
    segin: 预测结果；segmask：真值；
    edgein：预测结果的边界；edgemask：真值的边界；
    输入格式均为：(B,C,H,W)
'''


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class InverseNet(nn.Module):
    def __init__(self):
        super(InverseNet, self).__init__()
        # Regressor for the 3 * 2 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(224 * 224 * 2, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1.view(-1, 224 * 224), x2.view(-1, 224 * 224)), dim=1)
        return x1, x2, self.fc(x)


class InverseTransform2D(nn.Module):
    def __init__(self):
        super(InverseTransform2D, self).__init__()
        self.tile_factor = 3
        self.resized_dim = 672
        self.tiled_dim = self.resized_dim // self.tile_factor  # 234
        self.inversenet = InverseNet()

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs.float())

        inputs = F.interpolate(inputs.float(), size=(self.resized_dim, 2 * self.resized_dim),
                               mode='bilinear')  # (672, 2*672)
        targets = F.interpolate(targets.float(), size=(self.resized_dim, 2 * self.resized_dim), mode='bilinear')

        tiled_inputs = inputs[:, :, :self.tiled_dim, :self.tiled_dim]
        tiled_targets = targets[:, :, :self.tiled_dim, :self.tiled_dim]
        k = 1
        for i in range(0, self.tile_factor):
            for j in range(0, 2 * self.tile_factor):
                if i + j != 0:
                    tiled_targets = \
                        torch.cat((tiled_targets, targets[:, :, self.tiled_dim * i:self.tiled_dim * (i + 1),
                                                  self.tiled_dim * j:self.tiled_dim * (j + 1)]), dim=0)
                    k += 1

        k = 1
        for i in range(0, self.tile_factor):
            for j in range(0, 2 * self.tile_factor):
                if i + j != 0:
                    tiled_inputs = \
                        torch.cat((tiled_inputs, inputs[:, :, self.tiled_dim * i:self.tiled_dim * (i + 1),
                                                 self.tiled_dim * j:self.tiled_dim * (j + 1)]), dim=0)
                k += 1

        _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)

        mean_square_inverse_loss = (((distance_coeffs * distance_coeffs).sum(dim=1)) ** 0.5).mean()
        return mean_square_inverse_loss


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, edge_weight=0.3, inv_weight=0.5, seg_weight=1, att_weight=0.1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight
        self.edge_loss = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, segin, edgein, segmask, edgemask):
        edgemask = edgemask.cuda()
        total_loss = self.seg_weight * structure_loss(segin.float(),
                                                      segmask.float()) + self.inv_weight * self.inverse_distance(edgein,
                                                                                                                 edgemask)
        return total_loss
