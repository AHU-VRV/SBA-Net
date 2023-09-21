import os
import cv2
import argparse
import torch
import numpy as np
from torch import nn
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.edge import gradient
from utils.loss import JointEdgeSegLoss
import logging
from model.decoder import pvtv2HDNet6_aggmul


def test(model, test_path, dataset_name):
    data_path = os.path.join(test_path, dataset_name)
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    num_dataset = len(os.listdir(image_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    mdice = 0.0
    for i in range(num_dataset):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        pred = model(image)

        # ---- Evaluate Dice ----
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        input = pred
        target = np.array(gt)

        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        mdice = mdice + dice

    return mdice / num_dataset


def train(train_loader, model, optimizer, epoch, test_path):
    global best
    model.train()
    inverseform_loss = JointEdgeSegLoss(classes=1, edge_weight=0.3, inv_weight=0.5, seg_weight=1, att_weight=0.1, edge='none').cuda()

    # ---- Multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- Data preparation ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- Get GT edge ----
            gts_edge = []
            for j in range(opt.batchsize):
                x = gts[j, :, :, :]
                x = x.cpu().numpy()
                x = x.astype(np.uint8)
                x = x.transpose((1, 2, 0))
                x[x != 255] = 0
                gx, gy = gradient(x)
                x = gx * gx + gy * gy
                x[x != 0] = 1
                x = x.transpose((2, 0, 1))
                gts_edge.append(x)
            gts_edge = torch.tensor(gts_edge).cuda()

            # ---- Rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts_edge = F.upsample(gts_edge.float(), size=(trainsize, trainsize), mode='bilinear',
                                      align_corners=True)
            # ---- Forward ----
            lateral_map_5 = model(images)

            # ---- Get prediction edge ----
            pred_edge = []
            for k in range(opt.batchsize):
                x = lateral_map_5[k, :, :, :]
                x = x.detach().cpu().numpy()
                x = x.astype(np.uint8)
                x = x.transpose((1, 2, 0))
                x[x != 255] = 0
                gx, gy = gradient(x)
                x = gx * gx + gy * gy
                x[x != 0] = 1
                x = x.transpose((2, 0, 1))
                pred_edge.append(x)
            pred_edge = torch.tensor(pred_edge).cuda()

            # ---- Loss function ----
            loss5 = inverseform_loss(lateral_map_5, pred_edge, gts, gts_edge)
            loss = loss5

            # ---- Backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- Record loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        # ---- Train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    # ---- Save model ----
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + str(epoch) + 'checkpoint.pth')
    global dict_plot

    # ---- Choose the best model ----
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
            dice = test(model, test_path, dataset)
            logging.info('epoch: {}, dataset: {}, meandice: {}'.format(epoch, dataset, dice))
            print("epoch: ", epoch, dataset, ":", dice)
            dict_plot[dataset].append(dice)
        meandice = test(model, test_path, 'test')
        dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'checkpoint.pth')
            torch.save(model.state_dict(), save_path + str(epoch) + 'checkpoint.pth')
            print('##############################################################################best', best)
            logging.info(
                '##############################################################################best:{}'.format(best))


if __name__ == '__main__':
    gpu_list = [0]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=80, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')

    parser.add_argument('--augmentation',
                            default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=80, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default="D:/project/polyp/dataset/TrainDataset/",
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default="D:/project/polyp/dataset/TestDataset/",
                        help='path to test dataset')

    parser.add_argument('--log', type=str,
                        default='D:/project/polyp/log/model', help='path to loss')

    parser.add_argument('--train_save', type=str,
                        default='dog7-11', help='path to best .pth')

    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']

    opt = parser.parse_args()

    logging.basicConfig(filename=opt.log + '.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- Build models ----
    model = pvtv2HDNet6_aggmul()
    model.to(device)
    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    best = 0

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
