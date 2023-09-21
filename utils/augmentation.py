import torch
import cv2
import numpy as np


def data_aug(image, mask):
    img, msk = image, mask
    img = _normalize(img, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
    img, msk = _resize(img, msk)
    img, msk = _HorizontalFlip(img, msk)
    img, msk = _VerticleFlip(img, msk)
    img, msk = _Rotate90(img, msk)
    img, msk = _toTensor(img, msk)
    return img, msk


def no_data_aug(image, mask, size):
    img, msk = image, mask
    img = _normalize(img, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
    img, msk = _resize(img, msk, size)
    img, msk = _toTensor(img, msk)
    return img, msk


def _normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def _resize(image, mask, size=(256, 256)):
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size)
    return image, mask


def _toTensor(image, mask):
    img, msk = image.copy(), mask.copy()
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    msk = torch.from_numpy(msk).float()
    return img, msk


def _HorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def _VerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def _Rotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask
