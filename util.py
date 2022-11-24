import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import torch
import torch.nn as nn
import math
import cv2
from readFlowFile import *
from torch.autograd import Variable
from scipy.ndimage.morphology import binary_dilation
def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
def data_aug(img, mode):

    if mode ==0:
        image=img
    elif mode == 1:
        image =torch.flip(img,[2])
    elif mode == 2:
        image= torch.flip(img,[3])
    elif mode == 3:
        image= torch.flip(img,[2,3])
    elif mode == 4:
        image = img.transpose(2,3)
    elif mode == 5:
        image = img.transpose(2,3)
        image = torch.flip(img,[2])
    elif mode == 6:
        image = img.transpose(2,3)
        image = torch.flip(img,[3])
    elif mode == 7:
        image = img.transpose(2,3)
        image = torch.flip(img,[2,3])
    return image


def data_aug_self(img, mode):
#     print(img.shape)
    img=img.transpose([1,2,0])
    image = np.zeros(img.shape)
    if mode ==0:
        image=img
    elif mode == 1:
        image=np.flipud(img)
    elif mode == 2:
        image=np.flipud(np.rot90(img, 1))
    elif mode == 3:
        image=np.flipud(np.rot90(img, 2))
    elif mode == 4:
        image=np.rot90(img, 1)
    elif mode == 5:
        image=np.rot90(img, 2)
    elif mode == 6:
        image=np.rot90(img, -1)
    elif mode == 7:
        image=np.flipud(np.rot90(img, -1))
    return image.transpose([2,0,1])


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
        