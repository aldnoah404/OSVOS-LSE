import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import random


def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def overlay_mask(im, ma, color=np.array([255, 0, 0])/255.0):
    assert np.max(im) <= 1.0

    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    alpha = 0.5

    # fg    = im*alpha + np.ones(im.shape)*(1-alpha) * np.array([23,23,197])/255.0
    fg = im * alpha + np.ones(im.shape) * (1 - alpha) * color  # np.array([0,0,255])/255.0

    # Whiten background
    alpha = 1.0
    bg = im.copy()
    bg[ma == 0] = im[ma == 0] * alpha + np.ones(im[ma == 0].shape) * (1 - alpha)
    bg[ma == 1] = fg[ma == 1]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg

# 对像素值作归一化处理，调整到0-1之间
def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name
