# -*- coding: utf-8 -*-

import logging
import os
import random

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import equalize_hist
from skimage.morphology import binary_erosion, disk, opening
from skimage.restoration import denoise_tv_chambolle

# Directory in the root directory where the results will be saved
from traffic_signs.utils import gt_to_mask, get_img, gt_to_img, get_patch, rgb2hsv

# Useful directories
RESULT_DIR = os.path.join('results')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
TEST_DIR = os.path.join('dataset', 'test')

PICKLE_DATASET = 'train_data.pkl'

# Flags
FILL_HOLES = True
EQUALIZE_HIST = False
DENOISE = True


# Logger setup

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    df = pd.read_pickle(PICKLE_DATASET)

    for idx, d in df.iterrows():

        mask_path = gt_to_mask(d['gt_file'])

        orig_img = get_img(TRAIN_DIR, gt_to_img(d['gt_file']))

        # im_hsv = rgb2hsv(equalize_hist(orig_img))
        if DENOISE:
            orig_img = denoise_tv_chambolle(orig_img, weight=0.2, multichannel=True)

        im_hsv = rgb2hsv(orig_img)

        im = get_patch(orig_img, d['tlx'], d['tly'], d['brx'], d['bry'])
        patch_hsv = rgb2hsv(im)
        h = im_hsv[..., 0]

        # plt.figure()
        # plt.subplot(231)
        # plt.imshow(orig_img)
        # plt.title('Original: %s' % d['gt_file'])
        # plt.subplot(234)
        # plt.imshow(h)
        # plt.title('HSV: %s' % d['gt_file'])

        # kernel = disk(3)
        kernel = np.ones((3, 3))

        plt.subplot(232)
        mask = np.logical_and(h < 0.07, h < 0.93)
        plt.imshow(mask)
        plt.title('Rojo')
        plt.subplot(235)
        op = opening(mask, kernel)
        fin = binary_fill_holes(op)
        plt.imshow(fin)
        plt.title('Rojo erosionado y rellenado')
        # plt.imshow(opening(mask, disk(r_disk)))
        # plt.title('Rojo erosionado')

        plt.subplot(233)
        mask = np.logical_and(0.57 < h, h < 0.63)
        plt.imshow(mask)
        plt.title('Azul')
        plt.subplot(236)
        op = opening(mask, kernel)
        fin = binary_fill_holes(op)
        plt.imshow(fin)
        plt.title('Azul erosionado y rellenado')
        # plt.imshow(opening(mask, disk(r_disk)))
        # plt.title('Azul erosionado')
        # plt.show()
        print(4)

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(orig_img)
        # plt.title('Original image')
        # plt.subplot(122)
        # plt.imshow(equalize_hist(orig_img))
        # plt.title('Equalized image')

        plt.show()
        # Noise filtering, hole filling, object separation
        # mask = mask_patch != 0



