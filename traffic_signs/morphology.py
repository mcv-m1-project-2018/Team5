# -*- coding: utf-8 -*-

import logging
import os
from functools import reduce

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import equalize_hist
from skimage.morphology import binary_erosion, disk, opening
from skimage.restoration import denoise_tv_chambolle

# Directory in the root directory where the results will be saved
from traffic_signs.utils import gt_to_mask, get_img, gt_to_img, get_patch, rgb2hsv, save_image

# Useful directories
RESULT_DIR = os.path.join('results')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
TEST_DIR = os.path.join('dataset', 'test')

# Pickle filename with the training data
PICKLE_DATASET = 'train_data.pkl'

# Method number
METHOD_NUMBER = 1
METHOD_DIR = os.path.join(RESULT_DIR, 'method{number}'.format(number=METHOD_NUMBER))

# Flags
DENOISE = False
EQUALIZE_HIST = False
MORPHOLOGY = True
FILL_HOLES = False
CONNECTED_COMPONENTS = False
SLIDING_WINDOW = False
TEMPLATE_MATCHING = False
SLIDING_WINDOW_WITH_INTEGRAL_IMAGES = False
CONVOLUTIONS = False
PLOT = False


# Logger setup

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    df = pd.read_pickle(PICKLE_DATASET)

    # Iterate over traffic signal masks
    for idx, d in df.iterrows():
        # Get mask path
        mask_path = gt_to_mask(d['gt_file'])

        # Get original image
        orig_img = get_img(TRAIN_DIR, gt_to_img(d['gt_file']))

        # If denoise chambolle flag is set
        if DENOISE:
            orig_img = denoise_tv_chambolle(orig_img, weight=0.2, multichannel=True)

        # If histogram equalization flag is set
        if EQUALIZE_HIST:
            orig_img = equalize_hist(orig_img)

        # Get traffic signal patch
        # orig_traf_sign = get_patch(orig_img, d['tlx'], d['tly'], d['brx'], d['bry'])

        # Convert traffic signal image to HSV color space
        # patch_hsv = rgb2hsv(orig_traf_sign)

        # Convert image to HSV color space
        orig_img_hsv = rgb2hsv(orig_img)

        # Get Hue channel
        h = orig_img_hsv[..., 0]

        # Create masks based on color segmentation
        masks = [
            np.logical_and(h < 0.07, h < 0.93),
            np.logical_and(0.57 < h, h < 0.63)
        ]

        if MORPHOLOGY:
            # kernel = disk(3)
            kernel = np.ones((3, 3))

            morp_masks = [
                binary_fill_holes(opening(masks[0], kernel)),
                binary_fill_holes(opening(masks[1], kernel))
            ]

            if PLOT:
                plt.figure()
                plt.subplot(231)
                plt.imshow(orig_img)
                plt.title('Original: %s' % d['gt_file'])
                plt.subplot(234)
                plt.imshow(h)
                plt.title('HSV: %s' % d['gt_file'])

                plt.subplot(232)
                plt.imshow(masks[0])
                plt.title('Rojo')
                plt.subplot(235)
                plt.imshow(morp_masks[0])
                plt.title('Red eroded and filled')

                plt.subplot(233)
                plt.imshow(masks[1])
                plt.title('Azul')
                plt.subplot(236)
                plt.imshow(morp_masks[1])
                plt.title('Blue eroded and filled')

                plt.show()

            masks = morp_masks

        if CONNECTED_COMPONENTS:
            pass

        if SLIDING_WINDOW:
            pass

        if TEMPLATE_MATCHING:
            pass

        if SLIDING_WINDOW_WITH_INTEGRAL_IMAGES:
            pass

        if CONVOLUTIONS:
            pass

        # Final mask
        mask = 1 * reduce(np.bitwise_or, masks)

        fname = gt_to_mask(d['gt_file'])
        save_image(mask, METHOD_DIR, fname)
        logger.info('{fname} mask saved in {directory}'.format(fname=fname, directory=METHOD_DIR))
