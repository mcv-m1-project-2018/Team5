# -*- coding: utf-8 -*-

import logging
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import equalize_hist
from skimage.morphology import binary_erosion, disk, opening
from skimage.restoration import denoise_tv_chambolle

# Directory in the root directory where the results will be saved
from traffic_signs.utils import gt_to_mask, get_img, gt_to_img, rgb2hsv, save_image, non_max_suppression, merge_masks

# Useful directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join('results')
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
F_DENOISE = False
F_EQ_HIST = False
F_MORPH = True
F_FILL_HOLES = False
F_CONN_COMP = False
F_SLID_WIND = False
F_TEMP_MATCH = False
F_SLID_WIND_W_INT_IMG = False
F_CONV = False
F_PLOT = True


# Global variables
H_RED_MIN = 0.57
H_RED_MAX = 0.63
H_BLUE_MIN = 0.93
H_BLUE_MAX = 0.07
NON_MAX_SUP_TH = 0.5


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
        if F_DENOISE:
            orig_img = denoise_tv_chambolle(orig_img, weight=0.2, multichannel=True)

        # If histogram equalization flag is set
        if F_EQ_HIST:
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
            np.logical_and(H_RED_MIN < h, h < H_RED_MAX),
            np.logical_or(H_BLUE_MIN < h, h < H_BLUE_MAX)
        ]

        # Create list for bboxes
        bboxes = list()

        if F_MORPH:
            # kernel = disk(3)
            kernel = np.ones((3, 3))

            morp_masks = [
                binary_fill_holes(opening(masks[0], kernel)),
                binary_fill_holes(opening(masks[1], kernel))
            ]

            if F_PLOT:
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

        if F_CONN_COMP:
            for mask in masks:
                pass

            bboxes = non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_SLID_WIND:
            for mask in masks:
                pass

            bboxes = non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_TEMP_MATCH:
            for mask in masks:
                pass

            bboxes = non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_SLID_WIND_W_INT_IMG:
            for mask in masks:
                pass

            bboxes = non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_CONV:
            for mask in masks:
                pass

            bboxes = non_max_suppression(bboxes, NON_MAX_SUP_TH)

        # Final mask
        mask = merge_masks(masks)

        fname = gt_to_mask(d['gt_file'])
        save_image(mask, METHOD_DIR, fname)
        logger.info('{fname} mask saved in {directory}'.format(fname=fname, directory=METHOD_DIR))
