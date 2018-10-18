# -*- coding: utf-8 -*-

import logging
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib.patches as mpatches


from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import equalize_hist
from skimage.morphology import binary_erosion, disk, opening
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import label, regionprops

import utils as ut
import evaluation.evaluation_funcs as ef


# Useful directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join('results')
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
TEST_DIR = os.path.join('dataset', 'test')

# Pickle filename with the training data
PICKLE_TRAIN_DATASET = 'train_data.pkl'
PICKLE_TRAIN_TRAIN_DATASET = 'train_train_data.pkl'
PICKLE_TEST_TRAIN_DATASET = 'train_test_data.pkl'
PICKLE_TEST_DATASET = 'test_data.pkl'

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
F_PLOT = False
F_TRAIN = True


# Global variables
H_RED_MIN = 0.57
H_RED_MAX = 0.63
H_BLUE_MIN = 0.93
H_BLUE_MAX = 0.07
NON_MAX_SUP_TH = 0.5

# Geometrical filter variables (if 'None' that filter won't be applied):

AREA_MIN = 1000
AREA_MAX =  50000
FF_MIN = 0.5
FF_MAX = 2
FR_MIN = 0.5

# Geometrical filter features:
PLOT_BBOX = False
F_SAVE_BBOX_TXT = True


# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # df = ut.get_data(TRAIN_DIR, gt_dir=TRAIN_GTS_DIR, mask_dir=TRAIN_MASKS_DIR)
    # df.to_pickle(PICKLE_TRAIN_DATASET)
    # import sys
    # sys.exit(0)
    df = pd.read_pickle(PICKLE_TRAIN_DATASET)

    # Iterate over traffic signal masks
    for idx, d in df.iterrows():
        if 3 < idx:
            break
        # Get mask path
        mask_path = ut.raw2mask(d['img_file'])
        raw_name = d['img_file']

        # Get original image
        orig_img = ut.get_img(TRAIN_DIR, ut.raw2img(d['img_file']))

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
        orig_img_hsv = ut.rgb2hsv(orig_img)

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
                bboxes.extend(
                    ut.connected_components(mask, AREA_MIN, AREA_MAX, FF_MIN, FF_MAX, FR_MIN, PLOT_BBOX)
                )

            bboxes = ut.non_max_suppression(bboxes, NON_MAX_SUP_TH)

            if F_SAVE_BBOX_TXT:
                ut.bboxes_to_file(bboxes, 'cc.%s.txt' % raw_name, RESULT_DIR, sign_types=None)

        if F_SLID_WIND:
            for mask in masks:
                pass

            bboxes = ut.non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_TEMP_MATCH:
            for mask in masks:
                pass

            bboxes = ut.non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_SLID_WIND_W_INT_IMG:
            for mask in masks:
                pass

            bboxes = ut.non_max_suppression(bboxes, NON_MAX_SUP_TH)

        if F_CONV:
            for mask in masks:
                pass

            bboxes = ut.non_max_suppression(bboxes, NON_MAX_SUP_TH)

        # Final mask
        mask = ut.merge_masks(masks)

        fname = ut.raw2mask(d['img_file'])
        ut.save_image(mask, METHOD_DIR, fname)
        logger.info('{fname} mask saved in {directory}'.format(fname=fname, directory=METHOD_DIR))

    cf_m = ut.confusion_matrix(METHOD_DIR, TRAIN_MASKS_DIR)
    ut.text2file(ut.print_confusion_matrix(cf_m), 'point_based_metrics.txt', METHOD_DIR)
    ut.text2file(ut.print_metrics(ef.performance_evaluation_pixel(*cf_m)), 'point_based_metrics.txt', METHOD_DIR)

    if any(bboxes):
        ut.text2file(ut.print_confusion_matrix(cf_m), 'window_based_metrics.txt', METHOD_DIR)
        ut.text2file(ut.print_metrics(ef.performance_evaluation_pixel(*cf_m)), 'window_based_metrics.txt', METHOD_DIR)
