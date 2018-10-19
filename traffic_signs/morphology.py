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
PICKLE_TEST_DATASET = 'test_data.pkl'

# Method number
METHOD_NUMBER = 2
METHOD_DIR = os.path.join(RESULT_DIR, 'method{number}'.format(number=METHOD_NUMBER))


# Flags
F_DENOISE = False
F_EQ_HIST = False
F_MORPH = True
F_FILL_HOLES = False
F_CONN_COMP = True
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
AREA_MAX = 50000
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
    # If train flag is True, select train dataset
    if F_TRAIN:
        df = pd.read_pickle(PICKLE_TRAIN_DATASET)
    else:
        df = pd.read_pickle(PICKLE_TEST_DATASET)

    # Dictionary with raw names as keys and list of bboxes as values
    # (raw names are those without extension and prefix)
    bboxes_found = dict()

    # Iterate over traffic signal masks
    for idx, d in df.iterrows():
        # Get raw name and the asociated image name
        raw_name = d['img_file']
        img_name = ut.raw2img(raw_name)

        # Get original image
        orig_img = ut.get_img(TRAIN_DIR, img_name)

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

        # Create list for bboxes for this image
        bboxes_in_img = list()

        # If morphology flag is set
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
                plt.title('Original: %s' % img_name)
                plt.subplot(234)
                plt.imshow(h)
                plt.title('HSV: %s' % img_name)

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

        # If connected component flag is set
        if F_CONN_COMP:
            # Iterate over the different masks previously calculated.
            # For each max, compute the bounding boxes found in the mask
            for mask in masks:
                bboxes_in_img.extend(
                    ut.connected_components(mask, AREA_MIN, AREA_MAX, FF_MIN, FF_MAX, FR_MIN, PLOT_BBOX)
                )

            # As the bounding box can be found in different masks, non maximal supression
            # is applied in order to keep only those that are different
            bboxes_in_img = ut.non_max_suppression(bboxes_in_img, NON_MAX_SUP_TH)

            # If save bbox flag is set, save the bounding boxes in the image
            if F_SAVE_BBOX_TXT:
                ut.bboxes_to_file(bboxes_in_img, 'cc.%s.txt' % raw_name, METHOD_DIR, sign_types=None)

        # If sliding window flag is set
        if F_SLID_WIND:
            # Iterate over the different masks previously calculated.
            # For each max, compute the bounding boxes found in the mask
            for mask in masks:
                pass

            # As the bounding box can be found in different masks, non maximal supression
            # is applied in order to keep only those that are different
            bboxes_in_img = ut.non_max_suppression(bboxes_in_img, NON_MAX_SUP_TH)

        # If template matching flag is set
        if F_TEMP_MATCH:
            # Iterate over the different masks previously calculated.
            # For each max, compute the bounding boxes found in the mask
            for mask in masks:
                pass

            # As the bounding box can be found in different masks, non maximal supression
            # is applied in order to keep only those that are different
            bboxes_in_img = ut.non_max_suppression(bboxes_in_img, NON_MAX_SUP_TH)

        # If sliding window with integral image flag is set
        if F_SLID_WIND_W_INT_IMG:
            # Iterate over the different masks previously calculated.
            # For each max, compute the bounding boxes found in the mask
            for mask in masks:
                pass

            # As the bounding box can be found in different masks, non maximal supression
            # is applied in order to keep only those that are different
            bboxes_in_img = ut.non_max_suppression(bboxes_in_img, NON_MAX_SUP_TH)

        # If sliding window with convolution flag is set
        if F_CONV:
            # Iterate over the different masks previously calculated.
            # For each max, compute the bounding boxes found in the mask
            for mask in masks:
                pass

            # As the bounding box can be found in different masks, non maximal supression
            # is applied in order to keep only those that are different
            bboxes_in_img = ut.non_max_suppression(bboxes_in_img, NON_MAX_SUP_TH)

        # Final mask: Merge of the previous masks
        mask = ut.merge_masks(masks)

        # Get mask name from raw name
        fname = ut.raw2mask(raw_name)
        # Save mask in directory
        ut.save_image(mask, METHOD_DIR, fname)
        logger.info('{fname} mask saved in {directory}'.format(fname=fname, directory=METHOD_DIR))

        # If bounding boxes were found in the image, save in the dictionary bboxes_found
        if any(bboxes_in_img):
            l = bboxes_found.get(raw_name, [])
            l.extend(bboxes_in_img)
            bboxes_found[raw_name] = ut.non_max_suppression(l, NON_MAX_SUP_TH)

    # Compute confusion matrix for pixel based metrics and save into a file
    conf_mat = ut.confusion_matrix(METHOD_DIR, TRAIN_MASKS_DIR)
    ut.text2file(ut.print_confusion_matrix(conf_mat), 'point_based_metrics.txt', METHOD_DIR)
    ut.text2file(ut.print_pixel_metrics(ef.performance_evaluation_pixel(*conf_mat)), 'point_based_metrics.txt', METHOD_DIR)

    # Compute confusion matrix for window based metrics and save into a file
    if any(bboxes_found):
        for fname, bboxes in bboxes_found.items():
            bboxes_found[fname] = [ut.bbox2evalformat(bbox) for bbox in bboxes]

        # ut.text2file(ut.print_confusion_matrix(conf_mat), 'window_based_metrics.txt', METHOD_DIR)
        # ut.text2file(ut.print_pixel_metrics(ef.performance_evaluation_pixel(*conf_mat)), 'window_based_metrics.txt', METHOD_DIR)
