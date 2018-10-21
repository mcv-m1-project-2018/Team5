import cv2
import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from utils import sliding_window
from utils import mse
from skimage.measure import compare_ssim, label, regionprops


def invert_ssims(patts):
    for shp in patts:
        for eachsim in patts[shp]['ssim']:
            eachsim['value'] = -eachsim['value']
    return patts


def update_similarities(sim_dists, wind, cand_name, sim_name, new_score, xcoord, ycoord, scale):
    # Code patch: can work as MSE
    if sim_name == 'ssim':
        sim_dists = invert_ssims(sim_dists)
        new_score = - new_score


    if len(sim_dists[cand_name][sim_name]) > 9 and sim_dists[cand_name][sim_name][9]['value'] > new_score:
        del sim_dists[cand_name][sim_name][9]
    if len(sim_dists[cand_name][sim_name]) < 10:
        sim_dists[cand_name][sim_name].append(dict(value=new_score, crop=wind,
                                                   coords=[xcoord * scale,
                                                           ycoord * scale,
                                                           (xcoord + wind.shape[1]) * scale,
                                                           (ycoord + wind.shape[0]) * scale]))
        sim_dists[cand_name][sim_name] = sorted(sim_dists[cand_name][sim_name], key=lambda k: k['value'])
    if sim_name == 'ssim':
        sim_dists = invert_ssims(sim_dists)
    return sim_dists

def sliding_match(image_orig):
    """
    Returns 10 candidates of matches for each candidate shape using a sliding window over a pyramid of images of size
    ratios of 1, 2, 4

    :param image_orig: cv2 Image or numpy array of 3 dimensions (rows, cols, channels)
    :return: list of list where each list is a list of coordinates in the order Left, Top, Right, Bottom
    """

    GT_DIR = 'gt'

    # load the image and define the window width and height
    (winW, winH) = (32, 32)

    # load the groun truth windows
    triangle_1 = cv2.imread(os.path.join(GT_DIR, 'triangle.png'))
    triangle_2 = cv2.imread(os.path.join(GT_DIR, 'triangle_i.png'))
    square = cv2.imread(os.path.join(GT_DIR, 'square.png'))
    rectangle = cv2.imread(os.path.join(GT_DIR, 'rectangle.png'))
    circle = cv2.imread(os.path.join(GT_DIR, 'circle.png'))

    triangle_1 = triangle_1[:, :, 0]
    triangle_2 = triangle_2[:, :, 0]
    square = square[:, :, 0]
    rectangle = rectangle[:, :, 0]
    circle = circle[:, :, 0]

    triangle_1 = np.resize(triangle_1, (32, 32))
    triangle_2 = np.resize(triangle_2, (32, 32))
    square = np.resize(square, (32, 32))
    rectangle = np.resize(rectangle, (32, 32))
    circle = np.resize(circle, (32, 32))

    init_similarities = dict(mse=[], ssim=[])
    patterns = dict(triangle_1=init_similarities, triangle_2=init_similarities, circle=init_similarities,
                    rectangle=init_similarities, square=init_similarities)

    list_ratios = [1, 2, 4]

    pyramid = [np.resize(image_orig,
                          [int(image_orig.shape[0] / ratio_shp),
                           int(image_orig.shape[1] / ratio_shp),
                           int(image_orig.shape[2])])
               for ratio_shp in list_ratios]

    for ii, image in enumerate(pyramid):
        # loop over the sliding window
        for (x, y, window) in sliding_window(image, stepSize=int(32 / list_ratios[ii]), windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            window = window[:, :, 0]

            plt.imshow(window)

            if np.sum(image[:, :, 0]) > 0.3 * (image.shape[0] * image.shape[1]):

                #########################################################################################
                # calculate the mean square error for triangle_1
                score_mse = mse(window, triangle_1)

                patterns = update_similarities(patterns, window, 'triangle_1', 'mse', score_mse, x, y, list_ratios[ii])

                # calculate the structuaral similarity index for triangle_1
                (score_ssim, diff) = compare_ssim(window, triangle_1, full=True)

                patterns = update_similarities(patterns, window, 'triangle_1', 'ssim', score_ssim, x, y, list_ratios[ii])


                #########################################################################################
                # calculate the mean square error for triangle_2
                score_mse = mse(window, triangle_2)

                patterns = update_similarities(patterns, window, 'triangle_2', 'mse', score_mse, x, y, list_ratios[ii])

                # calculate the structuaral similarity index for triangle_2
                (score_ssim, diff) = compare_ssim(window, triangle_2, full=True)

                patterns = update_similarities(patterns, window, 'triangle_2', 'ssim', score_ssim, x, y, list_ratios[ii])


                #########################################################################################
                # calculate the mean square error for circle
                score_mse = mse(window, circle)

                patterns = update_similarities(patterns, window, 'circle', 'mse', score_mse, x, y, list_ratios[ii])

                # calculate the structuaral similarity index for circle
                (score_ssim, diff) = compare_ssim(window, circle, full=True)

                patterns = update_similarities(patterns, window, 'circle', 'ssim', score_ssim, x, y, list_ratios[ii])


                #########################################################################################
                # calculate the mean square error for rectangle
                score_mse = mse(window, rectangle)

                patterns = update_similarities(patterns, window, 'rectangle', 'mse', score_mse, x, y, list_ratios[ii])

                # calculate the structuaral similarity index for rectangle
                (score_ssim, diff) = compare_ssim(window, rectangle, full=True)

                patterns = update_similarities(patterns, window, 'rectangle', 'ssim', score_ssim, x, y, list_ratios[ii])

                #########################################################################################
                # calculate the mean square error for square
                score_mse = mse(window, square)

                patterns = update_similarities(patterns, window, 'square', 'mse', score_mse, x, y, list_ratios[ii])

                # calculate the structuaral similarity index for square
                (score_ssim, diff) = compare_ssim(window, square, full=True)

                patterns = update_similarities(patterns, window, 'square', 'ssim', score_ssim, x, y, list_ratios[ii])

    bboxes = []

    for eachpatt in ['triangle_1', 'triangle_2', 'circle', 'rectangle', 'square']:
        for eachcrop in patterns[eachpatt]['mse']:
            bboxes.append(eachcrop['coords'])
        for eachcrop in patterns[eachpatt]['ssim']:
            bboxes.append(eachcrop['coords'])


    return bboxes