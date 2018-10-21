import cv2
import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from utils import sliding_window
from utils import mse
from skimage.measure import compare_ssim, label, regionprops

IMAGE_DIR = 'results'
IMAGE = os.path.join(IMAGE_DIR, '01.002811.png')
GT_DIR = 'gt'

# load the image and define the window width and height
image = cv2.imread(IMAGE)
(winW, winH) = (128, 128)

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

triangle_1 = np.resize(triangle_1, (128, 128))
triangle_2 = np.resize(triangle_2, (128, 128))
square = np.resize(square, (128, 128))
rectangle = np.resize(rectangle, (128, 128))
circle = np.resize(circle, (128, 128))

#thresholds for mse
thresh_mse_triangle_1 = 29797.199863941692
thresh_mse_triangle_2 = 35555.44178485349
thresh_mse_square = 64990.24692872578
thresh_mse_rectangle = 63772.45925927437
thresh_mse_circle = 52027.36146592717

#thresholds for ssim
thresh_ssim_triangle_1 = 0.22022111088715035
thresh_ssim_triangle_2 = 0.12711353812978665
thresh_ssim_square = 0.0006343029487377155
thresh_ssim_rectangle = 0.00019161215566232428
thresh_ssim_circle = 0.08740093769675207

scores_mse_t1 = []
scores_ssim_t1 = []
scores_mse_t2 = []
scores_ssim_t2 = []
scores_mse_s = []
scores_ssim_s = []
scores_mse_c = []
scores_ssim_c = []
scores_mse_r = []
scores_ssim_r = []
count = 0
bbox_list = []

# loop over the sliding window
for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    window = window[:, :, 0]

    plt.imshow(window)

    if 1:

        #########################################################################################
        # calculate the mean square error for triangle_1
        score_mse = mse(window, triangle_1)
        scores_mse_t1.append(score_mse)

        if score_mse < thresh_mse_triangle_1:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        # calculate the structuaral similarity index for triangle_1
        (score_ssim, diff) = compare_ssim(window, triangle_1, full=True)
        scores_ssim_t1.append(score_ssim)

        if score_ssim < thresh_ssim_triangle_1:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])


        #########################################################################################
        # calculate the mean square error for triangle_2
        score_mse = mse(window, triangle_2)
        scores_mse_t2.append(score_mse)

        if score_mse < thresh_mse_triangle_2:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        # calculate the structuaral similarity index for triangle_2
        (score_ssim, diff) = compare_ssim(window, triangle_2, full=True)
        scores_ssim_t2.append(score_ssim)

        if score_ssim < thresh_ssim_triangle_2:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])


        #########################################################################################
        # calculate the mean square error for circle
        score_mse = mse(window, circle)
        scores_mse_c.append(score_mse)

        if score_mse < thresh_mse_circle:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        # calculate the structuaral similarity index for circle
        (score_ssim, diff) = compare_ssim(window, circle, full=True)
        scores_ssim_c.append(score_ssim)

        if score_ssim < thresh_ssim_circle:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])


        #########################################################################################
        # calculate the mean square error for rectangle
        score_mse = mse(window, rectangle)
        scores_mse_r.append(score_mse)

        if score_mse < thresh_mse_rectangle:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        # calculate the structuaral similarity index for rectangle
        (score_ssim, diff) = compare_ssim(window, rectangle, full=True)
        scores_ssim_r.append(score_ssim)

        if score_ssim < thresh_ssim_rectangle:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        #########################################################################################
        # calculate the mean square error for square
        score_mse = mse(window, square)
        scores_mse_s.append(score_mse)

        if score_mse < thresh_mse_square:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])

        # calculate the structuaral similarity index for square
        (score_ssim, diff) = compare_ssim(window, square, full=True)
        scores_ssim_s.append(score_ssim)

        if score_ssim < thresh_ssim_square:
            minr, minc, maxr, maxc = [x, y, x + window.shape[1], y + window.shape[0]]
            bbox_list.append([minr, minc, maxr, maxc])


print(min(scores_ssim_t1))
print(len(bbox_list))

