# -*- coding: utf-8 -*-

import os
import pandas as pd
from numpy import asarray

from scipy import signal
from scipy import misc
from PIL import Image
from utils import gt_to_img


def calculate_template(data, train_dir):
    """
    Calculate the average image of the four shapes

    :param data:
    :param train_dir:
    :param clase:

    :return: list of numpy arrays
    """


def calculate_template(data, TRAIN_DIR):
    """
    Calculate the average image of the four shapes

    :param data:
    :param train_dir:
    :param clase:

    :return: list of numpy arrays
    """
    size = 90, 90
    template_A, template_B, template_C, template_D, template_E, template_F = (0,) * 6
    count_A, count_B, count_C, count_D, count_E, count_F, = (0,) * 6

    for index, row in data.iterrows():

        # Read image
        img = Image.open(os.path.join(TRAIN_DIR, gt_to_img(row['gt_file'])))  # .convert('LA')

        # Crop image
        img = img.crop((row['tlx'], row['tly'], row['brx'], row['bry']))

        # Resize and change type to avoid overflow
        img = img.resize(size)
        im1arr = asarray(img)
        im1arrF = im1arr.astype('float')

        # Summatory of each type
        if (row['type'] == 'A'):
            template_A = template_A + im1arrF
            count_A = count_A + 1
            continue
        if (row['type'] == 'B'):
            template_B = template_B + im1arrF
            count_B = count_B + 1
            continue
        if (row['type'] == 'C'):
            template_C = template_C + im1arrF
            count_C = count_C + 1
            continue
        if (row['type'] == 'D'):
            template_D = template_D + im1arrF
            count_D = count_D + 1
            continue
        if (row['type'] == 'E'):
            template_E = template_E + im1arrF
            count_E = count_E + 1
            continue
        if (row['type'] == 'F'):
            template_F = template_F + im1arrF
            count_F = count_F + 1

    # Calculate average
    resultImage_A = Image.fromarray((template_A / count_A).astype('uint8'))
    resultImage_B = Image.fromarray((template_B / count_B).astype('uint8'))
    resultImage_C = Image.fromarray((template_C / count_C).astype('uint8'))
    resultImage_D = Image.fromarray((template_D / count_D).astype('uint8'))
    resultImage_E = Image.fromarray((template_E / count_E).astype('uint8'))
    resultImage_F = Image.fromarray((template_F / count_F).astype('uint8'))

    return [resultImage_A, resultImage_B, resultImage_C, resultImage_D, resultImage_E, resultImage_F]



def template_matching_candidates(candidates, templates, mode="correlation", threshold=0.5):
    if mode != "correlation" and mode != "subtraction":
        raise

    for candidate in candidates:
        for template in templates:
            # Match sizes of candidate and templates

            if mode == "correlation":
                # asdf
                0
            else:
                # Subtraction
                1

            # Apply threshold and save classification

    return 0


def template_matching_global(image, templates, mode="correlation", threshold=0.5):
    if mode != "correlation" and mode != "subtraction":
        raise

    # Create numpy with same size as image

    # Move image_size - template_size steps
    face = misc.face(gray=True) - misc.face(gray=True).mean()
    template = np.copy(face[300:365, 670:750])  # right eye
    template -= template.mean()
    face = face + np.random.randn(*face.shape) * 50  # add noise
    corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

    return 0

