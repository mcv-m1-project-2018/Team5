# -*- coding: utf-8 -*-

import os
import pandas as pd
from numpy import asarray
import matplotlib.pylab as plt

from scipy import signal
from scipy import misc
from PIL import Image
import utils as ut


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
        raw_name = row['img_file']
        img = Image.open(os.path.join(TRAIN_DIR, raw_name + '.jpg')).convert('LA')

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



def template_matching_candidates(image_dir, candidates, templates, mode="correlation", threshold=0.5):
    """

    :param image_dir: path to the image --> "example/image1.jpg"
    :param candidates: list of masks
    :param templates: list of templates
    :param mode:
    :param threshold:
    :return:
    """
    if mode != "correlation" and mode != "subtraction":
        raise

    # Read image
    img = Image.open(image_dir).convert('LA')

    for candidate in candidates:

        #print(candidate)
        # Crop image
        img_crop = img.crop((candidate[1], candidate[0], candidate[3], candidate[2]))

        # Match sizes of candidate and templates
        img_crop = img_crop.resize((90,90))
        #plt.figure()
        #plt.imshow(img_crop)
        #plt.show()
        img_crop = asarray(img_crop)
        img_crop = img_crop[:, :, 0]
        img_crop = (img_crop - img_crop.mean())/img_crop.std()

        for template in templates:
            template = asarray(template)
            template = template[:,:,0]
            template = (template - template.mean())/template.std()

            if mode == "correlation":
                #a = signal.correlate2d(img_crop, template, mode='valid')
                a = abs(img_crop - template).sum()
                #print(a)
            else:
                # Subtraction
                a = abs(img_crop - template).sum()
                print(a)
            # Apply threshold and save classification


    return 0


def template_matching_global(image, templates, mode="correlation", threshold=0.5):
    if mode != "correlation" and mode != "subtraction":
        raise

    # TODO
    # Create numpy with same size as image

    # Move image_size - template_size steps
    """
    face = misc.face(gray=True) - misc.face(gray=True).mean()
        template = np.copy(face[300:365, 670:750])  # right eye
    template -= template.mean()
    face = face + np.random.randn(*face.shape) * 50  # add noise
    corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    """
    return 0

