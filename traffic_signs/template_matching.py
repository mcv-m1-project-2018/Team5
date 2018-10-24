# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from numpy import asarray, argmax
import matplotlib.pylab as plt
import matplotlib.patches as mpatches

from scipy import signal
from scipy import misc
from PIL import Image
import utils as ut
from scipy.signal import correlate2d


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

def get_templates():
    resultImage_A = Image.open('gt/triangle.png')
    resultImage_B = Image.open('gt/triangle_i.png')
    resultImage_CDE = Image.open('gt/circle.png')
    resultImage_F = Image.open('gt/square.png')

    return [resultImage_A, resultImage_B, resultImage_CDE, resultImage_F]


def template_matching_candidates(mask, candidates, templates, mode="correlation", threshold=0.5):
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
    img = mask
    h, w = mask.shape
    print(mask.shape)

    for candidate in candidates:

        #print(candidate)
        # Crop image
        #img_crop = img.crop((candidate[1], candidate[0], candidate[3], candidate[2]))

        # Match sizes of candidate and templates
        #img_crop = img_crop.resize((90,90))
        #plt.figure()
        #plt.imshow(img_crop)
        #plt.show()
        img_crop = asarray(img_crop)
        img_crop = img_crop[:, :, 0]
        img_crop = (img_crop - img_crop.mean())/img_crop.std()

        for template in templates:
            template = template.resize((h, w))
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


def template_matching_candidates_2(mask, candidates, templates, mode="subtraction", threshold=0.5):
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

    filtered_candidates = []
    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(mask)

    for candidate in candidates:
        #print(candidate)
        w = candidate[2] - candidate[0]
        h = candidate[3] - candidate[1]
        #print("{},{}".format(h,w))

        img_crop = asarray(mask)
        img_crop = img_crop[candidate[1]:candidate[3], candidate[0]:candidate[2]]
        #plt.figure()
        #plt.imshow(img_crop)

        memoria = 1

        for n, template in enumerate(templates):
            template = template.resize((w, h))
            template = asarray(template)/255
            template = template[:, :]
            if mode == "correlation":
                #a = signal.correlate2d(img_crop, template, mode='valid')
                diff = correlate2d(img_crop, template)/ (h * w)
                print(np.argmax(diff))
                if memoria < diff:
                    memoria = diff
                    clase = n
            else:
                # Subtraction
                diff = (abs(img_crop - template)).sum() / (h * w)
                if memoria > diff:
                    memoria = diff
                    clase = n

            #print(a)
            #plt.imshow(template - img_crop)
            #plt.show()
            #plt.imshow(diff)
            #plt.show()
        """
        print(memoria)
        if clase == 0:
            print("triangulo")
        if clase == 1:
            print("triangulo invertido")
        if clase == 2:
            print("circulo")
        if clase == 3:
            print("rectangulo")
        """
        if memoria < 0.23:
            filtered_candidates.append(candidate)

            """
            rect = mpatches.Rectangle((candidate[1], candidate[0]), candidate[3] - candidate[1], candidate[2] - candidate[0],
                                      fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            """

    #plt.show()
    return filtered_candidates


def template_matching_global(image, templates, mode="correlation", threshold=0.5):
    if mode != "correlation" and mode != "subtraction":
        raise

    bbox_list = []

    # Create numpy with same size as image
    face = asarray(image)
    face = face.astype('float64')
    face = (face - face.mean()) / face.std()

    # size = 60,60
    # template = templates[5].resize(size)
    for template in templates:
        template = asarray(template)
        template = template.astype('float64')
        template = (template - template.mean()) / template.std()

        corr = signal.correlate2d(face, template, boundary='symm', mode='same')

        print(max(corr))

        y, x = np.unravel_index(argmax(corr), corr.shape)
        minr = y - 45
        minc = x - 45
        maxr = y - 45
        maxc = x - 45
        bbox_list.append([minc, minr, maxc, maxr])

    return bbox_list

