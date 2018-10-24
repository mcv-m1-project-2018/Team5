# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os

# 3rd party modules
import time

import numpy as np
import pickle

import imageio
from skimage import color, exposure, transform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_files_from_dir(directory, excl_ext=None):
    """
    Get only files from directory.

    :param directory: Directory path
    :param excl_ext: List with extensions to exclude
    :return: List of files in directory
    """

    logger.debug("Getting files in '{path}'".format(path=os.path.abspath(directory)))

    excl_ext = list() if excl_ext is None else excl_ext

    l = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1] not in excl_ext
    ]
    logger.debug("Retrieving {num_files} files from '{path}'".format(num_files=len(l), path=os.path.abspath(directory)))

    return l


def rgb2hsv(img):
    """
    Convert image from RGB to HSV,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the HSV pixel values of the image
    """

    logger.debug("Converting image from RGB to HSV")
    return color.rgb2hsv(img)


def hsv2rgb(img):
    """
    Convert image from HSV to RGB.

    :param img: Numpy array with the HSV pixel values of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    logger.debug("Converting image from HSV to RGB")
    return color.hsv2rgb(img)


def rgb2ycbcr(img):
    """
    Convert image from RGB to YCbCr,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the YCbCr pixel values of the image
    """

    logger.debug("Converting image from RGB to YCbCr")
    return color.rgb2ycbcr(img)


def rgb2lab(img):
    """
    Convert image from RGB to LAB,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the LAB pixel values of the image
    """

    logger.debug("Converting image from RGB to LAB")
    return color.rgb2lab(img)


def get_img(folder_dir, img_dir):
    """
    Get numpy array representation of the image.

    :param folder_dir: Folder path
    :param img_dir: Image path
    :return: Numpy array with the RGB pixel values of the image
    """

    img_path = os.path.join(folder_dir, img_dir)
    logger.debug("Getting image '{path}'".format(path=img_path))

    return imageio.imread(img_path)


def get_patch(img, x_min, y_min, x_max, y_max):
    """
    Get patch from an image.

    :param img: Numpy representation of the image
    :param x_min: X min coordinate
    :param y_min: Y min coordinate
    :param x_max: X max coordinate
    :param y_max: Y max coordinate
    :return: Numpy representation of the patch image
    """
    return img[y_min:y_max, x_min:x_max]


def save_image(img, directory, name, ext='png'):
    """
    Save a numpy array as an image.

    :param img: Numpy array with the pixel values of the image
    :param directory: Folder where the image will be saved
    :param name: Filename of the image
    :param ext: Extension of de image file
    :return: Filename of the image and folder where it was saved
    """

    # Get filename without extension and the new one
    filename = '.'.join(name.split('.')[:-1])
    filename = '{name}.{ext}'.format(name=filename, ext=ext)

    if not os.path.exists(directory):
        logger.debug("{directory} does not exist".format(directory=directory))
        os.mkdir(directory)

    img_path = os.path.join(directory, filename)

    logger.debug("Saving image in {path}".format(path=img_path))

    # Save image
    imageio.imwrite(img_path, img)

    return filename, directory


def histogram(im_array, bins=256):
    """
    This function returns the centers of bins and does not rebin integer arrays. For integer arrays,
    each integer value has its own bin, which improves speed and intensity-resolution.

    This funcion support multi channel images, returning a list of histogram and bin centers for
    each channel.

    :param im_array: Numpy array representation of an image
    :param bins: Number of bins of the histogram
    :return: List of histograms and bin centers for each channel
    """

    hist, bin_centers = list(), list()
    for i in range(im_array.shape[2]):
        _hist, _bin_centers = exposure.histogram(im_array[..., i], bins)
        hist.append(_hist)
        bin_centers.append(_bin_centers)

    return hist, bin_centers


def resize(im_array, width=512, height=512):
    """
    Resize an image represented as a numpy array.

    :param im_array: Numpy array representation of an image
    :param width: Final width of the image
    :param height: Final height of the image
    :return: Resized image
    """

    return transform.resize(im_array, (height, width), anti_aliasing=True)


def split_image(im_array, x_div, y_div):
    """
    Split an image
    :param im_array:
    :param x_div:
    :param y_div:
    :return:
    """

    if im_array.shape[0] < y_div or im_array.shape[1] < x_div:
        raise ValueError("x_div or y_div can't exceed the size of the image")
    if len(im_array.shape) == 3:
        return [
            im_array[y:y + y_div, x:x + x_div, :]
            for y in range(0, im_array.shape[0], y_div)
            for x in range(0, im_array.shape[1], x_div)
        ]
    else:
        return [
            im_array[y:y + y_div, x:x + x_div]
            for y in range(0, im_array.shape[0], y_div)
            for x in range(0, im_array.shape[1], x_div)
        ]


def get_histograms_for_color_spaces(img):
    data = dict()

    hist, bin_centers = histogram(img)
    data['rgb'] = {
        'hist': hist,
        'bin_centers': bin_centers
    }

    hsv = rgb2hsv(img)
    hist, bin_centers = histogram(hsv)
    data['hsv'] = {
        'hist': hist,
        'bin_centers': bin_centers
    }

    ycbcr = rgb2ycbcr(img)
    hist, bin_centers = histogram(ycbcr)
    data['ycbcr'] = {
        'hist': hist,
        'bin_centers': bin_centers
    }

    lab = rgb2lab(img)
    hist, bin_centers = histogram(lab)
    data['lab'] = {
        'hist': hist,
        'bin_centers': bin_centers
    }

    return data


def create_db(imgs_dir, blocks_x=4, blocks_y=4, level=4):
    """
    Create a database from images in a directory. The database is a dictionary of the form

    {
        img_file_name: {
            'hist': [hist_channel_0, hist_channel_1, ... ].
            'bin_centers: [bin_centers_0, bin_centers_1, ... ]
        },
        ...
    }

    :param imgs_dir: Directory of the images
    :param blocks_x:
    :param blocks_y:
    :param level:
    :return: Dictionary with the database
    """

    _color_spaces = ['rgb', 'hsv', 'ycbcr', 'lab']
    db = dict()

    for f in get_files_from_dir(imgs_dir):
        t0 = time.time()
        data = dict()

        img = get_img(imgs_dir, f)

        # Get data for Global Color Histogram
        global_hists = get_histograms_for_color_spaces(img)
        data['global'] = global_hists

        # Get data for Block-based Histogram
        # Create auxiliary dictionary to store concatenated histograms for each color space
        block_hists = {k: list() for k in _color_spaces}
        # For each block in the splitted image, compute the histograms for each color space
        for block in split_image(img, blocks_x, blocks_y):
            hists = get_histograms_for_color_spaces(block)
            # Append each histogram to the corresponding color space
            for k in block_hists.keys():
                block_hists[k].append(hists[k]['hist'])

        data['block'] = block_hists

        # Get data for Spatial Pyramid Representation
        # Create auxiliary dictionary to store concatenated histograms for each color space
        level_hists = dict()
        for l in range(1, level+1):
            level_hists[l] = {k: list() for k in _color_spaces}
            for sub_img in split_image(img, img.shape[1] // l**2, img.shape[0] // l**2):
                hists = get_histograms_for_color_spaces(sub_img)
                # Append each histogram to the corresponding color space
                for k in level_hists[l].keys():
                    level_hists[l][k].append(hists[k]['hist'])

        data['pyramid'] = level_hists

        db[f] = data

        print("Info of image '%s' saved (%.3f s)." % (f, (time.time() - t0)))
        logger.debug("Info of image '%s' saved (%.3f s)." % (f, (time.time() - t0)))

    return db


def save_db(db, fname, fdir=''):
    with open(os.path.join(fdir, fname), 'wb') as p:
        pickle.dump(db, p)


def get_db(fname, fdir=''):
    """
    Read database from pickle file.

    :param fname: Filename of the pickle file
    :param fdir: Directory where pickle file is located. Current file by default
    :return: Dictionary with the database
    """

    with open(os.path.join(fdir, fname), 'rb') as p:
        db = pickle.load(p)

    return db


def dist_euclidean(v1, v2):
    return np.linalg.norm(v1 - v2)


def dist_l1(v1, v2):
    return np.abs(np.sum(v1 - v2))


def dist_chi_squared(v1, v2):
    return np.sum((v1 - v2)**2 / (v1 + v2))


def dist_hist_intersection(v1, v2):
    return sum(list(map(lambda x, y: min(x, y), v1, v2)))


def dist_hellinger_kernel(v1, v2):
    return np.sqrt(v1, v2).sum()
