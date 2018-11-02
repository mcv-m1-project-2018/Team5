# -*- coding: utf-8 -*-

# Built-in modules
import logging
import math
import os
import time

# 3rd party modules
import imageio
#import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

from skimage import color, exposure, transform

# Local modules
import features as feat

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Logger setup
logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
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

    #return imageio.imread(img_path)
    return cv2.imread(img_path)


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


def histogram(im_array, bins=128):
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

    return transform.resize(im_array, (height, width), anti_aliasing=False)


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

    y_step = math.ceil(im_array.shape[0] / y_div)
    x_step = math.ceil(im_array.shape[1] / x_div)
    if len(im_array.shape) == 3:
        return [
            im_array[y:y + y_step, x:x + x_step, :]
            for y in range(0, im_array.shape[0], y_step)
            for x in range(0, im_array.shape[1], x_step)
        ]
    else:
        return [
            im_array[y:y + y_step, x:x + x_step]
            for y in range(0, im_array.shape[0], y_step)
            for x in range(0, im_array.shape[1], x_step)
        ]


def get_histograms_for_color_spaces(img):
    data = dict()

    hist, _ = histogram(img)
    data['rgb'] = {
        'hist': np.concatenate(hist)
    }

    # hsv = rgb2hsv(img)
    # hist, _ = histogram(hsv)
    # data['hsv'] = {
    #     'hist': np.concatenate(hist)
    # }
    #
    # ycbcr = rgb2ycbcr(img)
    # hist, _ = histogram(ycbcr)
    # data['ycbcr'] = {
    #     'hist': np.concatenate(hist)
    # }
    #
    # lab = rgb2lab(img)
    # hist, _ = histogram(lab)
    # data['lab'] = {
    #     'hist': np.concatenate(hist)
    # }

    return data


def create_db(imgs_dir, FEATURES):
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
    :return: Dictionary with the database
    """

    db = dict()

    for f in get_files_from_dir(imgs_dir, excl_ext=['DS_Store']):
        logger.debug(f)
        t0 = time.time()
        data = dict()

        img = get_img(imgs_dir, f)
        # img = resize(img)

        # Compute the different feature descriptors for the image
        for feat_name, feat_func in FEATURES.items():
            data[feat_name] = feat_func(img)

        db[f] = data

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
    return np.sum(np.abs(v1 - v2))


def dist_chi_squared(v1, v2):
    return np.sum((v1 - v2)**2 / (v1 + v2 + 0.00001))


def dist_hist_intersection(v1, v2):
    return sum(list(map(lambda x, y: min(x, y), v1, v2)))


def dist_hellinger_kernel(v1, v2):
    return np.sum(np.sqrt(v1*v2))


def bbox_to_pkl(list, fname, folder=''):

    if not os.path.exists(folder):
        os.mkdir(folder)

    fname = fname if fname.endswith('.pkl') else '{fname}.pkl'.format(fname=fname)
    path = os.path.join(folder, fname)

    with open(path, 'wb') as f:
        pickle.dump(list, f)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if k < len(predicted):
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def plot_images(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    for idx, img in enumerate(imgs):
        plt.figure()
        plt.title("Figure %.2d" % idx)
        plt.imshow(img)

    plt.show()


def get_number_from_filename(fname):
    return int(fname.split("_")[1].split(".")[0])
