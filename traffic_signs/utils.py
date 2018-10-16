# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import pickle
import random

# 3rd party modules
import imageio
import numpy as np
import pandas as pd

from skimage import color

# Local modules
from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_files_from_dir(directory):
    """
    Get only files from directory.

    :param directory: Directory path
    :return: List of files in directory
    """

    logger.debug("Getting files in '{path}'".format(path=os.path.abspath(directory)))
    l = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
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
    return img[y_min:y_max, x_min:x_max]


def img_name_to_mask_name(filename):
    """
    Get numpy array representation of the test image.

    :param filename: Image name
    :return: Numpy array with the RGB pixel values of the image
    """

    mask_name = 'mask.{filename}'.format(filename=filename.replace('jpg' or 'png', 'txt'))
    logger.debug("'{filename}' converted to '{mask_name}".format(filename=filename, mask_name=mask_name))

    return mask_name


def gt_to_mask(gt):
    return gt.replace('gt', 'mask').replace('txt', 'png')


def gt_to_img(gt):
    return gt.replace('gt.', '').replace('txt', 'jpg')


def mask_to_gt(gt):
    return gt.replace('mask', 'gt').replace('png', 'txt')


def get_gt_data(dir_path, gt_name):
    f = open(os.path.join(dir_path, gt_name), 'r')
    l = f.readlines()
    f.close()

    return l


def get_unique_values_of_col(df, col):
    return np.unique(df[col].tolist())


def get_n_samples_of_col(df, col, n, val=None):
    if val is None:
        res = df[col]
    else:
        res = df[df[col] == val][col]

    idxs = random.sample(res.index.tolist(), n)

    return [v[0] for v in idxs]


def parse_gt_data(line):
    l = line.strip().split(' ')
    vals = list(map(float, l[:-1]))
    vals.append(l[-1].strip())
    return vals


def threshold_image(img, ths, channel=0):
    """
    Get the mask of a image for the pixels that are between a set threshold values.

    :param img: Numpy array representation of the image
    :param ths: List of tuples of thresholds
    :param channel: Channel of interest of the image. Default 0
    :return: Numpy array with the mask for values between threshold values
    """

    logger.debug("Getting image pixels of channel {channel} that are between the thresholds".format(channel=channel))
    # Create a mask with the same shape of the image filled with 'False'
    mask = np.full(img.shape[:2], False)
    # Get the channel of the image
    c = img[..., channel]

    # Iterate over thresholds
    for th in ths:
        # Add the values between the thresholds as 'True' values to the mask
        mask += np.logical_and(th[0] <= c, c <= th[1])

    return mask


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
    img_path = os.path.join(directory, filename)

    logger.debug("Saving image in {path}".format(path=img_path))

    # Save image
    imageio.imwrite(img_path, img)

    return filename, directory


def confusion_matrix(results_dir, masks_dir):
    """
    Calculate confusion matrix.

    :param results_dir: Directory with calculated masks
    :param masks_dir: Ground truth masks
    :return: Confusion matrix
    """

    # Getting calculated masks
    result_imgs = get_files_from_dir(results_dir)

    # List with values TP, FP, FN, TN
    tf_values = np.zeros(4)

    # Iterate over image paths
    for img_path in result_imgs:
        # Convert image path to mask path
        mask_path = img_name_to_mask_name(img_path)

        # Compute perfomance measures
        tf_val = np.array(performance_accumulation_pixel(
            get_img(results_dir, img_path),
            get_img(masks_dir, mask_path)
        ))

        # Add them up
        tf_values += tf_val

    return tf_values.tolist()


def print_confusion_matrix(values):
    """
    Print a human-readable representation of a confusion matrix.

    :param values: Values of the confusion matrix
    :return: Nothing
    """

    # Reshape matrix values
    values = np.array(values).reshape((2, 2))

    # Turn off scientific notation for float values
    np.set_printoptions(suppress=True)
    print(values)
    np.set_printoptions(suppress=False)


def print_metrics(values):
    print('Precision  : %.3f' % values[0])
    print('Accuracy   : %.3f' % values[1])
    print('Specificity: %.3f' % values[2])
    print('Sensitivity: %.3f' % values[3])
    print('F1 Score   : %.3f' % (2 * values[0] * values[2] / (values[0] + values[2])))


def get_train_data(path_train, path_gt):
    columns = ['gt_file', 'type', 'width', 'height', 'bbox_area', 'form_factor',
             'tly', 'tlx', 'bry', 'brx', 'mask_area', 'filling_ratio', 'mask']
    gts = os.listdir(path_gt)

    l = list()
    for gt in gts:
        mask_path = gt_to_mask(gt)
        mask_img = get_img(path_train, mask_path)
        # orig_img = get_train_img(gt_to_img(gt))

        for gt_datum in get_gt_data(path_gt, gt):
            tly, tlx, bry, brx, signal_type = parse_gt_data(gt_datum)

            d = dict()

            w = brx - tlx
            h = bry - tly

            d['gt_file'] = gt
            d['type'] = signal_type.strip()
            d['width'] = w
            d['height'] = h
            d['bbox_area'] = w * h
            d['form_factor'] = w / h

            d['tly'] = round(tly)
            d['tlx'] = round(tlx)
            d['bry'] = round(bry)
            d['brx'] = round(brx)

            mask_patch = get_patch(mask_img, d['tlx'], d['tly'], d['brx'], d['bry'])
            mask_area = np.count_nonzero(mask_patch)
            d['mask_area'] = mask_area
            d['filling_ratio'] = mask_area / d['bbox_area']
            d['mask'] = mask_patch != 0

            l.append(list(d.values()))

    return pd.DataFrame(l, columns=columns)


def split_data_by(data, key=None, train_size=0.7):
    _keys = list(data.values())[0][0].keys()

    if key is None:
        keys = list(data.keys())
        train_sz = round(len(keys) * train_size)

    elif key in _keys:
        pass
    else:
        raise ValueError('"{key}" key not in data'.format(key=key))


def split_data_by_type(data, train_size=0.7):
    unique_data = data.drop_duplicates('gt_file', keep='first')

    print(unique_data)
    pass


def bbox_to_pkl(bboxes, fname, folder=''):
    if not isinstance(bboxes, list):
        raise ValueError('bboxes variable must be a list')

    fname = fname if fname.endswith('.pkl') else '{fname}.pkl'.format(fname=fname)
    path = os.path.join(folder, fname)

    with open(path, 'wb') as f:
        pickle.dump(bboxes, f)
