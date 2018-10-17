# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import pickle
import random

# 3rd party modules
from functools import reduce

import imageio
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from skimage import color
from skimage.measure import label, regionprops
from skimage.transform.integral import integral_image

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


def get_files_from_dir(directory, excl_exts=None):
    """
    Get only files from directory.

    :param directory: Directory path
    :param excl_exts: List with extensions to exclude
    :return: List of files in directory
    """

    logger.debug("Getting files in '{path}'".format(path=os.path.abspath(directory)))

    excl_exts = list() if excl_exts is None else excl_exts

    l = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1] not in excl_exts
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


def int_img(image):
    return integral_image(image)


def sum_over_int_img(int_image, tlx, tly, brx, bry):
    return int_image[bry, brx] - int_image[tly, brx] - int_image[bry, tlx] + int_image[tly, tlx]


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

    if not os.path.exists(directory):
        logger.debug("{directory} does not exist".format(directory=directory))
        os.mkdir(directory)

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


def non_max_suppression(bboxes, overlap_thresh):
    """
    Malisiewicz et al. method for non-maximum suppression.

    :param bboxes: List with bounding boxes
    :param overlap_thresh: Overlaping threshold
    :return: List of merger bounding boxes
    """
    # If there are no boxes, return an empty list
    if len(bboxes) == 0:
        return []

    # If the bounding boxes integers, convert them to floats
    # This is important since we'll be doing a bunch of divisions
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes
    # list
    while 0 < len(idxs):
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap_thresh < overlap)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return bboxes[pick].astype("int")


def bboxes_to_file(bboxes, fname, directory, sign_types=None):
    if sign_types is not None and len(bboxes) != len(sign_types):
        raise ValueError('bboxes and sign_types have different sizes')

    if not os.path.exists(directory):
        os.mkdir(directory)

    fpath = os.path.join(directory, fname)
    with open(fpath, 'w+') as f:
        for idx, bbox in enumerate(bboxes):
            f.write('{tlx} {tly} {brx} {bry} '.format(tlx=bbox[0], tly=bbox[1], brx=bbox[2], bry=bbox[3]))
            f.write('{signal_type}\n'.format(signal_type=sign_types[idx] if sign_types is not None else ''))


def merge_masks(masks):
    return 1 * reduce(np.bitwise_or, masks)


def connected_components(mask0, area_min=None, area_max=None, ff_min=None, ff_max=None, fr_min=None, plot=False):

    """
    :param mask0: Incoming masz (2D array)
    :param area_min: Min area allowed for bbox
    :param area_max: Max area allowed for bbox
    :param ff_min: Min form factor allowed for bbox
    :param ff_max: Max form factor allowed for bbox
    :param fr_min: Min filling ratio allowed for bbox
    :param plot: If 'true' plots mask + selected bboxes
    """


    label_image = label(mask0)
    bbox_list = []

    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(mask0)

    for region in regionprops(label_image):

        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        form_factor = w / h
        filling_ratio = region.filled_area / region.bbox_area

        # Filter by area:
        if area_min is not None and area_max is not None:
            if area_min <= region.bbox_area <= area_max:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        # Filter by form factor:
        if ff_min is not None and ff_max is not None:
            if ff_min < form_factor < ff_max:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        # Filter by filling ratio:
        if fr_min is not None:
            if filling_ratio > fr_min:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        bbox_list.append([minr, minc, maxr, maxc])

        if plot == True:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    return bbox_list


