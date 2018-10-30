# -*- coding: utf-8 -*-

# Built-in modules

# 3rd party modules
import numpy as np
from skimage import feature, color


def normalize(array):
    """
    Normalize a N-dimensional vector.

    :param array: Vector
    :return: Normlized vector
    """
    return array / np.sqrt(np.sum(array**2))


def harris(image, sigma=1):
    """
    Compute Harris corner measure response image.

    :param image: Input image
    :param sigma: Standard deviation used for the Gaussian kernel, which is used as
                  weighting function for the auto-correlation matrix
    :return: Harris response image
    """
    return feature.corner_harris(image, sigma=sigma)


def lap_of_gauss(image):
    """
    Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method.

    :param image: Input grayscale image
    :return: A 2d array with each row representing 3 values for a 2D image,
             and 4 values for a 3D image: (r, c, sigma) or (p, r, c, sigma)
             where (r, c) or (p, r, c) are coordinates of the blob and sigma
             is the standard deviation of the Gaussian kernel which detected
             the blob.
    """
    return feature.blob_log(color.rgb2gray(image))


def dif_of_gauss(image):
    """
    Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Gaussian (DoG) method.

    :param image: Input grayscale image
    :return: A 2d array with each row representing 3 values for a 2D image,
             and 4 values for a 3D image: (r, c, sigma) or (p, r, c, sigma)
             where (r, c) or (p, r, c) are coordinates of the blob and sigma
             is the standard deviation of the Gaussian kernel which detected
             the blob.
    """
    return feature.blob_dog(color.rgb2gray(image))


def det_of_hessi(image):
    """
    Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Hessian method.

    :param image: Input grayscale image
    :return: A 2d array with each row representing 3 values for a 2D image,
             and 4 values for a 3D image: (r, c, sigma) or (p, r, c, sigma)
             where (r, c) or (p, r, c) are coordinates of the blob and sigma
             is the standard deviation of the Gaussian kernel which detected
             the blob.
    """
    return feature.blob_doh(color.rgb2gray(image))


def sift(image):
    pass


def surf(image):
    pass


def daisy(image):
    """
    Extract DAISY feature descriptors densely for the given image.

    :param image: Input image (grayscale)
    :return: Grid of DAISY descriptors for the given image as an array
             dimensionality (P, Q, R) where
                P = ceil((M - radius*2) / step)
                Q = ceil((N - radius*2) / step)
                R = (rings * histograms + 1) * orientations
    """
    return feature.daisy(color.rgb2gray(image), visualize=False)


def lbp(image, p=4, r=4):
    """
    Gray scale and rotation invariant LBP (Local Binary Patterns).

    :param image: Graylevel image
    :param p: Number of circularly symmetric neighbour set points
              (quantization of the angular space)
    :param r: Radius of circle (spatial resolution of the operator)
    :return: LBP image
    """
    return feature.local_binary_pattern(color.rgb2gray(image), p, r)


def hog(image, orientations=9, pixels_per_cell=(8, 8)):
    """
    Extract Histogram of Oriented Gradients (HOG) for a given image.

    For this implementation, there is no visualizationn and the feature
    vector is returned as a 1D array.

    :param image: Input image
    :param orientations: Number of orientation bins
    :param pixels_per_cell: Size (in pixels) of a cell
    :return: HOG descriptor for the image
    """

    return feature.hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       visualise=False, feature_vector=True)
