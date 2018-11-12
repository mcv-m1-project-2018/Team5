# -*- coding: utf-8 -*-

# Built-in modules

# 3rd party modules
import numpy as np
from skimage import feature, color
import cv2 as cv


def normalize(array):
    """
    Normalize a N-dimensional vector.

    :param array: Vector
    :return: Normalized vector
    """
    return array / np.sqrt(np.sum(array**2))


def orb(image):
    """

    :param image:
    :return:
    """
    orb = cv.ORB_create()
    _, des = orb.detectAndCompute(image, None)
    return des


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

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create(1000)

    # find the keypoints and descriptors with SIFT
    _ , descriptors = sift.detectAndCompute(image, None)
    return descriptors

def surf(image):

    # Initiate SIFT detector
    surf = cv.xfeatures2d.SURF_create(1000)

    # find the keypoints and descriptors with SURF
    _ , descriptors = surf.detectAndCompute(image,None)
    return descriptors


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

######
# ORB:
######

# img1 = cv.imread('../dataset/query_devel_W4/ima_000007.jpg',0)          # queryImage
# img2 = cv.imread('../dataset/BBDD_W4/ima_000035.jpg',0)                 # trainImage

# Initiate ORB detector
#orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)


def compute_orb_descriptors(des1, des2, n_matches, thresh):

    #result = compute_orb_descriptors(des1, des2, 10, 500)
    # 500 < Threshold < 1000

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    dist = []
    for m in matches[:n_matches]:
        dist.append(m.distance)

    sqrt_sum = np.sum(np.array(dist)**2) / n_matches
    #return sqrt_sum <= thresh
    return sqrt_sum

########
# SIFT:
########

#img1 = cv.imread('../dataset/query_devel_W4/ima_000005.jpg',0)          # queryImage
#img2 = cv.imread('../dataset/BBDD_W4/ima_000099.jpg',0)                 # trainImage


def compute_sift_descriptor(des1, des2, metric, thresh):

    # Result: compute_sift_descriptor(des1, des2, 0.5, 50)
    # Metric < 0.5, 30 < Threshold < 100 

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < metric*n.distance:
            good.append([m])

    #return(len(good)>thresh)
    return(len(good))

########
# SURF:
########

#img1 = cv.imread('../dataset/query_devel_W4/ima_000005.jpg',0)          # queryImage
#img2 = cv.imread('../dataset/BBDD_W4/ima_000099.jpg',0)                 # trainImage

# Initiate SIFT detector
#surf = cv.xfeatures2d.SURF_create(5000)

# find the keypoints and descriptors with SURF
#kp1, des1 = surf.detectAndCompute(img1,None)
#kp2, des2 = surf.detectAndCompute(img2,None)

def compute_surf_descriptor(des1, des2, metric, thresh):

    # Result: compute_sift_descriptor(des1, des2, 0.5, 5)
    # Metric < 0.5, Threshold NOT SO GOOD! The ratio algorithm may not be the best...

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < metric*n.distance:
            good.append([m])
            
    # return(len(good)>=thresh)
    return(len(good))



#########
# RSift:
#########

'''
Steps for implementing RSift:

(kp1, des1) = rsift(img1)
(kp2, des2) = rsift(img2)

result = match_kpt_rsift(des1, des2, 20, 0.02) ---> (True / False)
'''

def rsift(image, eps=1e-7):
    
    '''Input = OpenCv color Image
       Output = Image keypoints + descriptors'''
    
    # Convert to gray scale:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create(1000)

    # compute SIFT descriptors
    (kps, descs) = sift.detectAndCompute(gray, None)

    # if there are no keypoints or descriptors, return an empty tuple
    if len(kps) == 0:
        return ([], None)

    # apply the Hellinger kernel by first L1-normalizing and taking the
    # square-root
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

    # return a tuple of the keypoints and descriptors
    #return (kps, descs)
    return descs

def compute_rsift_descriptor(des1, des2, n_matches, thresh):
    
    ''' Input = Images descriptors
    Output = True if images match, False otherwise'''
    
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    dist = []
    for m in matches[:n_matches]:
        dist.append(m.distance)


def exclude_kps(kps, descs, bbox_text):

    '''
    Input: Image keypoints + descriptors + Text Bbox
    Output: Filtered keypoints (dont consider kps inside Bbox)

    Implementation example:
    (kp1, des1) = compute_rsift(img1)
    bbox = [95, 867, 767, 991]
    (kp2, des2) = exclude_kps(kp1, des1, bbox) 
    '''
    
    ci = bbox_text[0]
    ri = bbox_text[1]
    cf = bbox_text[2]
    rf = bbox_text[3]
    
    valid_kps = []
    valid_descs = []
    
    for i, k in enumerate(kps):
        
        kr = k.pt[1] 
        kc = k.pt[0] 
        
        print(kr, kc)
        
        if kr < ri or kr > rf or kc < ci or kc > cf:
            
            valid_kps.append(k)
            valid_descs.append(descs[i])
    
    return valid_kps, valid_descs


