# -*- coding: utf-8 -*-

# Built-in modules
import cv2
import logging
import numpy as np
import os
import pickle

# Local modules
import utils as ut
import features as feat
import text as text

# Useful directories
TRAIN_MUSEUM_DIR = os.path.join('dataset', 'w5_BBDD_random')
TRAIN_QUERY_DIR = os.path.join('dataset', 'w5_devel_random')
GTS_DIR = os.path.join('dataset', 'w5_query_devel.pkl')
GTS_BBOXES_DIR = os.path.join('dataset', 'w5_text_bbox_list.pkl')
RESULT_DIR = os.path.join('pkl')

# Pickle filenames with the training data
PICKLE_MUSEUM_DATASET = 'train_museum.pkl'
PICKLE_QUERY_DATASET = 'train_query.pkl'

# Global variables
K = 10
COLOR_SPACE_LIST = ['rgb']
FEATURES = {
    'orb': feat.orb,
    'rsift': feat.rsift,
    # 'sift': feat.sift,
    # 'surf': feat.surf,
    # 'hog': feat.hog,
}

orb_values = [(10, 900),(20, 1100)]
rsift_values = [(10, 0.04), (15, 0.06)]
sift_values = [(0.4, 20), (0.4, 15)]
surf_values = [(0.5, 100), (0.4, 50)]

# TODO: COMENTAR SI NO ANDA
FEATURES = {
    'orb': {
        'values': [(10, 550)],
        'func': feat.compute_orb_descriptors,
    },
    # 'sift': {
    #     'values': [(0.4, 20), (0.4, 15)],
    #     'func': feat.compute_sift_descriptor,
    # },
    # 'surf': {
    #     'values': [(0.5, 100), (0.4, 50)],
    #     'func': feat.compute_surf_descriptor,
    # },
    'rsift': {
        'values': [(10, 0.04)],
        'func': feat.compute_rsift_descriptor,
    }
}

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info("Starting Museum Painting Retrieval")

    """
    ##################################  TASK 1: TEXT  ####################################
    """
    candidates = list()

    candidates = []

    if not os.path.exists("pkl/bboxes_iou_0.86658.pkl"):
        # Read groundtruth
        gt_annotations = ut.get_db(GTS_BBOXES_DIR)
        # Read images and find text_area
        for f in ut.get_files_from_dir(TRAIN_MUSEUM_DIR, excl_ext=['DS_Store']):
            img = ut.get_img(TRAIN_MUSEUM_DIR, f)
            candidates.append([ut.get_number_from_filename(f), text.get_text_area(img, f, gt=gt_annotations[ut.get_number_from_filename(f)])])

        # Sort bboxes
        candidates.sort(key=lambda x: x[0])
        candidates = [x[1] for x in candidates]

        # Compute intersection over union
        mean_iou = text.compute_mean_iou(candidates, gt_annotations)

        # Export pkl
        text = "bboxes_iou_{0:.5f}.pkl".format(mean_iou)
        ut.bbox_to_pkl(candidates, text, folder=RESULT_DIR)
    else:
        candidates = ut.get_db("pkl/bboxes_iou_0.86658.pkl")

    """
    ##################################  TASK 2: ROTATE & CROP  ####################################
    """

    # List of the frames of each painting
    frames = list()
    # Rescale the images to this width
    width = 512
    # Iterate over query images
    for img_fname in sorted(ut.get_files_from_dir(TRAIN_QUERY_DIR, excl_ext=['DS_Store'])):
        # Read image
        img = cv2.imread(os.path.join(TRAIN_QUERY_DIR, img_fname))

        # We are going to resize the image, so we need to keep track of the initial scale
        size_ini = img.shape[:2]
        # Resize the image
        img = ut.resize(img, width=width)
        # Save the final shape of the image
        size_end = img.shape[:2]
        # Copy
        color = img.copy()

        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        # Dilate and median blur to "mask" the data from the painting and focus only on the borders
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.medianBlur(img, 3)
        # Adaptive threshold and morp ops to connect lines
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        th2 = cv2.erode(th2, kernel, iterations=1)
        th2 = cv2.dilate(th2, kernel, iterations=1)

        # Detect the edges of the image
        edged = cv2.Canny(th2, 20, 190)
        kernel = np.ones((3, 3), np.uint8)
        # Dilate the image to connect lines
        edged = cv2.dilate(edged, kernel, iterations=1)

        # Detect contours in edged image
        _, cnts, __ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Painting frame contour
        frame = None
        # We will look for polygon of 4 sides. If we can't find them, we'll look for 5 sides polygons
        approx_5 = None
        # Loop over the contours
        for c in cnts:
            # Calculate the perimeter of the contour
            peri = cv2.arcLength(c, True)
            # Approximate the contour by a polygon
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)

            # If the approximated contour has four points, then we can assume that we have found the frame
            if len(approx) == 4:
                frame = approx
                break
            # If not, if the contour has five points, save the contour
            elif len(approx) == 5 and approx_5 is None:
                approx_5 = approx

        # If we didn't find the frame, check for 5 sides polygons
        if frame is None:
            # If we didn't find a 5 sides polygon either, use the first one (larger area)
            if approx_5 is None:
                peri = cv2.arcLength(cnts[0], True)
                frame = cv2.approxPolyDP(cnts[0], 0.05 * peri, True)
            else:
                frame = approx_5

        # Calculate convex hull of the frame
        frame = cv2.convexHull(frame)

        # Get the rectangle with the min area
        rect = cv2.minAreaRect(frame)
        frame = cv2.boxPoints(rect)
        frame = np.int0(frame)

        # Get the 2 main angles of the frame
        angles = ut.get_angles(frame)

        rect2 = (rect[0], rect[1], angles[-1])
        img_croped = ut.crop_min_area_rect(color, rect2)
        cv2.imshow("Cropped", img_croped)
        cv2.waitKey(0)

        # As we scale the image, we need to scale back the contour to the original image
        frame_orig = list(map(lambda x: ut.rescale(x, size_ini, size_end), frame))

        # Save data to export
        frames.append([ut.rad2deg(angles[-1]), frame_orig])

    # Save the data in a pickle file
    ut.bbox_to_pkl(frames, fname='frames', folder='pkl')

    """
    ##################################  TASK 3: Filter keypoints  ####################################
    """
    # Check for training database file
    if not os.path.exists(PICKLE_MUSEUM_DATASET):
        logger.info("Creating pickle database for museum dataset...")
        db_museum = ut.create_db(TRAIN_MUSEUM_DIR, FEATURES, candidates)
        ut.save_db(db_museum, PICKLE_MUSEUM_DATASET)
    else:
        logger.info("Reading pickle database for museum dataset...")
        db_museum = ut.get_db(PICKLE_MUSEUM_DATASET)

    logger.info("Loaded data")

    """
    ##################################  TASK 4: Retrieval system and evaluation  ####################################
    ############################### WARNING: Don't touch below this sign. Ask Pablo #################################
    """

    # Check for query database file
    if not os.path.exists(PICKLE_QUERY_DATASET):
        logger.info("Creating pickle database for query dataset...")
        db_query = ut.create_db(TRAIN_QUERY_DIR, FEATURES, query=True)
        ut.save_db(db_query, PICKLE_QUERY_DATASET)
    else:
        logger.info("Reading pickle database for query dataset...")
        db_query = ut.get_db(PICKLE_QUERY_DATASET)

    # Ground truth for query images
    if os.path.exists(GTS_DIR):
        with open(os.path.join(GTS_DIR), 'rb') as f:
            gts = pickle.load(f)
            logger.info("gt = {}".format(gts))
            # Change format to calculate mAP
            gts = [x[1] for x in gts]
    else:
        gts = False
        logger.info("No groundtruth")

    # Check if the directory to store the results exists
    if not os.path.exists(RESULT_DIR):
        logger.info("Creating %s" % RESULT_DIR)
        os.mkdir(RESULT_DIR)

    # Iterate over color spaces defined above
    for space_color in COLOR_SPACE_LIST:
        # Iterate over feature descriptors defined above
        for feat_func, feat_props in FEATURES.items():
            print(feat_func)
            logger.info(feat_func)
            values = ""
            pred = list()

            # TODO: COMENTAR SI NO ANDA y descomentar lo de abajo
            values = feat_props['values']
            # if feat_func == 'orb':
            #     values = orb_values
            # elif feat_func == 'sift':
            #     values = sift_values
            # elif feat_func == 'surf':
            #     values = surf_values
            # elif feat_func == 'rsift':
            #     values = rsift_values

            for (VAL_1, VAL_2) in values:
                pred = list()
                result_list = list()

                # Iterate over query data stored in the database
                for query in db_query:
                    logger.info("{}".format(query))
                    query_feats = db_query[query][feat_func]
                    query_pred = list()
                    match = 0

                    # Iterate over training data stored in the database
                    for image in db_museum:
                        image_feats = db_museum[image][feat_func]

                        # TODO: COMENTAR SI NO ANDA y descomentar lo de abajo
                        match = feat_props['func'](query_feats, image_feats, VAL_1, VAL_2)
                        # if feat_func == 'orb':
                        #     match = feat.compute_orb_descriptors(query_feats, image_feats, VAL_1, VAL_2)
                        # elif feat_func == 'sift':
                        #     match = feat.compute_sift_descriptor(query_feats, image_feats, VAL_1, VAL_2)
                        # elif feat_func == 'surf':
                        #     match = feat.compute_surf_descriptor(query_feats, image_feats, VAL_1, VAL_2)
                        # elif feat_func == 'rsift':
                        #     match = feat.compute_rsift_descriptor(query_feats, image_feats, VAL_1, VAL_2)

                        query_pred.append((match, ut.get_number_from_filename(image)))

                    if feat_func == 'orb' or feat_func == 'rsift':
                        query_pred.sort(key=lambda x: x[0])
                    if feat_func == 'sift' or feat_func == 'surf':
                        query_pred.sort(key=lambda x: x[0], reverse=True)

                    logger.info("{}".format(query_pred))

                    # If first value do not reach threshold
                    logger.info("{}--{}".format(query_pred[0][0], VAL_2))
                    if (feat_func == 'orb' or feat_func == 'rsift') and \
                            query_pred[0][0] > VAL_2 or \
                            (feat_func == 'sift' or feat_func == 'surf') and \
                            query_pred[0][0] < VAL_2:
                        query_pred = [-1]
                        logger.info("No matches found for {}".format(query))
                    else:
                        query_pred = [x[1] for x in query_pred[:K]]

                    pred.append([ut.get_number_from_filename(query), query_pred])

                # Sort predicted values for the query images
                pred.sort(key=lambda x: x[0])
                logger.info('Predicted values: {}'.format(pred))

                # Get only values
                result_list = [x[1] for x in pred]
                logger.info('Values: {}'.format(result_list))

                if not gts:
                    if feat_func == 'orb' or feat_func == 'rsift':
                        text = "{}_{}_nMatches_{}_thres_{}.pkl".format(space_color, feat_func, VAL_1, VAL_2)
                    elif feat_func == 'sift' or feat_func == 'surf':
                        text = "{}_{}_metric_{}_thres_{}.pkl".format(space_color, feat_func, VAL_1, VAL_2)
                    else:
                        text = "{}_{}.pkl".format(space_color, feat_func)

                    ut.bbox_to_pkl(result_list, text, folder=RESULT_DIR)

                else:
                    logger.info('GT: {}'.format(gts))

                    # Compute mAP
                    mAP = ut.mapk(gts, result_list, k=K)
                    logger.info('mAP: %.3f' % mAP)

                    mAP_text = "{0:.3f}".format(mAP)
                    if feat_func == 'orb' or feat_func == 'rsift':
                        text = "{}_{}_{}_nMatches_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
                    elif feat_func == 'sift' or feat_func == 'surf':
                        text = "{}_{}_{}_metric_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
                    else:
                        text = "{}_{}_{}.pkl".format(mAP_text, space_color, feat_func)

                    ut.bbox_to_pkl(result_list, text, folder=RESULT_DIR)
