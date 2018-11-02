# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import pickle

# 3rd party modules
import numpy as np

# Local modules
import utils as ut
import features as feat

# Useful directories
TRAIN_MUSEUM_DIR = os.path.join('dataset', 'BBDD_W4')
TRAIN_QUERY_DIR = os.path.join('dataset', 'query_devel_W4')
GTS_DIR = os.path.join('dataset', 'w4_query_devel.pkl')
RESULT_DIR = os.path.join('pkl')

# Pickle filenames with the training data
PICKLE_MUSEUM_DATASET = 'train_museum.pkl'
PICKLE_QUERY_DATASET = 'train_query.pkl'

# Global variables
K = 10
COLOR_SPACE_LIST = ['rgb', 'hsv']
FEATURES = {
    # 'orb': feat.orb,
    'sift': feat.sift,
    # 'surf': feat.surf,
    # 'hog': feat.hog,
}
# Orb: VAL_1 = 10 y VAL_2 = 500 y 1000
# VAL_1 = 10
# VAL_2 = 500
# SIFT: VAL_1 = 0.5 y VAL_2 = 50 y 100
VAL_1 = 0.5
VAL_2 = 50
# SURF: VAL_1 = 0.5 y VAL_2 = 5 y 10
# VAL_1 = 0.5
# VAL_2 = 5

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    logger.info("Starting Museum Painting Retrieval")

    # Check for training database file
    if not os.path.exists(PICKLE_MUSEUM_DATASET):
        logger.info("Creating pickle database for museum dataset...")
        db_museum = ut.create_db(TRAIN_MUSEUM_DIR, FEATURES)
        ut.save_db(db_museum, PICKLE_MUSEUM_DATASET)
    else:
        logger.info("Reading pickle database for museum dataset...")
        db_museum = ut.get_db(PICKLE_MUSEUM_DATASET)

    # Check for query database file
    if not os.path.exists(PICKLE_QUERY_DATASET):
        logger.info("Creating pickle database for query dataset...")
        db_query = ut.create_db(TRAIN_QUERY_DIR, FEATURES)
        ut.save_db(db_query, PICKLE_QUERY_DATASET)
    else:
        logger.info("Reading pickle database for query dataset...")
        db_query = ut.get_db(PICKLE_QUERY_DATASET)

    # Ground truth for query images
    gts = list()
    if os.path.exists(GTS_DIR):
        with open(os.path.join(GTS_DIR), 'rb') as f:
            gts = pickle.load(f)
            logger.info("gt = {}".format(gts))
            # Change format to calculate mAP
            gts = [x[1] for x in gts]

    # Check if the directory to store the results exists
    if not os.path.exists(RESULT_DIR):
        logger.info("Creating %s" % RESULT_DIR)
        os.mkdir(RESULT_DIR)

    logger.info("Loaded data")

    # Iterate over color spaces defined above
    for space_color in COLOR_SPACE_LIST:
        # Iterate over feature descriptors defined above
        for feat_func, func in FEATURES.items():
            pred = list()

            # Iterate over query data stored in the database
            for query in db_query:
                query_feats = db_query[query][feat_func]
                query_pred = list()
                match = False

                # Iterate over training data stored in the database
                for image in db_museum:
                    image_feats = db_museum[image][feat_func]
                    if feat_func == 'orb':
                        match = feat.compute_orb_descriptors(query_feats, image_feats, VAL_1, VAL_2)
                    if feat_func == 'sift':
                        metric = feat.compute_sift_descriptor(query_feats, image_feats, VAL_1, VAL_2)
                    if feat_func == 'surf':
                        metric = feat.compute_surf_descriptor(query_feats, image_feats, VAL_1, VAL_2)

                    # If the descriptors of the train image matches with those from the query
                    # image, add the train image value to the list
                    if match:
                        query_pred.append(ut.get_number_from_filename(image))
                        logger.info("Success --> Query:{} ----- GT:{}".format(query, image))

                if not match:
                    logger.info("No matches found for {}".format(query))
                # If no success images found
                if len(query_pred) == 0:
                    query_pred.append(-1)

                pred.append([ut.get_number_from_filename(query), query_pred])

            # Sort predicted values for the query images
            pred.sort(key=lambda x: x[0])
            logger.info('Predicted values: {}'.format(pred))

            # Get only values
            result_list = [x[1] for x in pred]
            logger.info('Values: {}'.format(result_list))

            # Compute mAP
            mAP = ut.mapk(gts, result_list, k=K)
            logger.info('mAP: %.3f' % mAP)

            mAP_text = "{0:.3f}".format(mAP)
            if feat_func == 'orb':
                text = "{}_{}_{}_nMatches_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
            elif feat_func == 'sift' or feat_func == 'surf':
                text = "{}_{}_{}_metric_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
            else:
                text = "{}_{}_{}.pkl".format(mAP_text, space_color, feat_func)

            ut.bbox_to_pkl(result_list, text, folder=RESULT_DIR)
