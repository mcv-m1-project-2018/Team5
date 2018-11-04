# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import pickle

# Local modules
import utils as ut
import features as feat

# Useful directories
# TRAIN
TRAIN_MUSEUM_DIR = os.path.join('dataset', 'BBDD_W4')
TRAIN_QUERY_DIR = os.path.join('dataset', 'query_devel_W4')
GTS_DIR = os.path.join('dataset', 'w4_query_devel.pkl')
RESULT_DIR = os.path.join('pkl')

# TEST
# TRAIN_QUERY_DIR = os.path.join('dataset', 'query_test')
# GTS_DIR = os.path.join('dataset', 'no_gt.pkl')

# Pickle filenames with the training data
PICKLE_MUSEUM_DATASET = 'train_museum.pkl'
PICKLE_QUERY_DATASET = 'train_query.pkl'

# Global variables
K = 10
COLOR_SPACE_LIST = ['rgb', 'hsv']
FEATURES = {
    'orb': feat.orb,
    'sift': feat.sift,
    'surf': feat.surf,
    # 'hog': feat.hog,
}

orb_values = [(10, 700), (20, 1100)]
sift_values = [(0.4, 20), (0.4, 15)]
surf_values = [(0.5, 100), (0.4, 50)]

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
    else:
        gts = False
        logger.info("No groundtruth")

    # Check if the directory to store the results exists
    if not os.path.exists(RESULT_DIR):
        logger.info("Creating %s" % RESULT_DIR)
        os.mkdir(RESULT_DIR)

    logger.info("Loaded data")

    # Iterate over color spaces defined above
    for space_color in COLOR_SPACE_LIST:
        # Iterate over feature descriptors defined above
        for feat_func, func in FEATURES.items():
            logger.info(feat_func)
            values = ""
            pred = list()

            if feat_func == 'orb':
                values = orb_values
            if feat_func == 'sift':
                values = sift_values
            if feat_func == 'surf':
                values = surf_values

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
                        if feat_func == 'orb':
                            match = feat.compute_orb_descriptors(query_feats, image_feats, VAL_1, VAL_2)
                        if feat_func == 'sift':
                            match = feat.compute_sift_descriptor(query_feats, image_feats, VAL_1, VAL_2)
                        if feat_func == 'surf':
                            match = feat.compute_surf_descriptor(query_feats, image_feats, VAL_1, VAL_2)

                        query_pred.append((match, ut.get_number_from_filename(image)))

                    if feat_func == 'orb':
                        query_pred.sort(key=lambda x: x[0])
                    if feat_func == 'sift' or feat_func == 'surf':
                        query_pred.sort(key=lambda x: x[0], reverse=True)

                    logger.info("{}".format(query_pred))

                    # If first value do not reach threshold
                    logger.info("{}--{}".format(query_pred[0][0], VAL_2))
                    if feat_func == 'orb' and query_pred[0][0] > VAL_2 or (feat_func == 'sift' or feat_func == 'surf') and query_pred[0][0] < VAL_2:
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
                    if feat_func == 'orb':
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
                    if feat_func == 'orb':
                        text = "{}_{}_{}_nMatches_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
                    elif feat_func == 'sift' or feat_func == 'surf':
                        text = "{}_{}_{}_metric_{}_thres_{}.pkl".format(mAP_text, space_color, feat_func, VAL_1, VAL_2)
                    else:
                        text = "{}_{}_{}.pkl".format(mAP_text, space_color, feat_func)

                    ut.bbox_to_pkl(result_list, text, folder=RESULT_DIR)
