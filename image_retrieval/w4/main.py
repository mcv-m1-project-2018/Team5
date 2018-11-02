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
RESULT_DIR = os.path.join('pkl')
TRAIN_MUSEUM_DIR = os.path.join('dataset', 'BBDD_W4')
TRAIN_QUERY_DIR = os.path.join('dataset', 'query_devel_W4')
GT_DIR = os.path.join("w4_query_devel.pkl")

# Pickle filename with the training data
PICKLE_MUSEUM_DATASET = 'train_museum.pkl'
PICKLE_QUERY_DATASET = 'train_query.pkl'

# Method number
METHOD_NUMBER = 1
METHOD_DIR = os.path.join(RESULT_DIR, 'method{number}'.format(number=METHOD_NUMBER))

# Flags
K = 10
#HISTOGRAM_LIST = ['global','block','pyramid']
#COLOR_SPACE_LIST = ['rgb','hsv','ycbcr','lab']
COLOR_SPACE_LIST = ['rgb']
FEATURES = {
    'orb': feat.orb,
    'sift': feat.sift,
    # 'surf': feat.surf,


    #'harris': feat.harris,
    #'log': feat.lap_of_gauss,
    #'dog': feat.dif_of_gauss,
    #'doh': feat.det_of_hessi,
    #'daisy': feat.daisy,
    #'lbp': feat.lbp
    #'hog': feat.hog
}

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    logger.info("START")

    if not os.path.exists(PICKLE_MUSEUM_DATASET):
        logger.info("Creating pickle database for museum dataset...")
        db_museum = ut.create_db(TRAIN_MUSEUM_DIR, FEATURES)
        ut.save_db(db_museum, PICKLE_MUSEUM_DATASET)
    else:
        logger.info("Reading pickle database for museum dataset...")
        db_museum = ut.get_db(PICKLE_MUSEUM_DATASET)

    if not os.path.exists(PICKLE_QUERY_DATASET):
        logger.info("Creating pickle database for query dataset...")
        db_query = ut.create_db(TRAIN_QUERY_DIR, FEATURES)
        ut.save_db(db_query, PICKLE_QUERY_DATASET)
    else:
        logger.info("Reading pickle database for query dataset...")
        db_query = ut.get_db(PICKLE_QUERY_DATASET)

    gt = []
    if os.path.exists(GT_DIR):
        with open(os.path.join(GT_DIR), 'rb') as p:
            gt = pickle.load(p)
            logger.info("gt = {}".format(gt))
            # Change format to calculate mAP
            gt = [x[1] for x in gt]

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    logger.info("LOADED DATA")

    print("Museum database with {} elements".format(len(db_museum)))
    print("Query database with {} elements".format(len(db_query)))

    for space_color in COLOR_SPACE_LIST:
        for feature_function in FEATURES:
            func = FEATURES[feature_function]
            total_list = []

            for query in db_query:
                query_h = db_query[query][feature_function]
                query_list = []
                metric = False

                for image in db_museum:
                    image_h = db_museum[image][feature_function]
                    if feature_function == 'orb':
                        metric = feat.compute_orb_descriptors(query_h, image_h, 10, 500)
                    if feature_function == 'sift':
                        metric = feat.compute_sift_descriptor(query_h, image_h, 0.5, 50)

                    # if SURF

                    if metric:
                        query_list.append(int(image.split("_")[1].split(".")[0]))
                        logger.info("Success --> Query:{} ----- GT:{}".format(query, image))

                # If no success images found
                if len(query_list) == 0:
                    query_list.append(-1)

                total_list.append([int(query.split("_")[1].split(".")[0]), query_list])
                #print(total_list)


        total_list.sort()
        logger.info(total_list)

        result_list = [x[1] for x in total_list]
        logger.info(result_list)

        mAP = ut.mapk(gt, result_list, k=K)
        logger.info(mAP)

        mAP_text = "{0:.3f}".format(mAP)
        ut.bbox_to_pkl(
            result_list,
            "{}_{}_{}.pkl".format(mAP_text, space_color, feature_function),
            folder=RESULT_DIR
        )


# for space_color in COLOR_SPACE_LIST:
    #     for dist_func in FEATURES:
    #         fun = FEATURES[dist_func]
    #         list_of_lists = []
    #         query_gt_list = []
    #         query_list = []
    #
    #         for query in db_query:
    #             query_list.append(query)
    #             query_gt_list.append([gt[query]])
    #             query_h = db_query[query][hist_type][space_color]
    #             logger.info("{}-{}-{}-{}".format(query, hist_type, space_color, dist_func))
    #             result_list = []
    #             for image in db_museum:
    #                 image_h = db_museum[image][hist_type][space_color]
    #                 metric = fun(np.array(query_h), np.array(image_h))
    #                 result_list.append((metric, image))
    #
    #             if dist_func == "hellinger" or dist_func == "histIntersection":
    #                 result_list.sort(key=lambda tup: tup[0], reverse=True)
    #             else:
    #                 result_list.sort(key=lambda tup: tup[0], reverse=False)
    #
    #             result_list = [x[1] for x in result_list[:K]]
    #             list_of_lists.append(result_list)
    #
    #         # print(list_of_lists)
    #
    #         # Calcular mAP
    #         mAP = ut.mapk(query_gt_list, list_of_lists, k=K)
    #
    #         # Exportar pkl
    #         mAP_text = "{0:.3f}".format(mAP)
    #
    #         ut.bbox_to_pkl(
    #             list_of_lists,
    #             "{}_{}_{}_{}.pkl".format(mAP_text, hist_type, space_color, dist_func),
    #             folder=RESULT_DIR
    #         )
    #
    #         # print(query_list)
    #         ut.bbox_to_pkl(
    #             query_list,
    #             "query_list.pkl",
    #             folder=RESULT_DIR
    #         )
