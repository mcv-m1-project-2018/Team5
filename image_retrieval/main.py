# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os

# 3rd party modules
import pickle

import numpy as np
import matplotlib.pyplot as plt

# Local modules
import utils as ut

# Useful directories
RESULT_DIR = os.path.join('pkl')
TRAIN_MUSEUM_DIR = os.path.join('dataset', 'museum_set_random')
TRAIN_QUERY_DIR = os.path.join('dataset', 'query_devel_random')

# Pickle filename with the training data
PICKLE_MUSEUM_DATASET = 'train_museum.pkl'
PICKLE_QUERY_DATASET = 'train_query.pkl'

# Method number
METHOD_NUMBER = 1
METHOD_DIR = os.path.join(RESULT_DIR, 'method{number}'.format(number=METHOD_NUMBER))


# Flags
K = 10
"""
HISTOGRAM_LIST = ['global','block','pyramid']
COLOR_SPACE_LIST = ['rgb','hsv','ycbcr','lab']
DISTANCES = {
    'euclidean': ut.dist_euclidean,
    'l1': ut.dist_l1,
    'chi2': ut.dist_chi_squared,
    'histIntersection': ut.dist_hist_intersection,
    'hellinger': ut.dist_hellinger_kernel
}
"""
HISTOGRAM_LIST = ['global', 'block', 'pyramid']
COLOR_SPACE_LIST = ['hsv']
DISTANCES = {
    'euclidean': ut.dist_euclidean,
    'l1': ut.dist_l1,
    'chi2': ut.dist_chi_squared,
    'histIntersection': ut.dist_hist_intersection,
    'hellinger': ut.dist_hellinger_kernel
}


# Global variables

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    if not os.path.exists(PICKLE_MUSEUM_DATASET):
        logger.info("Creating pickle database for museum dataset...")
        db_museum = ut.create_db(TRAIN_MUSEUM_DIR)
        ut.save_db(db_museum, PICKLE_MUSEUM_DATASET)
    else:
        logger.info("Reading pickle database for museum dataset...")
        db_museum = ut.get_db(PICKLE_MUSEUM_DATASET)

    if not os.path.exists(PICKLE_QUERY_DATASET):
        logger.info("Creating pickle database for query dataset...")
        db_query = ut.create_db(TRAIN_QUERY_DIR)
        ut.save_db(db_query, PICKLE_QUERY_DATASET)
    else:
        logger.info("Reading pickle database for museum dataset...")
        db_query = ut.get_db(PICKLE_QUERY_DATASET)

    print("Museum database with {} elements".format(len(db_museum)))
    print("Query database with {} elements".format(len(db_query)))
    gt = {
        "ima_000000.jpg": "ima_000076.jpg", "ima_000001.jpg": "ima_000105.jpg", "ima_000002.jpg": "ima_000034.jpg",
        "ima_000003.jpg": "ima_000083.jpg", "ima_000004.jpg": "ima_000109.jpg", "ima_000005.jpg": "ima_000101.jpg",
        "ima_000006.jpg": "ima_000057.jpg", "ima_000007.jpg": "ima_000027.jpg", "ima_000008.jpg": "ima_000050.jpg",
        "ima_000009.jpg": "ima_000084.jpg", "ima_000010.jpg": "ima_000025.jpg", "ima_000011.jpg": "ima_000060.jpg",
        "ima_000012.jpg": "ima_000045.jpg", "ima_000013.jpg": "ima_000099.jpg", "ima_000014.jpg": "ima_000107.jpg",
        "ima_000015.jpg": "ima_000044.jpg", "ima_000016.jpg": "ima_000065.jpg", "ima_000017.jpg": "ima_000063.jpg",
        "ima_000018.jpg": "ima_000111.jpg", "ima_000019.jpg": "ima_000092.jpg", "ima_000020.jpg": "ima_000067.jpg",
        "ima_000021.jpg": "ima_000022.jpg", "ima_000022.jpg": "ima_000087.jpg", "ima_000023.jpg": "ima_000085.jpg",
        "ima_000024.jpg": "ima_000013.jpg", "ima_000025.jpg": "ima_000039.jpg", "ima_000026.jpg": "ima_000103.jpg",
        "ima_000027.jpg": "ima_000006.jpg", "ima_000028.jpg": "ima_000062.jpg", "ima_000029.jpg": "ima_000041.jpg"
    }

    for hist_type in HISTOGRAM_LIST:
        for space_color in COLOR_SPACE_LIST:
            for dist_func in DISTANCES:
                fun = DISTANCES[dist_func]
                list_of_lists = []
                query_gt_list = []
                query_list = []

                for query in db_query:
                    query_list.append(query)
                    query_gt_list.append([gt[query]])
                    query_h = db_query[query][hist_type][space_color]
                    result_list = []
                    for image in db_museum:
                        image_h = db_museum[image][hist_type][space_color]
                        metric = fun(np.array(query_h), np.array(image_h))
                        result_list.append((metric, image))

                    if dist_func == "hellinger" or dist_func == "histIntersection":
                        result_list.sort(key=lambda tup: tup[0], reverse=True)
                    else:
                        result_list.sort(key=lambda tup: tup[0], reverse=False)

                    result_list = [x[1] for x in result_list[:K]]
                    list_of_lists.append(result_list)

                # Guardar lista
                print(list_of_lists)

                # Calcular mAP
                mAP = ut.mapk(query_gt_list, list_of_lists, k=K)
                mAP_text = "{0:.2f}".format(mAP)

                # Exportar pkl
                ut.bbox_to_pkl(
                    list_of_lists,
                    "{}_{}_{}_{}.pkl".format(mAP_text, hist_type, space_color, dist_func),
                    folder=RESULT_DIR
                )