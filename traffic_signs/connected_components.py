#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in modules
import os

# Import third party modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from skimage.measure import label, regionprops
from utils import get_files_from_dir
from utils import bboxes_to_file 

# Best values used at the moment:

# bbox_list_cc = connected_components(mask0, area_min=1000, area_max=50000, ff_min=0.5, ff_max=2, fr_min=0.5, 
#                                        plot=True, fname='', directory='')

def connected_components(mask0, area_min=None, area_max=None, ff_min=None, ff_max=None, fr_min=None, 
                         plot=False, fname=None, directory=None):

    label_image = label(mask0)
    bbox_list = []
    
    if plot==True:
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
                del(minr, minc, maxr, maxc)
                continue
                
        # Filter by form factor:
        if ff_min is not None and ff_max is not None:
            if ff_min < form_factor < ff_max:
                minr, minc, maxr, maxc = region.bbox
            else:
                del(minr, minc, maxr, maxc)
                continue
                
        # Filter by filling ratio:
        if fr_min is not None:
            if filling_ratio > fr_min:
                minr, minc, maxr, maxc = region.bbox
            else:
                del(minr, minc, maxr, maxc)
                continue
        
        bbox_list.append([minr, minc, maxr, maxc])
        
        if plot==True:

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    if fname is not None and directory is not None:
        bboxes_to_file(bbox_list, fname, directory, sign_types=None)
        
    return bbox_list
  
