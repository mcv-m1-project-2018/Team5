# -*- coding: utf-8 -*-

import os
import glob
import shutil

from PIL import Image

def export_image_and_mask(images_dir, masks_dir, results_dir):
    """
    :param images_dir: images directory
    :param masks_dir: masks directory
    :param results_dir: destination directory
    """

    # If the directory already exists, delete it
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    # Create directory
    os.makedirs(results_dir)

    final = Image.new("RGB", (2 * 1628, 1236), "black")
    for image_name_1 in glob.glob(images_dir + "/*.jpg"):
        imagen1 = Image.open(image_name_1)
        imagen2 = Image.open(image_name_1.replace(images_dir, masks_dir).replace("jpg", "png"))
        final.paste(imagen1, (0, 0))
        final.paste(imagen2, (1628, 0))
        # final.show()
        final.save(image_name_1.replace(images_dir, results_dir))

    print("Images + masks exported")