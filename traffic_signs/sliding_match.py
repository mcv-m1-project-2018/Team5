import cv2
import os
import matplotlib.pyplot as plt
from utils import sliding_window


IMAGE_DIR = 'results'
IMAGE = os.path.join(IMAGE_DIR, '01.002429.png')

# load the image and define the window width and height
image = cv2.imread(IMAGE)
(winW, winH) = (128, 128)

# loop over the sliding window
for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    # Aqui es donde la magia ocurre


    # since we do not have a classifier, we'll just draw the window
    clone = window.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)