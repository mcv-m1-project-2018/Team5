# -*- coding: utf-8 -*-


import numpy as np;
import cv2


def get_text_area(img, image_name):
    """

    :param img: color image read with openCV
    :return:
    """


#for path_image in glob.glob("dataset/w5_BBDD_random/*.jpg"):
 #   image_name = path_image.split("/")[-1]
  #  print(image_name)

    # Read image
    #img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = img.shape

    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=1)

    ret, thresh1 = cv2.threshold(laplacian, 200, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)
    im_th = thresh1 | thresh2

    # IMPORTANT: Delete one vertical line to not get the full frame
    #im_th[0:img_size[0], int(img_size[1] * 0.4)] = 0

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # draw line
    up_limit = int(img_size[0] * 0.29)
    down_limit = int(img_size[0] * 0.705)
    # cv2.line(erosion,(0,up_limit),(img_size[1],up_limit),(255,255,255),5)
    # cv2.line(erosion,(0,down_limit),(img_size[1],down_limit),(255,255,255),5)
    im_floodfill_inv[up_limit:down_limit, 0:img_size[1]] = 0

    # Count which part has more positive values
    top_positive_values = cv2.countNonZero(im_floodfill_inv[0:up_limit, 0:img_size[1]])
    bottom_positive_values = cv2.countNonZero(im_floodfill_inv[down_limit:img_size[0], 0:img_size[1]])
    if top_positive_values > bottom_positive_values:
        im_floodfill_inv[down_limit:img_size[0], 0:img_size[1]] = 0
    else:
        im_floodfill_inv[0:up_limit, 0:img_size[1]] = 0

    y_sum = cv2.reduce(im_floodfill_inv, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    for n, row in enumerate(y_sum):
        if row < 255 * 60:
            im_floodfill_inv[n, 0:img_size[1]] = 0

    x_sum = cv2.reduce(im_floodfill_inv, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    for x in x_sum:
        for n, col in enumerate(x):
            if col < 255 * 8:
                im_floodfill_inv[0:img_size[0], n] = 0

    # erosion
    kernel = np.ones((1, 1), np.uint8)
    # erosion = cv2.erode(im_floodfill_inv,kernel,iterations = 1)
    # erosion = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_OPEN, kernel)
    erosion = remove_isolated_pixels(im_floodfill_inv, 100)

    # rectangle im_floodfill_inv
    contours = cv2.findContours(im_floodfill_inv, 1, 2)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im_floodfill_inv, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # rectangle erosion
    #contours = cv2.findContours(erosion, 1, 2)
    #cnt = contours[0]
    #x, y, w, h = cv2.boundingRect(cnt)
    #cv2.rectangle(erosion, (x, y), (x + w, y + h), (255, 255, 255), 5)

    # plot
    #fig, ax = plt.subplots(figsize=(15, 8))
    #plt.subplot(121)
    #plt.imshow(im_floodfill_inv)
    #plt.subplot(122)
    #plt.imshow(erosion)

    #plt.show()
    # fig.savefig("adsf.png")

    cv2.imwrite('pruebas_9/' + image_name + '.png', im_floodfill_inv)
    return (x-15, y-10, x+w+15, y+h+10)


def remove_isolated_pixels(image, threshold):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] < threshold:
            new_image[labels == label] = 0

    return new_image

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_iou(candidates, annotations):

    TP = 0
    FP = 0
    for n, candidate in enumerate(candidates):
        annotation = annotations[n]
        if bbox_iou(candidate, annotation) > 0.5:
            TP += 1
            #print("{}: TRUE".format(n))
        else:
            FP += 1
            #print("{}: FALSE --------------------------------------------".format(n))

    FN = len(annotations) - TP - FP

    return TP, FN, FP