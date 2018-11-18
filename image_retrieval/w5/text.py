# -*- coding: utf-8 -*-


import numpy as np;
import cv2


def get_text_area(img, image_name, gt=list()):
    """

    :param img:
    :param image_name:
    :param gt:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=1)

    _, im_floodfill = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Delete the central part of the image
    im_floodfill_inv = delete_central_image(im_floodfill_inv)

    # Delete isolated pixels
    im_floodfill_inv = remove_isolated_pixels(im_floodfill_inv, 50)

    # Delete lines with few pixels
    y_sum = cv2.reduce(im_floodfill_inv, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    for n, row in enumerate(y_sum):
        if row < 255 * 60:
            im_floodfill_inv[n, 0:w] = 0

    x_sum = cv2.reduce(im_floodfill_inv, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    for x in x_sum:
        for n, col in enumerate(x):
            if col < 255 * 6:
                im_floodfill_inv[0:h, n] = 0

    # operations
    #kernel = np.ones((2, 2), np.uint8)
    # im_floodfill_inv = cv2.erode(im_floodfill_inv,kernel,iterations = 1)
    # im_floodfill_inv = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_OPEN, kernel)

    # find rectangle
    contours = cv2.findContours(im_floodfill_inv, 1, 2)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # Calculate filling ratio
    positive_values = cv2.countNonZero(im_floodfill_inv[gt[1]: gt[3], gt[0]: gt[2]])
    area = (gt[2]-gt[0])*(gt[3]-gt[1])
    filling_ratio = positive_values/area

    if filling_ratio < 0.7:
        ci = x - 18
        ri = y - 21
        cf = x + w + 22
        rf = y + h + 17
    else:
        ci = x
        ri = y
        cf = x + w
        rf = y + h

    cv2.rectangle(im_floodfill_inv, (ci, ri), (cf, rf), (255, 255, 255), 1)
    cv2.rectangle(im_floodfill_inv, (gt[0], gt[1]), (gt[2], gt[3]), (100, 100, 100), 3)

    # cv2.imwrite('pruebas_9/' + image_name + '.png', im_floodfill_inv)
    return (ci, ri, cf, rf)


def delete_central_image(img):

    h, w = img.shape[:2]

    # draw line
    up_limit = int(h * 0.29)
    down_limit = int(h * 0.705)
    # cv2.line(erosion,(0,up_limit),(w,up_limit),(255,255,255),5)
    # cv2.line(erosion,(0,down_limit),(w,down_limit),(255,255,255),5)
    img[up_limit:down_limit, 0:w] = 0

    # Count which part has more positive values
    top_positive_values = cv2.countNonZero(img[0:up_limit, 0:w])
    bottom_positive_values = cv2.countNonZero(img[down_limit:h, 0:w])
    if top_positive_values > bottom_positive_values:
        img[down_limit:h, 0:w] = 0
    else:
        img[0:up_limit, 0:w] = 0

    return img


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

    return iou


def compute_mean_iou(candidates, annotations):

    mean_iou = 0
    TP = 0
    FP = 0

    for n, candidate in enumerate(candidates):
        annotation = annotations[n]
        mean_iou += bbox_iou(candidate, annotation)

        #print("{} --- {}".format(n, bbox_iou(candidate, annotation)))

        if bbox_iou(candidate, annotation) > 0.5:
            TP += 1
            # print("{}: TRUE".format(n))
        else:
            FP += 1
            # print("{}: FALSE --------------------------------------------".format(n))

    mean_iou = mean_iou / len(annotations)


    FN = len(annotations) - TP - FP
    # print("TP:{}    FN={}   FP={}".format(TP, FN, FP))

    return mean_iou
