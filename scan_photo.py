import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime


outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


def load_image(file_name: str):
    """
    :param file_name:
    :return:  -> (int, int), double, double
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def scan_image(img_name, threshold=100, s_min=5, s_max=100):
    image = load_image(img_name)
    # make CSV file name from these params
    # time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    log_name = img_name.strip("./").lower().split(".")[0]
    logfile = open(f'{log_name}.csv', 'w')
    logfile.write('y, x, r , b\n')

    # threshold
    th, th_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # find contours
    contours = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # filter by area
    for c in contours:
        area = cv2.contourArea(c)
        if s_min < area < s_max:
            # cv2.drawContours(img, [c], -1, (255, 0, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(th_img, (cx, cy), 40, (255, 0, 0), 3)
            # cv2.putText(img, "Centre", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
            logfile.write(f'{cy}, {cx}, {area}, {image[cy,cx] / 255}\n')

    # save images
    images = np.concatenate((image, th_img), axis=1)
    cv2.imwrite(f'{log_name}_th.jpg', images)
    # return the threshold image
    return th_img


def find_local_max(image):
    h = image.shape[0]
    w = image.shape[1]
    local_max = 0
    y1, x1 = 0, 0
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            if image[y, x] >= local_max:
                local_max = image[y, x]
                y1, x1 = y, x
    return y1, x1
