# COMP 4102 FINAL PROJECT
# TITLE: License Detector
# Group: Rayhanne Bournas, Nicolas Mak-Fan, Sahil Kapal, Umar Farooq

from typing import Any, Union
import imutils
import numpy as np
import cv2
import scipy.stats as st
from numpy.core._multiarray_umath import ndarray
from skimage.exposure import rescale_intensity

# Nick's part: Apply thresholding to find the contours of the image
#once the contours are found, return the extreme
#points ((x,y) coordinates of the contour edges)

#license plate cropping doesn't work for: 2, 7, 9, and 10
#plate 1 cropping only works when resized
#plate 8 works but the edges are very faded

file = 'Images/plate1.jpg'

img = cv2.imread(file)
img = cv2.resize(img, (620, 480))

gray = cv2.imread(file, 0)
gray = cv2.resize(gray, (620, 480))

# Blur to reduce noise while preserving edges
gray = cv2.bilateralFilter(gray, 11, 17, 17)


# Convolution applies the kernel to the image to produce a new image
def _convolution(img0, ker):
    (imgH, imgW) = img0.shape[:2]
    (kerH, kerW) = ker.shape[:2]

    pad: Union[int, Any] = (kerW - 1) // 2
    image = cv2.copyMakeBorder(img0, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    result: ndarray = np.zeros((imgH, imgW), dtype="float32")

    for y in np.arange(pad, imgH + pad):
        for x in np.arange(pad, imgW + pad):
            r = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (r * ker).sum()
            result[y - pad, x - pad] = k

    result = rescale_intensity(result, in_range=(0, 255))
    result = (result * 255).astype("uint8")

    return result


# Create gaussian kernel to loop through the image and cause a
# blur at each pixel
def _gaussian_kernel(k, sig):
    x = np.linspace(-sig, sig, k + 1)
    ker1d = np.diff(st.norm.cdf(x))
    ker2d = np.outer(ker1d, ker1d)

    return ker2d / ker2d.sum()


kernel_y = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]], dtype="float32")

kernel_x = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]], dtype="float32")


def get_threshold(img):
    return cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _license_plate_detector(img0, sigma):
    gaussian = _convolution(img0, _gaussian_kernel(3, sigma))

    # Pass gaussian filter into threshold built-in func
    _, thresh = cv2.threshold(gaussian, 120, 255, 1)
    thresh = np.uint8(thresh)

    # fill some holes in the pic by increasing the white
    # region. Removes noise without shrinking img
    thresh = cv2.dilate(thresh, None)
    # Keeps the foreground of the img white so plate is more
    # easily distinguishable
    thresh = cv2.erode(thresh, None)

    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours based on the threshold
    # 2nd parametre is returns contours without caring about hierarchical relationships
    # 3rd parametre spepcifies how the the contour is shown (leaves only end points)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    LPCnt = None

    # Loops over the contours and approximates each one
    # to determine which are the ones we need
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * p, True)

        # Once a contour with 4 points is found
        # we assume it is the contour we're looking for
        if len(approx) == 4:
            LPCnt = approx
            break

    # display is none are found
    if LPCnt is None:
        detected = 0
        print("No contour detected")

    # Draw contour over the license plate
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [LPCnt], -1, (0, 255, 0), 3)

    # based on the contour that was found, we return
    # the 4 extreme point (vertex) coordinates (x,y)
    rec = cv2.minAreaRect(LPCnt)
    vtx = cv2.boxPoints(rec)
    for p in vtx:
        pt = (p[0], p[1])
        print(pt)
        cv2.circle(img, pt, 5, (200, 0, 0), 2)

    cv2.imshow('License Plate', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sahil's part: extract license plate from image:
    # Obtain coordinates of plate:
    topLeft_x = int(vtx[1][0])
    topLeft_y = int(vtx[1][1])

    bottomLeft_x = int(vtx[0][0])
    bottomLeft_y = int(vtx[0][1])

    topRight_x = int(vtx[2][0])

    # get width and height
    plate_height = bottomLeft_y - topLeft_y
    plate_width = topRight_x - topLeft_x

    # get crop image
    crop_img = img[topLeft_y:topLeft_y + plate_height, bottomLeft_x:bottomLeft_x + plate_width]
    cv2.imshow("Cropped License Plate", crop_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # increase plate size:
    large_plate = cv2.resize(crop_img, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # sharpen cropped image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply kernel to image
    # blurred = cv2.GaussianBlur(large_plate, (5, 5), 0)
    blurred = _convolution(large_plate, _gaussian_kernel(3, sigma))
    sharpened_image = cv2.filter2D(blurred, -1, kernel)
    img1 = get_threshold(sharpened_image)

    cv2.imshow("Blurred_image", img1)
    # cv2.imshow("sharpened plate", sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


_license_plate_detector(gray, 1)
