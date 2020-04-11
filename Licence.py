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
import os
import pytesseract

# Nick's part: Apply thresholding to find the contours of the image
#once the contours are found, return the extreme
#points ((x,y) coordinates of the contour edges)

#license plate cropping doesn't work for: 2, 7, 9, and 10
#plate 1 cropping only works when resized
#plate 8 works but the edges are very faded

file = 'Images/plate4.jpg'

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

# This function is used to take the small image
# Of the plate alone and resize it bigger so that one
# Can read the characters of the image
def resize_img(img, scale):

    # Resize the image according to the scale
    new_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    return new_img

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

    # if detected == 1:
    #     continue
        # cv2.drawContours(img, [LPCnt], -1, (255, 255, 255), 3)

    # based on the contour that was found, we return
    # the 4 extreme point (vertex) coordinates (x,y)
    rec = cv2.minAreaRect(LPCnt)
    vtx = cv2.boxPoints(rec)
    for p in vtx:
        pt = (p[0], p[1])
        print(pt)
        # cv2.circle(img, pt, 5, (200, 0, 0), 2)

    # cv2.imshow('License Plate', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    # cv2.imshow("Cropped License Plate", crop_img)
#     small_img = cv2.resize(char_img,(10,10))

    # crop_img = cv2.resize(crop_img, (108, 21))
    # gray_scale_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("license_plate.jpg", crop_img)
    # cv2.imshow("sharpened plate", crop_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return crop_img


# This will return the plate porsion of the image
plate_img = _license_plate_detector(gray, 1)
# plate_img_refined = cv2.resize(plate_img, (int(plate_img.shape[1]*1.5), int(plate_img.shape[0]*1.5)))
scale = 1.5
plate_img_refined = resize_img(plate_img, scale)
cv2.imshow('Plate_img_refined', plate_img_refined)
cv2.waitKey(0)
cv2.destroyAllWindows()
# apply the OCR library to read the characters of the plate
text = pytesseract.image_to_string(plate_img_refined)
# For testing purposes
# the digits represent all ten numbers that the plate may have
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# letters contains all the capital letters that the plate may contain
# Main purpose is to remove all unecessary characters that the OCR
# May pick up
letters = [chr(ord('A') + i) for i in range(25, -1, -1)]
print("This is the letters: ", letters)
print("These are the digits: ", digits)
# Create an empty string
license_plate = ''
for i in(text):
    # Only pick up the characters or digits
    if (i in digits) or (i in letters):
        license_plate = license_plate + i
    else:
        license_plate = license_plate + ' '
print(license_plate)
