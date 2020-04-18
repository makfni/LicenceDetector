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
from scipy import misc
import sys
# Nick's part: Apply thresholding to find the contours of the image
#once the contours are found, return the extreme
#points ((x,y) coordinates of the contour edges)

#license plate cropping doesn't work for: 2, 7, 9, and 10
#plate 1 cropping only works when resized
#plate 8 works but the edges are very faded






# return the path of the image
def image_path():
    # Get the image name from the argument
    image_arg = str(sys.argv[2])
    # Get the initail path of the image
    path = 'Images/'
    # Add the full path
    path = path + image_arg

    return path
    

# Scale::For the resize of the license plate
def get_scale():
    # The scale is argument 1
    scale = float(sys.argv[1])
    return scale

# Get the path from the user
file = image_path()

img = cv2.imread(file)
# img = cv2.resize(img, (620, 480))

gray = cv2.imread(file, 0)
# gray = cv2.resize(gray, (620, 480))

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

def test_text(text, digits, letters):
    # Create an empty string
    license_plate = ''
    for i in(text):
        # Only pick up the characters or digits
        if (i in digits) or (i in letters):
            license_plate = license_plate + i
        # unecessary characters that the ocr picked up randomly
        # ignore them
        else:
            license_plate = license_plate + ' '
    # return the correct text
    return license_plate

def get_threshold(img):
    return cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def sharpen_image(plate_img_refined):
    # Blur then filter
    gaussian_3 = cv2.GaussianBlur(plate_img_refined, (9, 9), 10.0)
    plate_img_refined = cv2.addWeighted(plate_img_refined, 1.5, gaussian_3, -0.5, 0, plate_img_refined)


    return plate_img_refined

def gen_information():

    # Create two empty arrays to hold the data for both the letters and the character
    digits = []
    letters = []

    # Get the valid numbers of a plate ie 0-9
    for digit in range(10):
        digits.append(chr(ord('1')+digit))

    # Get all the character of a plate A-Z
    for letter in range(25, -1, -1):
        letters.append(chr(ord('A')+letter))

    return digits, letters

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

    # Print all the contour points
    print("\n\n")
    print("-----------------------------------Contour Points-----------------------------------")

    rec = cv2.minAreaRect(LPCnt)
    vtx = cv2.boxPoints(rec)
    counter = 1
    for p in vtx:
        pt = (p[0], p[1])
        print("Contour point", counter, ":",  pt)
        cv2.circle(img, pt, 5, (200, 0, 0), 2)
        counter = counter + 1

    print("-----------------------------------------------------------------------------------")


    cv2.imshow('License Plate', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # License Plate Extraction
    crop_img = _license_extraction(vtx)

    return crop_img

def _license_extraction(vtx):
    # Sahil's part: extract license plate from image:
    # Obtain coordinates of plate and crop the image:
    minX = 10000
    minY = 10000
    for v in vtx:
        if v[1] < minX:
            minX = v[1]

        if v[0] < minY:
            minY = v[0]

    maxX = 0
    maxY = 0
    for v in vtx:
        if v[1] > maxX:
            maxX = v[1]

        if v[0] > maxY:
            maxY = v[0]

    minX = int(minX)
    minY = int(minY)
    maxX = int(maxX)
    maxY = int(maxY)
    coords = {
        "top_left": [minX, minY],
        "bottom_left": [maxX, minY],
        "top_right": [minX, maxY],
        "bottom_right": [maxX, maxY]
    }

    # get crop image
    crop_img = img[coords["top_left"][0]:coords["bottom_left"][0], coords["bottom_left"][1]:coords["bottom_right"][1]]

    return crop_img


# Rayhane Part 
# take in the license plate image
# Resize it with a specific scale
# Sharpen the scaled image
# Apply the character recognisiton
# Test the character recognisition
# Return the characters to the user
def character_recognistion(cropped_image):

    # Get the scale from the user
    scale = get_scale()

    # Resize the image with the scale
    resized_cropped_image = resize_img(cropped_image, scale)

    # Display it to the user
    cv2.imshow("Resized Cropped Image", resized_cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sharpen the image
    sharpened_image = sharpen_image(resized_cropped_image)

    # Display the sharpened image to the user
    cv2.imshow("Sharepend resized image", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get the letters and digits
    digits, letters = gen_information()


    print("\n\n\n")
    print("----------------------------------Testing purpose-----------------------------------")
    print("digits:", digits)
    print("letters:", letters)
    print("------------------------------------------------------------------------------------")

    # Apply the OCR library to read the characters of the plate
    text = pytesseract.image_to_string(sharpened_image)

    # Get the text with the testing
    tested_text = test_text(text, digits, letters)

    # Display the original text to the user
    print("\n\n\n")
    print("-------------------------------Character Recognistion-------------------------------")
    print("Text without testing:   ", text)
    print("\n")
    print("Text with testing:      ", tested_text)
    print("------------------------------------------------------------------------------------")


# Main function
def main():

    # Get the path from the user
    file = image_path()

    # Open the image with color, and open it in grayscale
    img = cv2.imread(file)
    gray = cv2.imread(file, 0)

    # Blur to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # This will return the plate porsion of the image
    plate_img = _license_plate_detector(gray, 1)

    # Character recognistion part
    character_recognistion(plate_img)




# Call the main
main()