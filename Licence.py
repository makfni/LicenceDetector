from typing import Any, Union
import imutils
import numpy as np
import cv2
import scipy.stats as st
from numpy.core._multiarray_umath import ndarray
from skimage.exposure import rescale_intensity

file = 'Images/blurr_plate10.jpg'

img = cv2.imread(file)
img = cv2.resize(img, (620, 480))

gray = cv2.imread(file, 0)
gray = cv2.resize(gray, (620, 480))

# # Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)


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
    result = (result * 255).astype(np.uint8)

    return result


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

'''
This calculates the edges along the x axis
'''


def sobel_filter_xaxis(img):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], np.float32)

    Ix = _convolution(img, kx)

    return Ix


'''
This calculates the edges along the y axis
'''


def sobel_filter_yaxis(img):
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], np.float32)

    Iy = _convolution(img, ky)

    return Iy


'''
This function simply normalizes the current matrix
'''


def normalize(value):
    value = value / value.max() * 255
    return value


'''
Sobel filter simply combines both sobel-y and sobel-x
to get the gradient, and calculate the hypothesis
the function returns both the gradient and theta
'''


def sobel_filter(Ix, Iy):
    Gxy = np.hypot(Ix, Iy)
    Gxy = normalize(Gxy)

    theta = np.arctan2(Iy, Ix)

    return Gxy, theta


# def _sobel_img(img1: object, img2: object) -> object:
# #     img_cpy = np.zeros(img1.shape)
# #
# #     for i in range(img1.shape[0]):
# #         for j in range(img1.shape[1]):
# #             q = (img1[i][j] ** 2 + img2[i][j] ** 2) ** (1 / 2)
# #             if q > 90:
# #                 img_cpy[i][j] = 255
# #             else:
# #                 img_cpy[i][j] = 0
# #
# #     return img_cpy


def _non_max_suppression(img, D):
    m, n = img.shape
    Z = np.zeros((m, n), dtype=np.int32)
    angle: float = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


weak_pixel = 30
strong_pixel = 200
lowThreshold = 0.05
highThreshold = 0.15


def threshold(img, highThres, lowThres):
    highThres = img.max() * highThres
    lowThres = highThres * lowThres

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= highThres)

    weak_i, weak_j = np.where((img <= highThres) & (img >= lowThres))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def get_threshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def hysteresis(img):
    M, N = img.shape
    weak = weak_pixel
    strong = strong_pixel

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def _my_edge_filter(img0, sigma):
    gaussian = _convolution(img0, _gaussian_kernel(3, sigma))

    # This is for the second part, the sobel filter
    Ix = sobel_filter_xaxis(gaussian)
    Iy = sobel_filter_yaxis(gaussian)

    sobel_values = sobel_filter(Ix, Iy)

    # Final part, this is the non_max refinement
    gradient_matrix = sobel_values[0]
    tetha_value = sobel_values[1]

    non_max = _non_max_suppression(gradient_matrix, tetha_value)

    thresh = threshold(non_max, highThreshold, lowThreshold)
    # gaussian = cv2.GaussianBlur(img0, (5, 5), 0)
    # _, thresh = cv2.threshold(gaussian, 120, 255, 1)
    thresh = np.uint8(thresh)
    thresh = cv2.dilate(thresh, None)  # fill some holes
    thresh = cv2.dilate(thresh, None)
    thresh = cv2.erode(thresh, None)  # dilate made our shape larger, revert that
    thresh = cv2.erode(thresh, None)

    canny = hysteresis(thresh.astype(np.uint8))

    canny = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    # thresh = thresh.astype(np.uint8)
    # cv2.imwrite('edge_dection.jpg', non_max)
    # img1 = cv2.imread('edge_dection.jpg', cv2.IMREAD_GRAYSCALE)
    #
    cv2.imshow('Edge detection', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours based on Edges
    contours = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    # sort contours based on their area keeping minimum
    # required area as '30' (anything smaller than this will not be considered)
    NumberPlateCnt = None  # we currently have no Number plate contour
    # # loop over our contours to find the best possible approximate contour of number plate
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            break

    if NumberPlateCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    # if detected == 1:
    #     cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

    rc = cv2.minAreaRect(NumberPlateCnt)
    box = cv2.boxPoints(rc)
    for p in box:
        pt = (p[0], p[1])
        print(pt)
        cv2.circle(img, pt, 5, (200, 0, 0), 2)

    cv2.imshow("Licence Plate Detected", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Sahil's part: extract license plate from image:

    # Obtain coordinates of plate:
    topLeft_x = int(box[1][0])
    topLeft_y = int(box[1][1])

    bottomLeft_x = int(box[0][0])
    bottomLeft_y = int(box[0][1])

    topRight_x = int(box[2][0])

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
    blurred = cv2.GaussianBlur(large_plate, (5, 5), 0)

    sharpened_image = cv2.filter2D(blurred, -1, kernel)
    img1 = get_threshold(sharpened_image)

    cv2.imshow("Blurred_image", img1)
    #cv2.imshow("sharpened plate", sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


_my_edge_filter(gray, 1)
