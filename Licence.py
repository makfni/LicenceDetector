import cv2
import numpy as np
# Read the image file
from skimage.exposure import rescale_intensity

image = cv2.imread('Images/blurry_plate3.jpg', 0)

# works for 1,2,5
image = cv2.resize(image, (325, 325))


# # Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(image, 11, 17, 17)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)


# Find contours based on Edges
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # sort contours based on their area keeping minimum
# required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None  # we currently have no Number plate contour
# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx  # This is our approx Number Plate Contour
        break

# rc = cv2.drawContours(image, [NumberPlateCnt], -1, 255, 1)
rc = cv2.minAreaRect(NumberPlateCnt)
box = cv2.boxPoints(rc)
for p in box:
    pt = (p[0], p[1])
    print(pt)
    cv2.circle(image, pt, 5, (200, 0, 0), 2)

cv2.imshow("Licence Plate Detected", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
