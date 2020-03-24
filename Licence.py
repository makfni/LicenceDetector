import cv2
import numpy as np
# Read the image file
from skimage.exposure import rescale_intensity

image = cv2.imread('Images/blurry_plate.jpg', 0)
def get_threshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# works for 1,2,5
image = cv2.resize(image, (325, 325))

# # Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(image, 11, 17, 17)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)


# Find contours based on Edges
cnts, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

cv2.imshow("Licence Plate Detected", image)

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
crop_img = image[topLeft_y:topLeft_y + plate_height, bottomLeft_x:bottomLeft_x + plate_width]

cv2.imshow("Cropped License Plate", crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
# increase plate size:
large_plate = cv2.resize(crop_img, (0,0), fx=3, fy=3,interpolation=cv2.INTER_CUBIC)

# sharpen cropped image
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# Apply kernel to image 
blurred = cv2.GaussianBlur(large_plate, (5, 5), 0)

sharpened_image = cv2.filter2D(blurred, -1, kernel)
img = get_threshold(sharpened_image)

cv2.imshow("Blurred_image", img)
# cv2.imshow("sharpened plate", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
