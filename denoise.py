import numpy as np
import cv2
from PIL import Image
from scipy import misc
import pytesseract


im = Image.open('new_coon.jpg')
data = np.array(im)
flattened = data.flatten()


#
# img = cv2.imread('new_coon.jpg', 0)
# data = np.array(flattened)
im = cv2.imread('new_coon.jpg',0)
# Calculate U (u), E (s) and V (vh)
u, s, vh = np.linalg.svd(im, full_matrices=False)

# Remove sigma values below threshold (250)
s_cleaned = np.array([si if si > 250 else 0 for si in s])

# Calculate A' = U * E (cleaned) * V
img_denoised = np.array(np.dot(u * s_cleaned, vh), dtype=int)

# Save the new image
cv2.imwrite('new_new_coon.jpg', img_denoised)

new_shits = cv2.imread('new_coon.jpg')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
img_dilation = cv2.dilate(new_shits, kernel, iterations=1)

cv2.imshow('img_dilation', img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the OCR library to read the characters of the plate
text = pytesseract.image_to_string(img_dilation)
print("OG TEXT: ", text)
# For testing purposes
# the digits represent all ten numbers that the plate may have
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# letters contains all the capital letters that the plate may contain
# Main purpose is to remove all unecessary characters that the OCR
# May pick up
letters = [chr(ord('A') + i) for i in range(25, -1, -1)]
print("This is the letters: ", letters)
print("These are the digits: ", digits)

print("This is the text: ", text)

# Create an empty string
license_plate = ''
for i in(text):
    # Only pick up the characters or digits
    if (i in digits) or (i in letters):
        license_plate = license_plate + i
    else:
        license_plate = license_plate + ' '
print(license_plate)
