import cv2
import numpy as np

img = cv2.imread("messi.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel gradients

'''
dx (0 or 1) - Order of derivative (x-direction)
dy (0 or 1) - Order of derivative (x-direction)
'''

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # horizontal gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # vertical gradient
sobel = np.sqrt(sobelx**2 + sobely**2)

cv2.imshow("Original", img)
cv2.imshow("Sobel Edge", sobel / sobel.max())

cv2.waitKey(0)
cv2.destroyAllWindows()
