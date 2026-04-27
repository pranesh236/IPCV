import cv2

# Read image in grayscale
img = cv2.imread("messi.jpg", 0)

# Apply Canny edge detector
t1 = 100 # lower threshold
t2 = 200 # upper threshold
edges = cv2.Canny(img, 100, 200)

# Display
cv2.imshow("Original", img)
cv2.imshow("Canny Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
