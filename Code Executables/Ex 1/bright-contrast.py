import cv2

# Read image
img = cv2.imread('messi.jpg')

# Create window
cv2.namedWindow('Image')

# Dummy function for trackbar
def nothing(x):
    pass

# Create trackbars
cv2.createTrackbar('Brightness', 'Image', 50, 100, nothing)
cv2.createTrackbar('Contrast', 'Image', 50, 100, nothing)

while True:
    # Get slider values
    brightness = cv2.getTrackbarPos('Brightness', 'Image')
    contrast = cv2.getTrackbarPos('Contrast', 'Image')

    # Adjust brightness and contrast
    adjusted = cv2.convertScaleAbs(img, alpha=contrast/50, beta=brightness-50)

    # Show image
    cv2.imshow('Image', adjusted)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
