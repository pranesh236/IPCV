import cv2

# Webcam
cap = cv2.VideoCapture(0)

# Read first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Find difference between consecutive frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Increase object area
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # Ignore small movements
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw rectangle around moving object
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame1,
            "Motion Detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Display the frame
    cv2.imshow("Motion Tracking from Video", frame1)

    # Move to next frame
    frame1 = frame2
    ret, frame2 = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Press q to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
