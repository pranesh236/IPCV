import cv2
import random
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open input video
cap = cv2.VideoCapture("video.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # For .mp4 output
out = cv2.VideoWriter(
    "output_tracked.mp4",
    fourcc,
    fps,
    (frame_width, frame_height)
)

# Function to give same color for same object class
def get_color(class_id):
    random.seed(class_id)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run tracking
    results = model.track(frame, persist=True)

    for result in results:
        for box in result.boxes:

            # Coordinates of box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Object class and confidence
            class_id = int(box.cls[0])

            # Tracking ID
            if box.id is not None:
                object_id = int(box.id[0])
            else:
                object_id = 0

            # Get color based on class
            color = get_color(class_id)

            # Object name
            object_name = result.names[class_id]

            # Label with object name and ID
            label = f"{object_name} ID:{object_id}"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # Save processed frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("YOLO Tracking", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
