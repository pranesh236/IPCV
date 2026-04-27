from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Give the image path
results = model("image.png")

# Draw detected objects on image
output = results[0].plot()

# Show image
Image.fromarray(output).show()

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print("Object:", results[0].names[class_id])
    print("Confidence:", confidence)
