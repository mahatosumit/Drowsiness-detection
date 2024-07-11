import torch
import cv2
from ultralytics import YOLO

# Model path (replace with your actual path)
model_path = r"G:\SUMIT\SGCAM\best.pt"  # Update with your YOLOv8 model path

# Class names (replace with the classes your model detects)
class_names = ['awake', 'drowsy']  # Update with your classes

# Load the YOLOv8 model
try:
    model = YOLO(model_path)  # Load the YOLOv8 model
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    # Run object detection on the frame using YOLO model
    results = model(frame)

    # Process the detection results
    for result in results:
        # Extract bounding box coordinates, confidence score, and class label
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

            # Display class label and confidence score (optional)
            class_label = class_names[int(cls)]
            text = f"{class_label} ({conf:.2f})"
            cv2.putText(frame, text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the resulting frame with detections
    cv2.imshow('Webcam Object Detection', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
