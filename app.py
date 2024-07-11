from ultralytics import YOLO

# Load a model configuration (YOLOv8 nano model in this case)
model = YOLO("yolov8n.yaml")  # Correctly specify the model configuration file

# Train the model
results = model.train(
    data="data.yaml",  # Path to the data.yaml file
    epochs=10,         # Number of epochs
    batch=16,          # Batch size (optional)
    imgsz=640          # Image size (optional)
)

# Optionally, save the trained model
model.save("yolov8n.pt")

# Print results summary
print(results)
