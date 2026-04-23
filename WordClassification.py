from ultralytics import YOLO

# Load the smallest, fastest YOLO model (perfect for iPhone)
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=50,       # 50 loops is good for a quick test
    imgsz=640,
    batch=8,         # Lower batch size is safer for standard laptops
    name='hieroglyph_test'
)

print("Finished! Look for your best.pt file in runs/detect/hieroglyph_test/weights/")