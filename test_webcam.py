from ultralytics import YOLO

# 1. Load YOUR custom trained model
model = YOLO('runs/detect/hieroglyph_test-4/weights/best.pt')

# 2. Turn on the Mac webcam (source='0') and show the results live
print("Opening webcam... Press 'q' in the terminal to stop.")
model.predict(source='0', show=True, conf=0.5)