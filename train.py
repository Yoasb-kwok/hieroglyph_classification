from ultralytics import YOLO

# Load the base model
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    print("Starting training on NVIDIA GPU...")
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',
        epochs=50,       
        imgsz=640,
        batch=16,           # Increased to 16 because your GPU is powerful!
        name='hieroglyph_gpu',
        device=0            # MAGIC WORD FOR PC: '0' means your primary NVIDIA GPU
    )
    
    print("Training finished! Look for best.pt in the runs/detect/hieroglyph_gpu/weights/ folder.")