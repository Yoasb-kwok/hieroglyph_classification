from pathlib import Path

from ultralytics import YOLO, settings

HERE = Path(__file__).resolve().parent
settings.update({"datasets_dir": str(HERE)})

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    print("Starting training on NVIDIA GPU...")
    results = model.train(
        data=str(HERE / "dataset" / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        name="hieroglyph_gpu",
        device=0,
    )
    print("Training finished! Look for best.pt in runs/detect/hieroglyph_gpu/weights/")
