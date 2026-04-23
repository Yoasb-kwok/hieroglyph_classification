from pathlib import Path

from ultralytics import YOLO, settings

HERE = Path(__file__).resolve().parent
settings.update({"datasets_dir": str(HERE)})

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    results = model.train(
        data=str(HERE / "dataset" / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        name="hieroglyph_test",
    )
    print("Finished! Look for your best.pt file in runs/detect/hieroglyph_test/weights/")
