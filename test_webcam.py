from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

# Prefer the shipped pretrained model; fall back to the newest local training run.
shipped = HERE / "hieroglyph.pt"
if shipped.exists():
    weights = shipped
else:
    candidates = sorted(
        (HERE / "runs" / "detect").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No trained weights found. Expected 'hieroglyph.pt' in the repo root "
            "or a 'runs/detect/*/weights/best.pt' from a previous training run. "
            "Run `python train.py` first, or re-clone the repo to pick up hieroglyph.pt."
        )
    weights = candidates[0]

print(f"Loading weights: {weights}")
model = YOLO(str(weights))

print("Opening webcam... Press 'q' in the preview window to stop.")
model.predict(source="0", show=True, conf=0.5)
