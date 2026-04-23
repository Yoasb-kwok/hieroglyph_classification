# Hieroglyph Classification (YOLOv8)

Two-class YOLOv8 detector for the hieroglyphs `de_hieroglyph` and `ren_hieroglyph`.

A pretrained model (`hieroglyph.pt`, ~6 MB) is shipped in the repo so you can run
inference immediately without training.

## Setup (Windows, PowerShell)

```powershell
git clone https://github.com/Yoasb-kwok/hieroglyph_classification.git
cd hieroglyph_classification

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Install dependencies. **Order matters** — install PyTorch first from the CUDA
wheel index, otherwise `pip` will silently install a CPU-only build and
`device=0` will fail.

NVIDIA GPU (recommended):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

No GPU / CPU only:

```powershell
pip install -r requirements.txt
```

(And edit `device=0` → `device="cpu"` in `train.py` if you plan to train.)

Verify CUDA is visible to PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Should print `True 1` (or more).

## Run inference on webcam

```powershell
python test_webcam.py
```

Uses `hieroglyph.pt` by default. If you train your own model it will instead
auto-pick the newest `runs/detect/*/weights/best.pt`. Press `q` in the preview
window to stop.

## Train from scratch

```powershell
python train.py
```

Produces weights at `runs/detect/hieroglyph_gpu/weights/best.pt`. 50 epochs on
an RTX 5070 takes a couple of minutes for the bundled dataset (62 training
images, 2 classes).

Training config lives in `dataset/data.yaml`. The dataset path is resolved
portably at runtime via `ultralytics.settings.datasets_dir`, so you do not
need to edit the YAML.

## Dataset layout

```
dataset/
  data.yaml
  images/train/*.jpg|*.jpeg
  labels/train/*.txt      # YOLO format, one row per box
```

`val` currently reuses `images/train` — fine for smoke tests, useless for real
mAP. Split out ~15% into `dataset/images/val` + `dataset/labels/val` before
taking validation metrics seriously.
