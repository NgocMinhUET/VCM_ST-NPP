# VCM-STNPP: Spatio-Temporal Neural Preprocessing for Video Coding in Machine Vision Tasks

## 🎯 Goal
This project aims to improve video compression for downstream machine vision tasks (e.g., detection, segmentation, tracking) using a neural preprocessing module trained end-to-end with task-driven loss, while maintaining compatibility with HEVC/VVC codecs.

---

## 📁 Project Structure
```
models/
  st_npp.py, qal.py, proxy_codec.py, combined_model.py
utils/
  data_utils.py, codec_utils.py, model_utils.py, metric_utils.py
scripts/
  run_comprehensive_evaluation.sh, run_comprehensive_evaluation.py
train.py
evaluate.py
requirements.txt
```

---

## 🚀 Installation

### Create environment:
```bash
python -m venv vcm_env
source vcm_env/bin/activate
pip install -r requirements.txt
```

### Prepare dataset:
Structure:
```
data/
  coco/
  kitti/
  motchallenge/
```
Use scripts or instructions in `data/README.md` to download and format datasets.

---

## 🧠 Training
```bash
python train.py --dataset coco --batch_size 4 --lr 1e-4 --epochs 20 --checkpoint_dir checkpoints/
```

---

## 📈 Evaluation
```bash
python evaluate.py --dataset coco --task detection --checkpoint checkpoints/stnpp_qal_epoch20.pt
```

---

## 🔁 Comprehensive Evaluation
```bash
bash scripts/run_comprehensive_evaluation.sh
```

---

## 📊 Metrics
- Detection: mAP (COCO)
- Segmentation: mIoU (KITTI)
- Tracking: MOTA, IDF1 (MOTChallenge)
- Rate-Distortion: BD-Rate (via ffmpeg)

---

## 📎 Notes
- FFmpeg must support `libx264`, `libx265`.
- Proxy codec is used during training only; inference uses real codecs.
