# Model weights (not included in this archive)

This folder should contain the following three files before `inference.py` can run. They are deliberately excluded from this code archive — see the top-level `README.md` §3 for why and for hosting recommendations (Git LFS or Zenodo).

| File | Size | Role | Reported validation metrics (internal validation set) |
|---|---|---|---|
| `yolo_best.pt` | ~22 MB | YOLOv8s spine-ROI detector | — |
| `binary_best.pt` | ~90 MB | ResNet50 disease-presence (binary) classifier | AUC 0.9938, Accuracy 0.9684, F1 0.9779, ECE 0.0210 |
| `multilabel_best.pt` | ~91 MB | ResNet50 six-disease multilabel classifier | F1_macro 0.8926, AUC_macro 0.9648, Exact-match 70.6%, ECE 0.0544 |

Once hosted (Git LFS / Zenodo / other), place all three files directly in this `deploy/` folder (or point `SpineInference(...)` / `config.json` at wherever you put them) and `deploy/inference.py` will work exactly as documented in `deploy/README.md`.
