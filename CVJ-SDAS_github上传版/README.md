# CVJ-SDAS

Craniovertebral Junction Deformity Screening and Diagnosis Assistance System — a deep-learning pipeline that detects the spine ROI in a lateral cervical radiograph (YOLOv8s) and then screens for six craniovertebral-junction abnormalities via two ResNet50 classifiers (binary disease-presence + six-way multilabel).

This repository contains the code used to produce the results reported in the manuscript *"Development and validation of the craniovertebral junction deformity screening and diagnosis assistance system (CVJ-SDAS) based on deep learning models."* It does not contain patient radiographs, trained model weights, or any AI-manuscript-editing artifacts — see below for how to obtain those.

---

## 1. Pipeline overview

The scripts in `pipeline/` are meant to be run from that same directory (some of them import each other directly, e.g. the evaluation scripts do `import train_classifier as tc`), in the following order:

| Stage | Script(s) | What it does |
|---|---|---|
| 01 — Prepare YOLO dataset | `prepare_yolov8_bbox_dataset.py` | Builds a YOLOv8-format bbox dataset (images/labels/data.yaml) from the raw radiographs and bbox annotations |
| 02 — Train ROI detector | `train_yolov8_bbox.py` | Trains the YOLOv8s spine-ROI detector |
| 03 — Predict bboxes | `predict_yolov8_bboxes.py` | Runs the trained YOLO detector over the full image pool to get ROI bboxes for every image |
| 04 — Build metadata | `build_final_bbox_datasets.py`, `build_public_normal_csv.py`, `build_unified_metadata.py` | Assembles the final per-image label/bbox tables (retrospective disease set + public normal set) into one unified metadata table |
| 05 — Train classifiers | `train_classifier.py` | Trains **both** final classifiers from one script via a `--mode` flag:<br>`python train_classifier.py --mode binary` → disease-presence model<br>`python train_classifier.py --mode multilabel` → six-disease multilabel model |
| 06 — Evaluation | `eval_binary_final.py`, `eval_multilabel_refresh.py`, `eval_patient_level.py`, `eval_prospective.py`, `eval_external_val.py`, `gen_binary_ext_pro.py`, `compute_stats.py` | Internal validation, prospective test, and 4-hospital external validation evaluation + the statistical tests reported in the manuscript |

`deploy/inference.py` is a **standalone**, dependency-free-of-the-rest-of-the-repo inference script/module — see `deploy/README.md` for full usage, model-config and performance-table details. `api/cervical_api_v17.py` is the HTTP service (Python stdlib `http.server`, no extra web framework required) that the web-based interactive platform described in the manuscript talks to.

### Which script produced the paper's reported models — a note on script naming

The delivery package originally contained five similarly-named training scripts. We traced provenance by matching `import` statements, log timestamps, and logged config values against the curated final results, and confirmed:

- **`train_classifier.py`** (used with `--mode binary` and `--mode multilabel`) produced **both** final, paper-reported models. Every evaluation script in this repo imports it directly (`import train_classifier as tc` / `from train_classifier import *`), which is the strongest evidence of which script is actually "live." Its logged run configuration matches the file timestamps of the curated `results/internal_val/` outputs exactly, and `eval_binary_final.py`'s internal-validation output (N=1,676, accuracy 0.97) matches the manuscript's Para 119 figure.
- `train_disease_presence.py` and `train_retrospective_multilabel.py` are earlier, separate-script versions of the same two models (predating `train_classifier.py` by about 13.6 hours based on file timestamps) that nothing else in the codebase imports or calls. They were superseded by the unified `train_classifier.py --mode ...` approach and are **not included** in this release.
- `train_calibrated_v2.py` / `train_multilabel_calibrated.py` are **not** the paper's main multilabel model, despite being the most recently modified files in the delivery package. Their log explicitly starts with `"Starting training (Focal γ=2.0, label_smoothing=0.1, mutex λ=0.5)..."`, and their output folder has no confusion matrix, ROC curve, or per-label metrics — i.e. this run was never fully evaluated. This matches an unfilled placeholder still sitting in the manuscript draft: *"The calibrated model variant trained with Focal Loss and label smoothing achieved [F1_macro = X, AUC_macro = X, ECE = X — FILL AFTER TRAINING]."* **Please decide, before submission, whether to (a) finish evaluating this variant and fill in that sentence with real numbers, using this script, or (b) delete that placeholder sentence from the manuscript** — right now it's an incomplete claim in the draft. Because it isn't part of the reported pipeline, this script is not included in this release; ask if you'd like it added back in once you've decided.

---

## 2. Environment

```bash
pip install -r requirements.txt
```

Python ≥ 3.9. GPU (CUDA) strongly recommended for training; CPU is fine for single-image inference via `deploy/inference.py`.

---

## 3. Model weights

Trained weights are **not included in this archive** (three files, ~22–95 MB each — right at GitHub's soft size limit, and intermediate training checkpoints in the original delivery package are ~282 MB each and definitely should not be versioned in git). Recommended options:

- Host `yolo_best.pt`, `binary_best.pt`, `multilabel_best.pt` via **Git LFS** if you want them in the same GitHub repo as the code, or
- Archive them on **Zenodo** (or similar) and cite the resulting DOI in the manuscript's Code/Data Availability statement — this is the more common choice for medical-AI papers and gives you a permanent, citable reference instead of just a GitHub link.

Only the three final, deployment-ready weight files should be published (see `deploy/README.md` §2 for exact file names/sizes/expected metrics). Do **not** publish the intermediate `checkpoint_epochNNN.pt` files or the raw `best_model.pt` training-state files — they're training artifacts, not needed for reproducing the reported results, and are far too large for routine hosting.

---

## 4. Data availability

No patient radiographs or per-patient label files are included in this repository, consistent with the manuscript's Data Availability Statement. Requests for access to the de-identified image data should go through the corresponding author, per that statement.

---

## 5. License

See `LICENSE` (MIT). Change before publishing if your institution/journal requires a different license.

---

## 6. Citation

If you use this code, please cite:

> [Full citation to be added once the manuscript is accepted/published.]
