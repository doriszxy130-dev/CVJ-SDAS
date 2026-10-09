Atlantoaxial Disease Classification System

> Automated analysis of atlantoaxial disease: YOLOv8 spine ROI detection → ResNet50 disease classification.

## 1. Overview

This system analyzes cervical spine radiographs by first detecting the spine region of interest (ROI), then identifying the presence of atlantoaxial-related disease and performing multilabel classification for six conditions.

**Model development data:** 7,778 retrospective images from 2,802 patients, plus 3,054 publicly available normal images.  
**External validation:** 643 images from four independent hospitals.  
**Prospective testing:** 267 images collected prospectively at Hospital A.

---

## 2. Architecture

```text
Input cervical spine radiograph
    │
    ▼
┌───────────────────────────────────────────┐
│ ① YOLOv8s                                 │
│    Spine ROI detection                    │
│    Bounding box can be manually adjusted  │
└───────────────────────────────────────────┘
    │ Crop + 10% padding + resize to 224 × 224
    ▼
┌───────────────────────────────────────────┐
│ ② Binary ResNet50 (1 output)              │
│    Disease presence + confidence          │
└───────────────────────────────────────────┘
┌───────────────────────────────────────────┐
│ ③ Multilabel ResNet50 (6 outputs)         │
│    Probabilities for six disease classes  │
└───────────────────────────────────────────┘
    │
    ▼
Output: disease presence, detected diseases, and confidence scores
```

### Model Configuration

| Component | Model | Input Size | Output | Weight File |
|---|---|---|---|---|
| Bounding-box detection | YOLOv8s | 640 × 640 | One bounding box and confidence score | `yolo_best.pt` (22 MB) |
| Binary classification | ResNet50 | 224 × 224 | One logit | `binary_best.pt` (90 MB) |
| Multilabel classification | ResNet50 | 224 × 224 | Six logits | `multilabel_best.pt` (91 MB) |

**Weight availability:** The three study-trained final weight files are not currently included in the repository or submission package. Running inference requires obtaining these files separately and placing them in `deploy/`. Their access or sharing arrangements remain to be finalized.

---

## 3. Disease Classes

| # | Disease | Abbreviation |
|---|---|---|
| 1 | Anterior Atlantoaxial Dislocation | Anterior AAD |
| 2 | Posterior Atlantoaxial Dislocation | Posterior AAD |
| 3 | Basilar Invagination | BI |
| 4 | Os Odontoideum | OO |
| 5 | Occipitalization of the Atlas | — |
| 6 | C2–3 Non-segmentation | C2–3 NS |

**Mutual-exclusion constraint:** Anterior and posterior atlantoaxial dislocation cannot both be positive. A mutual-exclusion penalty with λ = 0.5 is included during training.

---

## 4. Performance

### Internal Validation

The multilabel validation set contains 1,153 images and 2,530 positive label instances from the same hospital, using a patient-level split.

#### Multilabel Classification: Six Classes

| Disease | F1 | AUC | Support |
|---|---|---|---|
| Anterior AAD | 0.9485 | 0.9545 | 876 |
| Posterior AAD | 0.8528 | 0.9845 | 129 |
| Basilar Invagination | 0.8453 | 0.9499 | 344 |
| Os Odontoideum | 0.8911 | 0.9548 | 511 |
| Occipitalization of the Atlas | 0.9119 | 0.9659 | 414 |
| C2–3 Non-segmentation | 0.9059 | 0.9792 | 256 |
| **Macro Average** | **0.8926** | **0.9648** | — |

| Metric | Value |
|---|---|
| Macro F1 | 0.8926 |
| Macro AUC | 0.9648 |
| Exact Match | 70.60% |
| Expected Calibration Error (ECE) | 0.0544 |

#### Binary Classification: Disease Present or Absent

| Metric | Value |
|---|---|
| AUC | **0.9938** |
| Accuracy | 0.9684 |
| F1 | 0.9779 |
| ECE | 0.0210 |

### Prospective Testing (N = 267)

This cohort was collected prospectively at Hospital A, from October 1, 2024, to January 31, 2025.

| Multilabel Evaluation | Macro F1 | Macro AUC | Exact Match |
|---|---|---|---|
| Internal validation | 0.8926 | 0.9648 | 70.6% |
| **Prospective testing** | **0.8868** | **0.9612** | **65.4%** |
| Change | −0.006 | −0.004 | −5.2 percentage points |

| Binary Evaluation | AUC | Accuracy |
|---|---|---|
| Internal validation | 0.9938 | 0.9684 |
| **Prospective testing** | **0.9119** | **0.9362** |

### External Multicenter Validation (N = 643)

The external validation cohort includes images from four independent hospitals.

| Multilabel Evaluation | Macro F1 | Macro AUC | Exact Match |
|---|---|---|---|
| Internal validation | 0.8926 | 0.9648 | 70.6% |
| **External validation** | **0.8091** | **0.9329** | **55.8%** |
| Change | −0.083 | −0.032 | −14.8 percentage points |

| Hospital | Images | Multilabel Macro F1 | Multilabel Exact Match | Binary AUC |
|---|---|---|---|---|
| Second Hospital of Zhejiang University | 380 | 0.8258 | 56.3% | 0.8735 |
| Shengjing Hospital | 57 | 0.8699 | 77.2% | 0.9340 |
| Hebei Third Hospital | 71 | 0.7460 | 40.9% | 0.7575 |
| Jilin University Third Hospital | 135 | 0.7169 | 53.3% | 0.6953 |

---

## 5. Deployment

### Dependencies

```text
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
ultralytics >= 8.0 (for YOLOv8)
pillow
numpy
```

```bash
pip install torch torchvision ultralytics pillow numpy
```

### Quick Start

Before running the following commands, place the three required final weight files in `deploy/`. The repository does not currently include demonstration radiographs.

```bash
# Run inference on a single cervical spine radiograph
python deploy/inference.py --image cervical_xray.jpg

# Specify a bounding box to bypass YOLO detection
python deploy/inference.py --image cervical_xray.jpg --bbox "100,200,300,400"

# Run batch inference
python deploy/inference.py --dir /path/to/images/ --output results.csv

# Run inference on a GPU
python deploy/inference.py --image xray.jpg --device cuda
```

### Output Format

The following example illustrates the output structure. The `name_cn` fields contain Chinese disease names returned by the API. The timing values are illustrative and are not a runtime guarantee for other hardware.

```json
{
  "success": true,
  "image_size": [628, 707],
  "bbox": {
    "xmin": 241.2, "ymin": 162.5, "xmax": 391.6, "ymax": 277.9,
    "confidence": 0.871, "source": "yolo_predicted"
  },
  "disease_present": true,
  "disease_present_confidence": 0.9999,
  "diseases_found": [
    {"name_en": "Anterior AAD", "name_cn": "寰椎前脱位", "confidence": 0.9995},
    {"name_en": "Basilar Invagination (BI)", "name_cn": "颅底凹陷", "confidence": 0.9887}
  ],
  "timing_ms": {"yolo_ms": 487, "binary_ms": 53, "multilabel_ms": 11, "total_ms": 561}
}
```

### Platform Integration

1. **Manual bounding-box adjustment:** Call `engine.predict(image, bbox=(xmin, ymin, xmax, ymax))` to bypass YOLO detection and use the supplied bounding box.
2. **Result display:** Use `disease_present` to indicate disease presence and `diseases_found` to display detected diseases and confidence scores.
3. **No disease detected:** When `diseases_found=[]` and `disease_present=false`, display `disease_present_confidence`.
4. **Disease detected:** Display `diseases_found` in descending order of confidence.

### Python API

```python
from deploy.inference import SpineInference

engine = SpineInference("deploy/", device="cuda")

# Automatic YOLO detection
result = engine.predict("xray.jpg")

# Override detection with a manual bounding box
result = engine.predict("xray.jpg", bbox=(100, 150, 300, 350))

print(result["disease_present"])
for d in result["diseases_found"]:
    print(f"  {d['name_en']}: {d['confidence']:.4f}")
```

---

## 6. Project Structure

The following layout shows the deployment files and result folders. Weight files must be supplied separately; the layout does not imply that all listed artifacts are included in the current release.

```text
Atlantoaxial_Disease_Classifier/
├── deploy/
│   ├── inference.py          # Standalone inference script; no imports from other project modules
│   ├── config.json           # Model configuration and performance metrics
│   ├── yolo_best.pt          # YOLOv8s spine detector (22 MB; not included)
│   ├── binary_best.pt        # Binary ResNet50 (90 MB; not included)
│   ├── multilabel_best.pt    # Multilabel ResNet50 (91 MB; not included)
│   └── README.md             # This document
└── results/
    ├── internal_val/         # Internal validation results
    │   ├── multilabel/       # 6 × 6 confusion matrix, ROC curves, calibration plots, etc.
    │   └── binary/           # 2 × 2 confusion matrix, ROC curves, etc.
    ├── prospective_test/     # Prospective test results (267 images)
    │   ├── summary.json
    │   ├── multilabel/
    │   └── binary/
    └── external_val/         # External multicenter validation (643 images, four hospitals)
        ├── summary.json
        ├── multilabel/
        └── binary/
```

---

## 7. Training Details

| Parameter | Value |
|---|---|
| Backbone | ResNet50 with ImageNet-pretrained weights |
| Optimizer | AdamW (learning rate = 1e-4, weight decay = 1e-4) |
| Scheduler | CosineAnnealingLR (T_max = 100) |
| Binary loss | BCEWithLogitsLoss |
| Multilabel loss | BCEWithLogitsLoss + pos_weight + mutex_penalty (λ = 0.5) |
| Image size | 224 × 224 after cropping/resizing |
| ROI sampling | 50% cropped ROI / 50% full image |
| Batch size | 32 |
| Automatic mixed precision (AMP) | Enabled |
| Early stopping | Patience = 15 |
| Data split | Patient-grouped GroupShuffleSplit, with a 15% validation fraction |

---

## 8. Plot Guide

| File | Evaluation Level | Description |
|---|---|---|
| `confusion_matrix.png` | Label level | 6 × 6 matrix of true-disease versus predicted-disease counts |
| `error_decomposition.png` | Label level | False-positive counts excluding co-occurring ground-truth diseases, with precision/recall/F1 bar charts |
| `diagnostic_image_level.png` | Image level | Three-category evaluation (complete match, missed diagnosis, incorrect diagnosis) and distribution of the number of correctly predicted labels |
| `roc_curves.png` | Label level | ROC curves for six classes and the macro average |
| `calibration_curves.png` | Label level | Reliability diagrams for probability calibration |
| `confidence_dist.png` | Label level | Confidence distributions for positive and negative samples |

---

## 9. Changelog

| Version | Date | Description |
|---|---|---|
| v1.0 | 2026-06-17 | Initial release: internal validation, prospective testing, and external multicenter validation |

---

*For questions, please contact the project lead.
