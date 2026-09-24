"""
Unified inference pipeline for Atlantoaxial Disease Classification.

Pipeline:
  1. YOLOv8 detects spine ROI bbox
  2. (Optional) User can adjust the bbox manually
  3. Crop image with 10% padding using final bbox
  4. Binary model → disease_present (yes/no) + confidence
  5. Multilabel model → 6 disease probabilities
  6. Return structured result with per-disease confidence

Usage:
  python inference.py --image path/to/image.jpg
  python inference.py --image path/to/image.jpg --bbox 100,200,300,400   # skip YOLO
  python inference.py --dir path/to/images/ --output results.csv

Programmatic:
  from inference import SpineInference
  engine = SpineInference("deploy/")
  result = engine.predict("image.jpg")
  result = engine.predict("image.jpg", bbox=(100, 200, 300, 400))  # manual bbox
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# ── Constants ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABELS_EN = [
    "Anterior AAD",
    "Posterior AAD",
    "Basilar Invagination (BI)",
    "Os Odontoideum (OO)",
    "Occipitalization of Atlas",
    "C2-3 Non-segmentation",
]

LABELS_CN = [
    "寰椎前脱位",
    "寰椎后脱位",
    "颅底凹陷",
    "齿突不连",
    "寰椎枕化",
    "颈2-3分节不全",
]


# ═══════════════════════════════════════════════════════════════════════
#  Model builder (exactly matches training)
# ═══════════════════════════════════════════════════════════════════════

def build_resnet50(n_classes: int) -> torch.nn.Module:
    """ResNet50 backbone with ImageNet-pretrained fc replaced."""
    from torchvision import models
    if hasattr(models, 'ResNet50_Weights'):
        w = models.ResNet50_Weights.DEFAULT
    else:
        w = None
    m = models.resnet50(weights=w)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    return m


# ═══════════════════════════════════════════════════════════════════════
#  BBox utilities
# ═══════════════════════════════════════════════════════════════════════

def crop_with_padding(img: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """
    Crop image using bbox with 10% padding on each side (matches training).
    bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    w, h = img.size
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    bw = xmax - xmin
    bh = ymax - ymin

    left = max(0, int(xmin - bw * 0.1))
    top = max(0, int(ymin - bh * 0.1))
    right = min(w, int(xmax + bw * 0.1))
    bottom = min(h, int(ymax + bh * 0.1))

    if right > left and bottom > top:
        return img.crop((left, top, right, bottom))
    return img


def clamp_bbox(bbox: Tuple[float, ...], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Clamp bbox coordinates to image boundaries."""
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    xmin = max(0.0, min(xmin, img_w))
    ymin = max(0.0, min(ymin, img_h))
    xmax = max(xmin + 1, min(xmax, img_w))
    ymax = max(ymin + 1, min(ymax, img_h))
    return (xmin, ymin, xmax, ymax)


# ═══════════════════════════════════════════════════════════════════════
#  Main inference engine
# ═══════════════════════════════════════════════════════════════════════

class SpineInference:
    """
    Unified inference engine for spine disease classification.

    Parameters
    ----------
    deploy_dir : str or Path
        Directory containing:
          - yolo_best.pt          (YOLOv8 bbox detector)
          - binary_best.pt        (Binary ResNet50)
          - multilabel_best.pt    (Multilabel ResNet50)
          - config.json           (optional, overridden by explicit args)
    device : str
        'cuda', 'cpu', or None (auto-detect).
    yolo_conf : float
        YOLO confidence threshold (default 0.25).
    yolo_imgsz : int
        YOLO input image size (default 640).
    """

    def __init__(
        self,
        deploy_dir: str = "deploy",
        device: Optional[str] = None,
        yolo_conf: float = 0.25,
        yolo_imgsz: int = 640,
    ):
        self.deploy_dir = Path(deploy_dir)
        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Transform (matches training exactly)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Load YOLO
        self._load_yolo()

        # Load classifiers
        self._load_classifiers()

        logging.info("SpineInference ready on %s", self.device)

    def _load_yolo(self):
        """Load YOLOv8 bbox detector."""
        yolo_path = self.deploy_dir / "yolo_best.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")

        try:
            from ultralytics import YOLO
            self.yolo = YOLO(str(yolo_path))
            logging.info("YOLO loaded: %s", yolo_path)
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    def _load_classifiers(self):
        """Load binary and multilabel ResNet50 models."""
        binary_path = self.deploy_dir / "binary_best.pt"
        multilabel_path = self.deploy_dir / "multilabel_best.pt"

        if not binary_path.exists():
            raise FileNotFoundError(f"Binary weights not found: {binary_path}")
        if not multilabel_path.exists():
            raise FileNotFoundError(f"Multilabel weights not found: {multilabel_path}")

        # Binary model (1 output)
        ckpt_b = torch.load(binary_path, map_location=self.device, weights_only=False)
        self.binary_model = build_resnet50(1)
        self.binary_model.load_state_dict(ckpt_b["model_state"])
        self.binary_model.to(self.device)
        self.binary_model.eval()
        logging.info("Binary model loaded (epoch %d)", ckpt_b.get("epoch", -1))

        # Multilabel model (6 outputs)
        ckpt_m = torch.load(multilabel_path, map_location=self.device, weights_only=False)
        self.multilabel_model = build_resnet50(6)
        self.multilabel_model.load_state_dict(ckpt_m["model_state"])
        self.multilabel_model.to(self.device)
        self.multilabel_model.eval()
        logging.info("Multilabel model loaded (epoch %d)", ckpt_m.get("epoch", -1))

    def detect_bbox(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Detect spine ROI bbox using YOLOv8.
        Returns dict with bbox + confidence, or None if not detected.
        """
        # Save temp file (YOLO API works best with file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG", quality=95)
            tmp_path = tmp.name

        try:
            results = self.yolo(tmp_path, conf=self.yolo_conf, imgsz=self.yolo_imgsz,
                               device=self.device, verbose=False)
            if isinstance(results, list):
                results = results[0]

            if results.boxes is None or len(results.boxes) == 0:
                return None

            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            xmin, ymin, xmax, ymax = boxes[best_idx].tolist()
            confidence = float(confs[best_idx])

            return {
                "bbox": (xmin, ymin, xmax, ymax),
                "confidence": confidence,
                "source": "yolo_predicted",
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def predict(
        self,
        image_path_or_pil,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        binary_threshold: float = 0.5,
        multilabel_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run full inference pipeline.

        Parameters
        ----------
        image_path_or_pil : str, Path, or PIL.Image
            Input X-ray image.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Manual bbox override. If provided, skips YOLO detection.
        binary_threshold : float
            Decision threshold for disease present (default 0.5).
        multilabel_threshold : float
            Decision threshold for each disease class (default 0.5).

        Returns
        -------
        dict with keys:
            success : bool
            image_size : (w, h)
            bbox : dict with xmin/ymin/xmax/ymax, confidence, source
            binary_result : dict with disease_present (bool), confidence (float)
            multilabel_result : dict with per-class present (bool), confidence (float)
            diseases_found : list of dicts [{name_en, name_cn, confidence}, ...]
            timing_ms : dict with yolo_ms, binary_ms, multilabel_ms
            error : str (if failed)
        """
        t_start = time.time()

        # ── Load image ──
        if isinstance(image_path_or_pil, (str, Path)):
            img = Image.open(image_path_or_pil).convert("RGB")
            img = ImageOps.exif_transpose(img)
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil.convert("RGB")
        else:
            return {"success": False, "error": f"Unsupported input type: {type(image_path_or_pil)}"}

        img_w, img_h = img.size
        timing = {}

        # ── Step 1: BBox detection ──
        t0 = time.time()
        if bbox is not None:
            bbox_clamped = clamp_bbox(bbox, img_w, img_h)
            bbox_info = {
                "xmin": bbox_clamped[0], "ymin": bbox_clamped[1],
                "xmax": bbox_clamped[2], "ymax": bbox_clamped[3],
                "confidence": 1.0,
                "source": "manual",
            }
        else:
            detected = self.detect_bbox(img)
            if detected is None:
                return {
                    "success": False,
                    "error": "YOLO bbox detection failed — no spine ROI found",
                    "image_size": (img_w, img_h),
                    "timing_ms": {"yolo_ms": (time.time() - t0) * 1000},
                }
            bbox_clamped = clamp_bbox(detected["bbox"], img_w, img_h)
            bbox_info = {
                "xmin": bbox_clamped[0], "ymin": bbox_clamped[1],
                "xmax": bbox_clamped[2], "ymax": bbox_clamped[3],
                "confidence": detected["confidence"],
                "source": detected["source"],
            }
        timing["yolo_ms"] = (time.time() - t0) * 1000

        # ── Step 2: Crop ──
        img_cropped = crop_with_padding(img, bbox_clamped)
        img_tensor = self.transform(img_cropped).unsqueeze(0).to(self.device)

        # ── Step 3: Binary classification ──
        t0 = time.time()
        with torch.no_grad():
            binary_logit = self.binary_model(img_tensor)
        binary_prob = float(torch.sigmoid(binary_logit).cpu().item())
        binary_present = binary_prob >= binary_threshold
        timing["binary_ms"] = (time.time() - t0) * 1000

        binary_result = {
            "disease_present": binary_present,
            "confidence": binary_prob,
            "threshold": binary_threshold,
        }

        # ── Step 4: Multilabel classification ──
        t0 = time.time()
        with torch.no_grad():
            multilabel_logits = self.multilabel_model(img_tensor)
        multilabel_probs = torch.sigmoid(multilabel_logits).cpu().numpy()[0]
        timing["multilabel_ms"] = (time.time() - t0) * 1000

        multilabel_result = {}
        diseases_found = []
        for i, (name_en, name_cn) in enumerate(zip(LABELS_EN, LABELS_CN)):
            prob = float(multilabel_probs[i])
            present = prob >= multilabel_threshold
            multilabel_result[name_en] = {
                "present": present,
                "confidence": prob,
                "name_cn": name_cn,
            }
            if present:
                diseases_found.append({
                    "name_en": name_en,
                    "name_cn": name_cn,
                    "confidence": prob,
                })

        # Sort diseases by confidence descending
        diseases_found.sort(key=lambda x: x["confidence"], reverse=True)

        # ── Consensus: overall disease_present from multilabel (more calibrated) ──
        any_disease = len(diseases_found) > 0

        timing["total_ms"] = (time.time() - t_start) * 1000

        return {
            "success": True,
            "image_size": (img_w, img_h),
            "bbox": bbox_info,
            "disease_present": any_disease,                         # ← key platform field
            "disease_present_confidence": binary_prob,              # ← binary score
            "binary_result": binary_result,
            "multilabel_result": multilabel_result,
            "diseases_found": diseases_found,                       # ← key: list of found diseases
            "timing_ms": timing,
        }

    def predict_batch(
        self,
        image_dir: str,
        output_csv: Optional[str] = None,
        bboxes: Optional[Dict[str, Tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference on a directory of images.

        Parameters
        ----------
        image_dir : str
            Directory containing images.
        output_csv : str, optional
            Path to save results CSV.
        bboxes : dict, optional
            {filename_stem: (xmin, ymin, xmax, ymax)} for manual bbox override.

        Returns
        -------
        List of result dicts (same format as predict()).
        """
        import csv
        image_dir = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix in exts])
        bboxes = bboxes or {}

        results = []
        for i, img_path in enumerate(image_paths):
            stem = img_path.stem
            manual_bbox = bboxes.get(stem)
            logging.info("[%d/%d] %s", i+1, len(image_paths), img_path.name)
            result = self.predict(img_path, bbox=manual_bbox)
            result["image_name"] = img_path.name
            result["image_stem"] = stem
            results.append(result)

        if output_csv:
            self._save_csv(results, output_csv)

        return results

    @staticmethod
    def _save_csv(results: List[Dict], output_csv: str):
        """Save batch results to CSV."""
        import csv
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "image_name", "success", "error",
                "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "bbox_source", "bbox_conf",
                "disease_present", "disease_confidence",
            ]
            for name in LABELS_EN:
                fieldnames.append(f"{name}_present")
                fieldnames.append(f"{name}_confidence")
            fieldnames.append("yolo_ms")
            fieldnames.append("binary_ms")
            fieldnames.append("multilabel_ms")
            fieldnames.append("total_ms")

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {
                    "image_name": r.get("image_name", ""),
                    "success": r.get("success", False),
                    "error": r.get("error", ""),
                    "bbox_xmin": r.get("bbox", {}).get("xmin"),
                    "bbox_ymin": r.get("bbox", {}).get("ymin"),
                    "bbox_xmax": r.get("bbox", {}).get("xmax"),
                    "bbox_ymax": r.get("bbox", {}).get("ymax"),
                    "bbox_source": r.get("bbox", {}).get("source"),
                    "bbox_conf": r.get("bbox", {}).get("confidence"),
                    "disease_present": r.get("binary_result", {}).get("disease_present"),
                    "disease_confidence": r.get("binary_result", {}).get("confidence"),
                    "yolo_ms": r.get("timing_ms", {}).get("yolo_ms"),
                    "binary_ms": r.get("timing_ms", {}).get("binary_ms"),
                    "multilabel_ms": r.get("timing_ms", {}).get("multilabel_ms"),
                    "total_ms": r.get("timing_ms", {}).get("total_ms"),
                }
                for name in LABELS_EN:
                    ml = r.get("multilabel_result", {}).get(name, {})
                    row[f"{name}_present"] = ml.get("present")
                    row[f"{name}_confidence"] = ml.get("confidence")
                writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Spine Disease Classification Inference")
    parser.add_argument("--deploy-dir", default=None,
                       help="Directory with yolo_best.pt, binary_best.pt, multilabel_best.pt")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--dir", help="Batch directory of images")
    parser.add_argument("--bbox", help="Manual bbox: xmin,ymin,xmax,ymax (overrides YOLO)")
    parser.add_argument("--output", help="Output CSV for batch mode")
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--multilabel-threshold", type=float, default=0.5)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Default deploy dir: same directory as this script
    if args.deploy_dir is None:
        args.deploy_dir = Path(__file__).resolve().parent

    engine = SpineInference(
        deploy_dir=args.deploy_dir,
        device=args.device,
        yolo_conf=args.yolo_conf,
    )

    # Parse manual bbox
    manual_bbox = None
    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(parts) != 4:
            parser.error("--bbox requires 4 comma-separated values: xmin,ymin,xmax,ymax")
        manual_bbox = (parts[0], parts[1], parts[2], parts[3])

    # Single image
    if args.image:
        result = engine.predict(
            args.image,
            bbox=manual_bbox,
            binary_threshold=args.binary_threshold,
            multilabel_threshold=args.multilabel_threshold,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # Batch
    elif args.dir:
        results = engine.predict_batch(args.dir, output_csv=args.output)
        print(json.dumps({
            "total": len(results),
            "success": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
        }, indent=2))

    else:
        parser.error("Either --image or --dir is required")


if __name__ == "__main__":
    main()
