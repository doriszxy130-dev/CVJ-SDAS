import argparse
import csv
import json
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
import importlib.util


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LOCAL_YOLOV8_PATH = Path(__file__).resolve().parents[1] / "yolov8"


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict spine ROI boxes with a trained YOLOv8 detector.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLOv8 best.pt.")
    parser.add_argument("--image-dir", type=Path, nargs="+", default=[script_dir / "data1"])
    parser.add_argument("--image-zip", type=Path, nargs="*", default=[])
    parser.add_argument("--output-csv", type=Path, default=script_dir / "outputs" / "03_predict_bboxes" / "predicted_bboxes.csv")
    parser.add_argument("--target-list", type=Path, default=None, help="Optional CSV with image/file_stem column; only predict these images.")
    parser.add_argument("--exclude-list", type=Path, default=None, help="Optional CSV/XLSX with image/file_stem column; skip these images.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=100, help="Number of images per YOLO predict batch (reduce if OOM).")
    parser.add_argument("--resume", action="store_true", help="Skip images already present in the output CSV.")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def setup_logging(output_csv, log_file):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = output_csv.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )
    return log_file


def maybe_extract(zips):
    for zip_path in zips:
        if not zip_path.exists():
            logging.warning("Zip not found: %s", zip_path)
            continue
        target = zip_path.with_suffix("")
        if target.exists() and any(target.rglob("*")):
            logging.info("Skip extraction, target already exists: %s", target)
            continue
        logging.info("Extracting %s", zip_path)
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if n and not n.endswith("/")]
                top_levels = {Path(n).parts[0] for n in names if Path(n).parts}
                extract_dir = zip_path.parent if target.name in top_levels else target
                extract_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_dir)
        else:
            try:
                with tarfile.open(zip_path) as tf:
                    names = [n for n in tf.getnames() if n and not n.endswith("/")]
                    top_levels = {Path(n).parts[0] for n in names if Path(n).parts}
                    extract_dir = zip_path.parent if target.name in top_levels else target
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    tf.extractall(extract_dir)
            except tarfile.TarError:
                import subprocess

                target.mkdir(parents=True, exist_ok=True)
                logging.info("Python tarfile failed; falling back to system tar for %s", zip_path)
                subprocess.run(["tar", "-xf", str(zip_path), "-C", str(target)], check=True)


def load_target_stems(target_list):
    if target_list is None or not target_list.exists():
        return None
    import pandas as pd

    if target_list.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(target_list)
    else:
        frame = pd.read_csv(target_list)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    col = columns.get("file_stem") or columns.get("image") or columns.get("filename") or frame.columns[0]
    return {Path(str(v).strip()).stem for v in frame[col].dropna()}


def iter_images(image_dirs, target_stems=None, exclude_stems=None):
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if target_stems is not None and path.stem not in target_stems:
                    continue
                if exclude_stems is not None and path.stem in exclude_stems:
                    continue
                yield path


def main():
    args = parse_args()
    log_file = setup_logging(args.output_csv, args.log_file)
    yolo_config_dir = (args.output_csv.parent / "ultralytics_config").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    if importlib.util.find_spec("ultralytics") is None and LOCAL_YOLOV8_PATH.exists():
        sys.path.insert(0, str(LOCAL_YOLOV8_PATH))
    logging.info("BBox prediction started")
    logging.info("Arguments: %s", vars(args))
    logging.info("YOLO_CONFIG_DIR=%s", os.environ["YOLO_CONFIG_DIR"])
    logging.info("Local YOLOv8 path=%s exists=%s", LOCAL_YOLOV8_PATH, LOCAL_YOLOV8_PATH.exists())
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Please install ultralytics first: pip install ultralytics") from exc

    model = YOLO(str(args.weights))
    maybe_extract(args.image_zip)
    target_stems = load_target_stems(args.target_list)
    if target_stems is not None:
        logging.info("Loaded %d target stems from %s", len(target_stems), args.target_list)
    exclude_stems = load_target_stems(args.exclude_list)
    if exclude_stems is not None:
        logging.info("Loaded %d exclude stems from %s", len(exclude_stems), args.exclude_list)
    images = [str(p) for p in iter_images(args.image_dir, target_stems=target_stems, exclude_stems=exclude_stems)]
    logging.info("Found %d images under %s", len(images), [str(p) for p in args.image_dir])

    if not images:
        logging.warning("No images found, exiting.")
        return

    # Resume: skip images already in output CSV
    if args.resume and args.output_csv.exists():
        existing = set()
        try:
            with args.output_csv.open("r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    existing.add(Path(row["image"]).stem)
            logging.info("Resume: loaded %d already-processed stems from %s", len(existing), args.output_csv)
        except Exception:
            pass
        if existing:
            before = len(images)
            images = [p for p in images if Path(p).stem not in existing]
            logging.info("Resume: skipping %d already done, %d remaining", before - len(images), len(images))

    BATCH_SIZE = args.batch_size
    detected = 0
    missing = []
    confidence_values = []

    append_mode = args.resume and args.output_csv.exists()
    open_mode = "a" if append_mode else "w"
    if append_mode:
        logging.info("Appending to existing output CSV")
    with args.output_csv.open(open_mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "xmin", "ymin", "xmax", "ymax", "confidence", "bbox_source"],
        )
        if not append_mode:
            writer.writeheader()
            f.flush()

        total = len(images)
        for idx, img_path_str in enumerate(images):
            img_path = Path(img_path_str)
            image_name = img_path.name  # 保留原始文件名

            result = model(img_path_str, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
            # 处理 result 可能是 list 的情况
            if isinstance(result, list):
                result = result[0]

            if result.boxes is None or len(result.boxes) == 0:
                missing.append(image_name)
                if (idx + 1) % 100 == 0:
                    logging.info("Progress %d/%d: %d detected, %d missing", idx + 1, total, detected, len(missing))
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            xmin, ymin, xmax, ymax = boxes[best_idx].tolist()
            confidence = float(confs[best_idx])
            confidence_values.append(confidence)
            writer.writerow(
                {
                    "image": image_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": confidence,
                    "bbox_source": "predicted",
                }
            )
            f.flush()
            detected += 1

            if (idx + 1) % 100 == 0:
                logging.info("Progress %d/%d: %d detected, %d missing", idx + 1, total, detected, len(missing))

    missing_csv = args.output_csv.with_name(args.output_csv.stem + "_missing.csv")
    with missing_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image"])
        writer.writeheader()
        for image_name in missing:
            writer.writerow({"image": image_name})

    summary = {
        "weights": str(args.weights),
        "image_dirs": [str(p) for p in args.image_dir],
        "output_csv": str(args.output_csv),
        "missing_csv": str(missing_csv),
        "log_file": str(log_file),
        "total_images": len(images),
        "detected_images": detected,
        "missing_images": len(missing),
        "mean_confidence": float(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
        "min_confidence": float(min(confidence_values)) if confidence_values else None,
        "max_confidence": float(max(confidence_values)) if confidence_values else None,
    }
    summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Saved predicted boxes to %s", args.output_csv)
    logging.info("Saved missing list to %s", missing_csv)
    logging.info("Saved summary to %s", summary_path)
    logging.info("BBox prediction finished: %s", summary)


if __name__ == "__main__":
    main()
