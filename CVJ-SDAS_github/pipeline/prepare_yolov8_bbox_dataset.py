import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 bbox dataset from AAD-bbox-export.csv.")
    parser.add_argument("--bbox-csv", type=Path, default=script_dir / "AAD-bbox-export.csv")
    parser.add_argument("--image-dir", type=Path, default=script_dir / "data1")
    parser.add_argument("--image-zip", type=Path, default=script_dir / "data1.zip")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs" / "01_prepare_yolov8_dataset")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-class", action="store_true", default=True)
    parser.add_argument("--skip-empty", action="store_true", help="Skip rows whose bbox label is Empty.")
    return parser.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "prepare_yolov8_bbox_dataset.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )
    return log_path


def maybe_extract_images(image_dir, image_zip):
    if image_dir.exists() and any(image_dir.rglob("*")):
        return
    if image_zip.exists():
        logging.info("Extracting %s ...", image_zip)
        with zipfile.ZipFile(image_zip) as zf:
            zf.extractall(image_zip.parent)


def normalize_stem(value):
    return Path(str(value).strip()).stem


def patient_id_from_stem(stem):
    if stem.endswith("-2") or stem.endswith("-3"):
        return stem[:-2]
    return stem


def build_image_index(image_dir):
    image_index = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index.setdefault(path.stem, path)
    return image_index


def load_boxes(bbox_csv, skip_empty=False):
    rows = []
    with bbox_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row.get("label", "")).strip()
            if skip_empty and label.lower() == "empty":
                continue
            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
            except (KeyError, TypeError, ValueError):
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            stem = normalize_stem(row.get("image", ""))
            rows.append(
                {
                    "file_stem": stem,
                    "patient_id": patient_id_from_stem(stem),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "raw_label": label,
                }
            )
    return pd.DataFrame(rows)


def clip_box(xmin, ymin, xmax, ymax, width, height):
    xmin = max(0.0, min(float(width - 1), xmin))
    ymin = max(0.0, min(float(height - 1), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def yolo_line(box, width, height, class_id=0):
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"{class_id} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"


def copy_split(split_name, split_df, image_index, output_dir):
    image_out = output_dir / "images" / split_name
    label_out = output_dir / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for stem, group in split_df.groupby("file_stem"):
        source = image_index.get(stem)
        if source is None:
            skipped += 1
            continue
        target_image = image_out / source.name
        shutil.copy2(source, target_image)

        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size

        lines = []
        for _, row in group.iterrows():
            box = clip_box(row["xmin"], row["ymin"], row["xmax"], row["ymax"], width, height)
            if box is not None:
                lines.append(yolo_line(box, width, height, class_id=0))

        if not lines:
            skipped += 1
            target_image.unlink(missing_ok=True)
            continue

        (label_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied += 1

    return copied, skipped


def write_yaml(output_dir):
    data_yaml = output_dir / "data.yaml"
    text = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: spine_roi",
            "",
        ]
    )
    data_yaml.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    log_path = setup_logging(args.output_dir)
    logging.info("Preparing YOLOv8 bbox dataset")
    logging.info("Arguments: %s", vars(args))
    random.seed(args.seed)

    maybe_extract_images(args.image_dir, args.image_zip)
    image_index = build_image_index(args.image_dir)
    boxes = load_boxes(args.bbox_csv, skip_empty=args.skip_empty)
    boxes = boxes[boxes["file_stem"].isin(image_index.keys())].reset_index(drop=True)
    if boxes.empty:
        raise RuntimeError("No bbox rows matched local images.")

    image_level = boxes[["file_stem", "patient_id"]].drop_duplicates().reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(image_level, groups=image_level["patient_id"]))
    train_stems = set(image_level.iloc[train_idx]["file_stem"])
    val_stems = set(image_level.iloc[val_idx]["file_stem"])

    train_df = boxes[boxes["file_stem"].isin(train_stems)].reset_index(drop=True)
    val_df = boxes[boxes["file_stem"].isin(val_stems)].reset_index(drop=True)

    if args.output_dir.exists():
        logging.info("Output directory exists; files may be overwritten: %s", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_count, train_skipped = copy_split("train", train_df, image_index, args.output_dir)
    val_count, val_skipped = copy_split("val", val_df, image_index, args.output_dir)
    write_yaml(args.output_dir)

    split_df = image_level.copy()
    split_df["split"] = split_df["file_stem"].map(lambda x: "train" if x in train_stems else "val")
    split_df.to_csv(args.output_dir / "bbox_split.csv", index=False, encoding="utf-8-sig")

    logging.info("YOLOv8 dataset ready: %s", args.output_dir)
    logging.info("Train images: %d, skipped: %d", train_count, train_skipped)
    logging.info("Val images: %d, skipped: %d", val_count, val_skipped)
    logging.info("Train with: yolo detect train data=%s model=yolov8s.pt imgsz=640 epochs=100", args.output_dir / "data.yaml")

    summary = {
        "output_dir": str(args.output_dir),
        "data_yaml": str(args.output_dir / "data.yaml"),
        "log_path": str(log_path),
        "matched_box_rows": int(len(boxes)),
        "matched_images": int(image_level["file_stem"].nunique()),
        "train_images": int(train_count),
        "val_images": int(val_count),
        "train_skipped": int(train_skipped),
        "val_skipped": int(val_skipped),
    }
    with (args.output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
