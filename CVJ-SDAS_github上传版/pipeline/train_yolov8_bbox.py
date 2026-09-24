import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
import importlib.util


LOCAL_YOLOV8_PATH = Path(__file__).resolve().parents[1] / "yolov8"


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train YOLOv8 spine ROI detector and save stable best/last checkpoints.")
    parser.add_argument("--data", type=Path, default=script_dir / "outputs" / "01_prepare_yolov8_dataset" / "data.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs" / "02_train_yolov8_bbox")
    parser.add_argument("--name", default="spine_roi_yolov8s")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_yolov8_bbox.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )
    return log_path


def main():
    args = parse_args()
    log_path = setup_logging(args.output_dir)
    yolo_config_dir = (args.output_dir / "ultralytics_config").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    if importlib.util.find_spec("ultralytics") is None and LOCAL_YOLOV8_PATH.exists():
        sys.path.insert(0, str(LOCAL_YOLOV8_PATH))
    logging.info("YOLOv8 bbox training started")
    logging.info("Arguments: %s", vars(args))
    logging.info("YOLO_CONFIG_DIR=%s", os.environ["YOLO_CONFIG_DIR"])
    logging.info("Local YOLOv8 path=%s exists=%s", LOCAL_YOLOV8_PATH, LOCAL_YOLOV8_PATH.exists())

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Please install ultralytics in the active environment: pip install ultralytics") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        project=str(args.output_dir),
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(getattr(results, "save_dir", args.output_dir / args.name))
    weights_dir = run_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    best_dst = args.output_dir / "best_yolov8_bbox.pt"
    last_dst = args.output_dir / "last_yolov8_bbox.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        logging.info("Saved stable best checkpoint: %s", best_dst)
    else:
        logging.warning("Best checkpoint not found at %s", best_src)

    if last_src.exists():
        shutil.copy2(last_src, last_dst)
        logging.info("Saved stable last checkpoint: %s", last_dst)
    else:
        logging.warning("Last checkpoint not found at %s", last_src)

    summary = {
        "run_dir": str(run_dir),
        "weights_dir": str(weights_dir),
        "best_checkpoint": str(best_dst) if best_dst.exists() else None,
        "last_checkpoint": str(last_dst) if last_dst.exists() else None,
        "log_path": str(log_path),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    with (args.output_dir / "train_yolov8_bbox_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("YOLOv8 bbox training finished")


if __name__ == "__main__":
    main()
