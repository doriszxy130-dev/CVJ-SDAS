"""公共无病集：合并 YOLO 预测框 + 中心fallback + 全0标签 -> 独立 CSV"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_COLS = ["寰椎前脱位", "寰椎后脱位", "颅底凹陷", "齿突不连", "寰椎枕化", "颈2-3分节不全"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def center_fallback_rows(stems, path_map):
    rows = []
    for s in stems:
        if s not in path_map:
            continue
        _, _, fpath = path_map[s]
        try:
            img = Image.open(fpath); w, h = img.size
        except:
            w, h = 2048, 2048
        cw, ch = w // 3, h // 3
        rows.append({
            "image": path_map[s][0], "file_stem": s,
            "xmin": (w - cw)//2, "ymin": (h - ch)//2,
            "xmax": (w - cw)//2 + cw, "ymax": (h - ch)//2 + ch,
            "bbox_confidence": 0.0, "bbox_source": "center_fallback",
            **{c: 0 for c in LABEL_COLS}, "disease_present": 0,
            "data_source_dir": path_map[s][1]
        })
    return rows

def main():
    dnames = ["CSXA_datasets-PNG", "F1000_X_ray_images_of_C_spine", "vindr_spinexr_train_images_png"]
    exts = {".bmp",".jpg",".jpeg",".png",".tif",".tiff"}
    pmap = {}
    for d in dnames:
        dp = SCRIPT_DIR / d
        if dp.exists():
            for fp in dp.rglob("*"):
                if fp.suffix.lower() in exts:
                    pmap[fp.stem] = (fp.name, d, str(fp))
    logging.info("Public image map: %d", len(pmap))

    # YOLO 预测框
    pred_csv = SCRIPT_DIR / "outputs/03_predict_bboxes/public_normal_predicted.csv"
    miss_csv = SCRIPT_DIR / "outputs/03_predict_bboxes/public_normal_predicted_missing.csv"

    rows = []
    if pred_csv.exists() and pred_csv.stat().st_size > 100:
        pdf = pd.read_csv(pred_csv)
        for _, r in pdf.iterrows():
            stem = Path(str(r["image"])).stem
            rows.append({
                "image": r["image"], "file_stem": stem,
                "xmin": r["xmin"], "ymin": r["ymin"], "xmax": r["xmax"], "ymax": r["ymax"],
                "bbox_confidence": float(r.get("confidence", 0.5)),
                "bbox_source": "predicted",
                **{c: 0 for c in LABEL_COLS}, "disease_present": 0,
                "data_source_dir": pmap.get(stem, (None, None, None))[1]
            })
        logging.info("Predicted rows: %d", len(rows))

    # 未检出
    missing_stems = set()
    if miss_csv.exists():
        mdf = pd.read_csv(miss_csv)
        missing_stems = {Path(str(s)).stem for s in mdf["image"]}
        logging.info("Missing stems: %d", len(missing_stems))

    # 中心fallback
    cf = center_fallback_rows(missing_stems, pmap)
    logging.info("Center fallback: %d", len(cf))
    rows.extend(cf)

    df = pd.DataFrame(rows)
    # 排序
    out_cols = ["image","xmin","ymin","xmax","ymax",
                *LABEL_COLS,"bbox_source","bbox_confidence",
                "data_source_dir","disease_present"]
    df = df[out_cols]
    out = SCRIPT_DIR / "outputs/04_unified_metadata/public_normal_3054_bbox_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n公共无病集: {len(df)} 行")
    print(f"  predicted: {(df['bbox_source']=='predicted').sum()}")
    print(f"  center_fallback: {(df['bbox_source']=='center_fallback').sum()}")
    print(f"  数据源: {df['data_source_dir'].value_counts().to_dict()}")
    print(f"  输出: {out}")

if __name__ == "__main__":
    main()
