"""
一次性构建两个最终 bbox+标签 数据集:
  1. 回顾性 7830 张 (人工框 + 预测框 + 中心fallback) -> outputs/04_unified_metadata/retrospective_7830_bbox_labels.csv
  2. 公共无病 3054 张 (YOLO预测框 + 中心fallback)          -> outputs/04_unified_metadata/public_normal_3054_bbox_labels.csv

输出格式参考 AAD-bbox-export.csv:
  image, xmin, ymin, xmax, ymax, label (6个疾病标签), bbox_source, bbox_confidence,
  patient_id, position, data_source_dir, disease_present
"""
import os, sys, csv, logging, json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "04_unified_metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["寰椎前脱位", "寰椎后脱位", "颅底凹陷", "齿突不连", "寰椎枕化", "颈2-3分节不全"]
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- helpers ----

def scan_images(dirs):
    """返回 {stem: (filename, dir_name, full_path)}"""
    m = {}
    for d in dirs:
        dp = SCRIPT_DIR / d
        if not dp.exists():
            continue
        for fp in dp.rglob("*"):
            if fp.suffix.lower() in IMAGE_EXTS:
                m[fp.stem] = (fp.name, d, str(fp))
    return m

def parse_pp(stem):
    if stem.endswith("-2"): return stem[:-2], "extension"
    elif stem.endswith("-3"): return stem[:-2], "flexion"
    else: return stem, "neutral"

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
            "image": path_map[s][0], "xmin": (w - cw)//2, "ymin": (h - ch)//2,
            "xmax": (w - cw)//2 + cw, "ymax": (h - ch)//2 + ch,
            "confidence": 0.0, "bbox_source": "center_fallback"
        })
    return rows

def load_labels():
    xp = SCRIPT_DIR / "X_RAY_Label.xlsx"
    df = pd.read_excel(xp)
    df = df.rename(columns={df.columns[0]: "file_stem"})
    df["file_stem"] = df["file_stem"].astype(str).str.strip()
    actual = df.columns[1:7].tolist()
    if set(actual) != set(LABEL_COLS):
        df = df.rename(columns=dict(zip(actual, LABEL_COLS)))
    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["disease_present"] = df[LABEL_COLS].max(axis=1).astype(int)
    return df

def load_bbox_csv(csv_path):
    """读 bbox CSV, 返回 DataFrame 含 image/stem/xmin/ymin/xmax/ymax/confidence/bbox_source"""
    if not csv_path.exists() or csv_path.stat().st_size < 50:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

def merge_and_enrich(bbox_rows, labels_df, path_map):
    """合并 bbox_rows + labels + 路径信息, 补齐中心fallback"""
    df = pd.DataFrame(bbox_rows)
    df["file_stem"] = df["image"].astype(str).str.strip().apply(lambda x: Path(x).stem)
    merged = df.merge(labels_df, on="file_stem", how="left")
    # 标签缺失填0
    for c in LABEL_COLS:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype(int)
        else:
            merged[c] = 0
    merged["disease_present"] = merged.get("disease_present", merged[LABEL_COLS].max(axis=1))
    merged["disease_present"] = merged["disease_present"].fillna(0).astype(int)
    # 路径
    merged["filename"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[0])
    merged["data_source_dir"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[1])
    merged["image_path"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[2])
    # 患者/体位
    pp = merged["file_stem"].apply(parse_pp)
    merged["patient_id"] = pp.apply(lambda x: x[0])
    merged["position"] = pp.apply(lambda x: x[1])
    # 数值
    for c in ["xmin","ymin","xmax","ymax"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged["bbox_confidence"] = pd.to_numeric(merged.get("confidence", merged.get("bbox_confidence", 0)), errors="coerce").fillna(0)

    # 去重: 同一 file_stem 优先 manual > predicted > center_fallback
    prio = {"manual": 0, "predicted": 1, "center_fallback": 2}
    merged["_p"] = merged["bbox_source"].map(prio).fillna(2)
    merged = merged.sort_values("_p").drop_duplicates("file_stem", keep="first").drop(columns=["_p"])

    return merged

def yolo_predict_images(image_list, model, args_conf, args_imgsz, args_device):
    """逐张推理，返回 bbox_rows 列表"""
    rows = []
    total = len(image_list)
    from ultralytics import YOLO
    detected = 0; missing = 0
    for idx, (img_name, img_path_str) in enumerate(image_list):
        res = model(img_path_str, conf=args_conf, imgsz=args_imgsz, device=args_device, verbose=False)
        if isinstance(res, list): res = res[0]
        if res.boxes is None or len(res.boxes) == 0:
            rows.append({
                "image": img_name, "xmin": None, "ymin": None, "xmax": None, "ymax": None,
                "confidence": 0.0, "bbox_source": "center_fallback_placeholder"
            })
            missing += 1
        else:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            bi = int(confs.argmax())
            rows.append({
                "image": img_name, "xmin": boxes[bi][0], "ymin": boxes[bi][1],
                "xmax": boxes[bi][2], "ymax": boxes[bi][3],
                "confidence": float(confs[bi]), "bbox_source": "predicted"
            })
            detected += 1
        if (idx+1) % 200 == 0:
            logging.info("  YOLO progress %d/%d, detected=%d missing=%d", idx+1, total, detected, missing)
    logging.info("  YOLO done: %d total, %d detected, %d missing", total, detected, missing)
    return rows

# ---- main ----

def main():
    # 加载标签
    labels_df = load_labels()
    logging.info("Loaded labels: %d rows", len(labels_df))

    # 扫描回顾性图片
    retro_dirs = ["data1","data2","data3","data4","data5","data6"]
    retro_map = scan_images(retro_dirs)
    logging.info("Retrospective images: %d", len(retro_map))

    # 扫描公共无病图片
    public_dirs = ["CSXA_datasets-PNG", "F1000_X_ray_images_of_C_spine", "vindr_spinexr_train_images_png"]
    public_map = scan_images(public_dirs)
    logging.info("Public normal images: %d", len(public_map))

    # ---- 1. 回顾性 7830 ----
    logging.info("=== Building retrospective 7830 ===")

    # 人工框
    man_csv = SCRIPT_DIR / "AAD-bbox-export.csv"
    man_df = load_bbox_csv(man_csv)
    if not man_df.empty:
        man_df = man_df[man_df["label"] != "Empty"]
        man_df["bbox_source"] = "manual"
        man_df["confidence"] = 1.0
        manual_rows = man_df.to_dict("records")
        logging.info("Manual bboxes: %d", len(manual_rows))
    else:
        manual_rows = []

    # 预测框
    pred_csv = SCRIPT_DIR / "outputs" / "03_predict_bboxes" / "retrospective_missing_manual.csv"
    pred_df = load_bbox_csv(pred_csv)
    if not pred_df.empty:
        pred_df["bbox_source"] = "predicted"
        predicted_rows = pred_df.to_dict("records")
        logging.info("Predicted bboxes: %d", len(predicted_rows))
    else:
        predicted_rows = []

    # 未检出 stem 列表
    miss_csv = SCRIPT_DIR / "outputs" / "03_predict_bboxes" / "retrospective_missing_manual_missing.csv"
    missing_stems = set()
    if miss_csv.exists():
        mdf = pd.read_csv(miss_csv)
        missing_stems = {Path(str(s)).stem for s in mdf["image"]}
        logging.info("Missing stems: %d", len(missing_stems))

    # 中心fallback
    # 实际上需要给“回顾性里所有不在 manual+predicted 中的 stem”加, 但 predicted 已经覆盖了除 missing 外的全部
    # 所以给 missing_stems 加中心框
    center_rows = center_fallback_rows(missing_stems, retro_map)
    logging.info("Center fallback: %d", len(center_rows))

    # 合并
    all_rows = manual_rows + predicted_rows + center_rows
    retro_merged = merge_and_enrich(all_rows, labels_df, retro_map)

    # 检查 — 应该有 7830 行左右
    logging.info("Retrospective unified: %d rows, manual=%d predicted=%d center=%d",
                 len(retro_merged),
                 (retro_merged["bbox_source"]=="manual").sum(),
                 (retro_merged["bbox_source"]=="predicted").sum(),
                 (retro_merged["bbox_source"]=="center_fallback").sum())

    # 确保覆盖 7830 (Excel 中有的但图片不存在的会缺行, 记录一下)
    label_stems = set(labels_df["file_stem"])
    map_stems = set(retro_map.keys())
    missing_from_disk = label_stems - map_stems
    if missing_from_disk:
        logging.warning("Label stems missing from disk: %d -> %s", len(missing_from_disk), list(missing_from_disk)[:10])

    # 保存回顾性 CSV (格式参考 AAD-bbox-export.csv)
    retro_out_cols = ["image","xmin","ymin","xmax","ymax",
                      *LABEL_COLS, "bbox_source","bbox_confidence",
                      "patient_id","position","data_source_dir","disease_present"]
    retro_csv = OUTPUT_DIR / "retrospective_7830_bbox_labels.csv"
    retro_merged[retro_out_cols].to_csv(retro_csv, index=False, encoding="utf-8-sig")
    logging.info("Saved retrospective: %s (%d rows)", retro_csv, len(retro_merged))

    # ---- 2. 公共无病 3054 ----
    logging.info("=== Building public normal dataset ===")
    # 加载 YOLO 模型
    sys.path.insert(0, str(SCRIPT_DIR.parent / "yolov8"))
    from ultralytics import YOLO
    weights = SCRIPT_DIR / "outputs" / "02_train_yolov8_bbox_allboxes" / "best_yolov8_bbox.pt"
    model = YOLO(str(weights))
    logging.info("YOLO model loaded for public set: %s", weights)

    # 准备公共集图片列表
    public_img_list = [(v[0], v[2]) for v in public_map.values()]
    logging.info("Public images to infer: %d", len(public_img_list))

    # YOLO 推理
    pub_rows = yolo_predict_images(public_img_list, model, conf=0.5, imgsz=640, device=None)

    # 给未检出的加中心fallback (bbox_source=center_fallback_placeholder 的)
    pub_missing = [r for r in pub_rows if r["bbox_source"] == "center_fallback_placeholder"]
    pub_detected = [r for r in pub_rows if r["bbox_source"] != "center_fallback_placeholder"]
    pub_missing_stems = {Path(r["image"]).stem for r in pub_missing}
    pub_center = center_fallback_rows(pub_missing_stems, public_map)

    # 公共集全是无病 (6标签全0)
    pub_labels_df = pd.DataFrame([
        {"file_stem": s, **{c: 0 for c in LABEL_COLS}, "disease_present": 0}
        for s in public_map.keys()
    ])

    pub_all = pub_detected + pub_center
    pub_merged = merge_and_enrich(pub_all, pub_labels_df, public_map)
    # 公共集填死 disease_present=0, 标签全0
    for c in LABEL_COLS:
        pub_merged[c] = 0
    pub_merged["disease_present"] = 0

    pub_csv = OUTPUT_DIR / "public_normal_3054_bbox_labels.csv"
    pub_out_cols = ["image","xmin","ymin","xmax","ymax",
                    *LABEL_COLS, "bbox_source","bbox_confidence",
                    "patient_id","position","data_source_dir","disease_present"]
    pub_merged[pub_out_cols].to_csv(pub_csv, index=False, encoding="utf-8-sig")
    logging.info("Saved public normal: %s (%d rows)", pub_csv, len(pub_merged))

    # ---- 汇总报告 ----
    print("\n" + "=" * 60)
    print("  最终框数据集构建完成")
    print("=" * 60)
    print(f"  回顾性 7830:")
    print(f"    总行数: {len(retro_merged)}")
    print(f"    人工框: {(retro_merged['bbox_source']=='manual').sum()}")
    print(f"    预测框: {(retro_merged['bbox_source']=='predicted').sum()}")
    print(f"    中心fallback: {(retro_merged['bbox_source']=='center_fallback').sum()}")
    print(f"    有病: {retro_merged['disease_present'].sum()}  无病: {len(retro_merged)-retro_merged['disease_present'].sum()}")
    print(f"    输出: {retro_csv}")
    print(f"  公共无病 3054:")
    print(f"    总行数: {len(pub_merged)}")
    print(f"    预测框: {(pub_merged['bbox_source']=='predicted').sum()}")
    print(f"    中心fallback: {(pub_merged['bbox_source']=='center_fallback').sum()}")
    print(f"    输出: {pub_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
