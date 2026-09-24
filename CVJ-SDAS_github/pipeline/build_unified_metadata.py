"""
合并所有 bbox 数据（人工 + 预测 + 中心fallback），关联 X_RAY_Label.xlsx 标签，
生成一张完整的 unified_metadata.csv，用于后续分类训练。

输入：
  - AAD-bbox-export.csv  (人工框, 1208张)
  - outputs/03_predict_bboxes/retrospective_missing_manual.csv  (预测框, 6369张)
  - outputs/03_predict_bboxes/retrospective_missing_manual_missing.csv  (未检出, 252张)
  - X_RAY_Label.xlsx  (7830条标签)
  - data1-data6 图片目录

输出：
  - outputs/04_unified_metadata/unified_bbox_labels.csv
    字段: file_stem, filename, patient_id, position, image_path,
          xmin, ymin, xmax, ymax, bbox_source, bbox_confidence,
          寰椎前脱位, 寰椎后脱位, 颅底凹陷, 齿突不连, 寰椎枕化, 颈2-3分节不全,
          disease_present, data_source_dir
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import csv
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "04_unified_metadata"
LABEL_COLS = ["寰椎前脱位", "寰椎后脱位", "颅底凹陷", "齿突不连", "寰椎枕化", "颈2-3分节不全"]
IMAGE_DIRS = ["data1", "data2", "data3", "data4", "data5", "data6"]
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_image_path_map():
    """扫描 data1-data6，建立 file_stem -> (filename, dir_name, full_path) 映射"""
    path_map = {}
    for dname in IMAGE_DIRS:
        dpath = SCRIPT_DIR / dname
        if not dpath.exists():
            continue
        for fpath in dpath.rglob("*"):
            if fpath.suffix.lower() in IMAGE_EXTS:
                stem = fpath.stem
                path_map[stem] = (fpath.name, dname, str(fpath))
    logging.info("Scanned %d images across %s", len(path_map), IMAGE_DIRS)
    return path_map


def parse_patient_position(stem):
    """从 file_stem 提取患者ID和体位
    例: '00A444628500-2' -> patient='00A444628500', position='extension'
        '00A444628500'   -> patient='00A444628500', position='neutral'
    """
    if stem.endswith("-2"):
        return stem[:-2], "extension"
    elif stem.endswith("-3"):
        return stem[:-2], "flexion"
    else:
        return stem, "neutral"


def load_manual_bboxes():
    """加载人工框, 去掉 label=Empty 的行"""
    mp = SCRIPT_DIR / "AAD-bbox-export.csv"
    if not mp.exists():
        logging.warning("AAD-bbox-export.csv not found")
        return pd.DataFrame(columns=["image", "xmin", "ymin", "xmax", "ymax", "bbox_source"])

    df = pd.read_csv(mp)
    # 去重: 同一张图可能有多个人工框, 取置信度最高/面积最大的
    # 这里直接保留所有人, 按 filename 去重时取第一个
    df = df[df["label"] != "Empty"].copy()
    df["bbox_source"] = "manual"
    df["bbox_confidence"] = 1.0  # 人工框置信度标记为1
    logging.info("Loaded %d manual bbox rows", len(df))
    return df[["image", "xmin", "ymin", "xmax", "ymax", "bbox_source", "bbox_confidence"]]


def load_predicted_bboxes():
    """加载 YOLO 预测框"""
    pp = SCRIPT_DIR / "outputs" / "03_predict_bboxes" / "retrospective_missing_manual.csv"
    if not pp.exists() or pp.stat().st_size < 100:
        logging.warning("Predicted bbox CSV empty or missing")
        return pd.DataFrame(columns=["image", "xmin", "ymin", "xmax", "ymax", "confidence", "bbox_source"])

    df = pd.read_csv(pp)
    df["bbox_source"] = "predicted"
    df["bbox_confidence"] = df["confidence"]
    logging.info("Loaded %d predicted bbox rows", len(df))
    return df[["image", "xmin", "ymin", "xmax", "ymax", "bbox_source", "bbox_confidence"]]


def add_center_fallback(missing_stems, path_map):
    """对未检出图用图像中心 1/3 范围生成 fallback box"""
    rows = []
    for stem in missing_stems:
        if stem not in path_map:
            continue
        fname, dname, fpath = path_map[stem]
        try:
            img = Image.open(fpath)
            w, h = img.size
        except Exception:
            w, h = 2048, 2048  # 默认尺寸

        # 中心 1/3 范围
        cw, ch = w // 3, h // 3
        xmin = (w - cw) // 2
        ymin = (h - ch) // 2
        xmax = xmin + cw
        ymax = ymin + ch
        rows.append({
            "image": fname,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "bbox_source": "center_fallback", "bbox_confidence": 0.0,
        })
    logging.info("Added %d center-fallback bboxes", len(rows))
    return pd.DataFrame(rows)


def load_labels():
    """加载 X_RAY_Label.xlsx 标签
    Excel 表头: File_Name | 寰椎前脱位 | 寰椎后脱位 | ... (6列)
    File_Name 值不带扩展名, 如 '331687', '331687-2'
    """
    xp = SCRIPT_DIR / "X_RAY_Label.xlsx"
    if not xp.exists():
        raise FileNotFoundError(f"X_RAY_Label.xlsx not found at {xp}")

    df = pd.read_excel(xp)
    col0 = df.columns[0]  # 'File_Name'
    # 重命名第一列为 file_stem
    df = df.rename(columns={col0: "file_stem"})
    df["file_stem"] = df["file_stem"].astype(str).str.strip()

    # 重命名6个标签列（Excel 第2-7列即为中文标签名，直接用位置映射）
    actual_label_cols = df.columns[1:7].tolist()
    logging.info("Excel label columns: %s", actual_label_cols)

    # 如果列名已经是中文标签名（UTF-8 解码后），直接使用
    # 否则按位置重命名为 LABEL_COLS
    if set(actual_label_cols) == set(LABEL_COLS):
        pass  # 已经匹配
    else:
        rename_map = dict(zip(actual_label_cols, LABEL_COLS))
        df = df.rename(columns=rename_map)

    # 确保标签列为 int
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 计算 disease_present
    df["disease_present"] = df[LABEL_COLS].max(axis=1).astype(int)
    logging.info("Loaded %d label rows, disease_pos=%d", len(df), df["disease_present"].sum())
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载标签
    labels_df = load_labels()
    labels_df["file_stem"] = labels_df["file_stem"].astype(str).str.strip()

    # 2. 构建图片路径映射
    path_map = build_image_path_map()

    # 3. 加载人工框 + 预测框
    manual_df = load_manual_bboxes()
    predicted_df = load_predicted_bboxes()

    # 4. 加载未检出列表
    missing_csv = SCRIPT_DIR / "outputs" / "03_predict_bboxes" / "retrospective_missing_manual_missing.csv"
    missing_stems = set()
    if missing_csv.exists():
        miss_df = pd.read_csv(missing_csv)
        missing_stems = {Path(s).stem for s in miss_df["image"]}
        logging.info("Loaded %d missing stems", len(missing_stems))

    # 5. 中心 fallback
    center_df = add_center_fallback(missing_stems, path_map)

    # 6. 合并所有 bbox
    all_bboxes = pd.concat([manual_df, predicted_df, center_df], ignore_index=True)
    # 从 bbox CSV 的 'image' 列提取 file_stem（去掉扩展名，去掉可能的前缀路径）
    all_bboxes["file_stem"] = all_bboxes["image"].astype(str).str.strip().apply(
        lambda x: Path(x).stem
    )

    # 7. 关联标签（Excel file_stem 无扩展名，bbox file_stem 也无扩展名，直接匹配）
    merged = all_bboxes.merge(labels_df, on="file_stem", how="left")

    # 标记缺失标签的行
    no_label = merged[LABEL_COLS[0]].isna()
    if no_label.any():
        logging.warning("%d bbox rows have NO matching label in X_RAY_Label.xlsx!", no_label.sum())
        # 填充默认值
        for col in LABEL_COLS:
            merged[col] = merged[col].fillna(0).astype(int)
        merged["disease_present"] = merged["disease_present"].fillna(0).astype(int)

    # 8. 关联图片路径
    merged["filename"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[0])
    merged["data_source_dir"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[1])
    merged["image_path"] = merged["file_stem"].map(lambda s: path_map.get(s, (None, None, None))[2].replace("\\", "/") if path_map.get(s) else None)

    # 9. 解析患者ID和体位
    patient_positions = merged["file_stem"].apply(parse_patient_position)
    merged["patient_id"] = patient_positions.apply(lambda x: x[0])
    merged["position"] = patient_positions.apply(lambda x: x[1])

    # 10. 确保数值列类型正确
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["bbox_confidence"] = pd.to_numeric(merged["bbox_confidence"], errors="coerce").fillna(0.0)

    # 11. 按患者排重: 同一 file_stem 优先人工框 > 预测框 > fallback
    priority = {"manual": 0, "predicted": 1, "center_fallback": 2}
    merged["_priority"] = merged["bbox_source"].map(priority)
    merged = merged.sort_values("_priority").drop_duplicates(subset=["file_stem"], keep="first")
    merged = merged.drop(columns=["_priority"])

    # 12. 整理输出列顺序
    output_cols = [
        "file_stem", "filename", "patient_id", "position", "image_path",
        "xmin", "ymin", "xmax", "ymax", "bbox_source", "bbox_confidence",
    ] + LABEL_COLS + ["disease_present", "data_source_dir"]
    merged = merged[output_cols]

    # 13. 保存
    out_path = OUTPUT_DIR / "unified_bbox_labels.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Saved unified metadata: %d rows -> %s", len(merged), out_path)

    # 14. 统计报告
    print("\n" + "=" * 60)
    print("  统一元数据构建完成")
    print("=" * 60)
    print(f"  总图数:          {len(merged)}")
    print(f"  人工框:          {(merged['bbox_source'] == 'manual').sum()}")
    print(f"  预测框:          {(merged['bbox_source'] == 'predicted').sum()}")
    print(f"  中心fallback:    {(merged['bbox_source'] == 'center_fallback').sum()}")
    print(f"  有病图:          {merged['disease_present'].sum()}")
    print(f"  无病图:          {len(merged) - merged['disease_present'].sum()}")
    print(f"  唯一患者数:      {merged['patient_id'].nunique()}")
    print(f"  体位分布:        {merged['position'].value_counts().to_dict()}")
    print(f"  数据源分布:      {merged['data_source_dir'].value_counts().to_dict()}")
    print(f"  标签分布:")
    for col in LABEL_COLS:
        print(f"    {col}: {merged[col].sum()}")
    print(f"  输出:            {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
