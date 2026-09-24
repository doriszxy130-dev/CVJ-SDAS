"""Evaluate Binary and Multilabel models on external validation set (655 images, 4 hospitals)."""
import sys, logging, numpy as np, torch, json
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_classifier as tc

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tc.LABEL_COLS = ['寰椎前脱位', '寰椎后脱位', '颅底凹陷', '齿突不连', '寰椎枕化', '颈2-3分节不全']
EN_LABELS = ['Anterior AAD', 'Posterior AAD', 'Basilar Invagination (BI)',
             'Os Odontoideum (OO)', 'Occipitalization of Atlas', 'C2-3 Non-segmentation']

OUT_DIR = SCRIPT_DIR / 'outputs' / '06_prospective' / 'external_val'
OUT_DIR.mkdir(parents=True, exist_ok=True)
BBOX_CSV = SCRIPT_DIR / 'outputs' / '06_prospective' / 'external_val_bboxes.csv'
LABEL_XLSX = SCRIPT_DIR / '多中心标签.xlsx'
ML_CKPT = SCRIPT_DIR / 'outputs' / '05_train_multilabel' / 'best_model.pt'
BN_CKPT = SCRIPT_DIR / 'outputs' / '05_train_binary' / 'best_model.pt'
EXTERNAL_DIR = SCRIPT_DIR / 'external_val'

# Setup logging
for h in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(h)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(OUT_DIR / 'eval.log', mode='w', encoding='utf-8')])

logging.info("=== External Validation (4 hospitals) ===")

# ── Build image path mapping ──
image_map = {}  # {stem: (full_path, hospital_name)}
for hospital_dir in sorted(EXTERNAL_DIR.iterdir()):
    if not hospital_dir.is_dir():
        continue
    hname = hospital_dir.name
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for img_path in hospital_dir.glob(f'*{ext}'):
            if img_path.is_file():
                stem = img_path.stem
                if stem in image_map:
                    logging.warning("Duplicate stem %s: %s vs %s", stem, img_path, image_map[stem][0])
                image_map[stem] = (str(img_path), hname)
logging.info("Image map: %d entries across %d hospitals",
             len(image_map), len({v[1] for v in image_map.values()}))

# ── Load labels ──
label_df = pd.read_excel(LABEL_XLSX)
# Columns: 患者编号, 文件名, 1寰椎前脱位, 2寰椎后脱位, 3颅底凹陷, 4齿突不连, 5寰椎枕化, 6颈2-3分节不全
id_col = label_df.columns[0]   # patient ID
fname_col = label_df.columns[1]  # file name (e.g. 100012)
disease_cols = list(label_df.columns[2:8])

label_df = label_df.rename(columns={
    fname_col: 'image_id',
    disease_cols[0]: tc.LABEL_COLS[0],
    disease_cols[1]: tc.LABEL_COLS[1],
    disease_cols[2]: tc.LABEL_COLS[2],
    disease_cols[3]: tc.LABEL_COLS[3],
    disease_cols[4]: tc.LABEL_COLS[4],
    disease_cols[5]: tc.LABEL_COLS[5],
})
# Keep patient_id for potentially grouping later
label_df['patient_id'] = label_df[id_col].astype(str).str.strip()
label_df['image_id'] = label_df['image_id'].astype(str).str.strip()

# Fill NaN in label columns with 0
for c in tc.LABEL_COLS:
    label_df[c] = label_df[c].fillna(0).astype(int)

# Map image_id to (full_path, hospital_name, relative_dir, filename)
label_df['img_info'] = label_df['image_id'].map(lambda x: image_map.get(x))
label_df['hospital'] = label_df['img_info'].apply(lambda x: x[1] if x else None)
label_df['file_path'] = label_df['img_info'].apply(lambda x: x[0] if x else None)

no_img = label_df[label_df['img_info'].isna()]
if len(no_img) > 0:
    logging.warning("Labels without image file: %d — %s", len(no_img),
                    no_img['image_id'].head(10).tolist())
label_df = label_df.drop(columns=['img_info'])

logging.info("Labels: %d rows, disease distribution:\n%s",
             len(label_df), label_df[tc.LABEL_COLS].sum().to_dict())

# ── Load bboxes ──
bbox_df = pd.read_csv(BBOX_CSV)
bbox_df['image_id'] = bbox_df['image'].apply(lambda x: Path(str(x).strip()).stem)
bbox_df = bbox_df.set_index('image_id')
logging.info("BBoxes: %d entries", len(bbox_df))

# ── Merge: labels + bboxes ──
# Add actual filename from bbox CSV (has extension like .JPG)
bbox_df['bbox_file'] = bbox_df['image']  # original filename with extension

merged = label_df.merge(
    bbox_df[['bbox_file', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence']].reset_index(),
    left_on='image_id', right_on='image_id', how='inner')

# Build the columns SpineDataset.resolve_path() needs:
#   data_source_dir: relative path from SCRIPT_DIR to hospital folder
#   image: filename with extension (e.g. "100012.JPG")
merged['data_source_dir'] = merged.apply(
    lambda r: str(Path(r['file_path']).relative_to(SCRIPT_DIR).parent), axis=1)
merged['image'] = merged['bbox_file']
merged['bbox_source'] = 'predicted'

missing_bbox = label_df[~label_df['image_id'].isin(bbox_df.index)]
missing_label = set(bbox_df.index) - set(label_df['image_id'])
logging.info("Merged: %d images (missing bbox: %d, no label: %d)",
             len(merged), len(missing_bbox), len(missing_label))
if len(missing_bbox) > 0:
    logging.warning("Missing bbox samples: %d — %s", len(missing_bbox),
                    missing_bbox['image_id'].head(10).tolist())
if missing_label:
    logging.warning("Bbox without label: %s", list(missing_label)[:10])

# Keep only rows with valid file paths
merged = merged[merged['file_path'].notna()].copy()
logging.info("Final merged: %d images", len(merged))

# Add disease_present for binary
merged['disease_present'] = (merged[tc.LABEL_COLS].sum(axis=1) > 0).astype(int)
logging.info("disease_present: %d positive, %d negative",
             merged['disease_present'].sum(), (merged['disease_present'] == 0).sum())

# Per-hospital summary
for h in sorted(merged['hospital'].dropna().unique()):
    h_df = merged[merged['hospital'] == h]
    logging.info("  %s: %d images, %d positive, %d labels",
                 h, len(h_df), h_df['disease_present'].sum(),
                 int(h_df[tc.LABEL_COLS].sum().sum()))

# ── Transform ──
tf_val = tc.transforms.Compose([
    tc.transforms.Resize((224, 224)), tc.transforms.ToTensor(),
    tc.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_eval(model, frame, is_binary, en_labels, out_subdir):
    """Run evaluation using SpineDataset format."""
    out_dir = OUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = tc.SpineDataset(frame, tf_val, 'crop', is_binary, label_cols=tc.LABEL_COLS)
    loader = tc.DataLoader(ds, 32, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(dev)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(targets.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_labels, axis=0)

    if is_binary:
        m, p, pr = tc.binary_metrics(targets.ravel(), logits.ravel(), 0.5)
        logging.info("=== BINARY (%d images) ===", len(targets))
        logging.info("AUC=%.4f Acc=%.4f F1=%.4f ECE=%.4f", m['auc'], m['acc'], m['f1'], m['ece'])

        tc.plot_all_visualizations(targets.ravel().astype(int), p, pr,
                                   ['Normal', 'Disease'], True, out_dir)
        tc.per_label_report(targets.ravel().astype(int).reshape(-1, 1),
                           pr.reshape(-1, 1), p.reshape(-1, 1), ['Disease'], out_dir)

        # Save predictions
        pred_df = frame[['image_id', 'hospital', 'patient_id']].copy()
        pred_df['true'] = targets.ravel().astype(int)
        pred_df['prob'] = p
        pred_df['pred'] = pr.astype(int)
        pred_df.to_csv(out_dir / 'predictions.csv', index=False, encoding='utf-8-sig')

        # Per-hospital binary metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        for h in sorted(frame['hospital'].dropna().unique()):
            mask = frame['hospital'].values == h
            if mask.sum() == 0:
                continue
            th = targets.ravel()[mask].astype(int)
            ph = p[mask]
            prh = pr[mask].astype(int)
            try:
                auc_h = float(roc_auc_score(th, ph)) if len(np.unique(th)) > 1 else np.nan
            except:
                auc_h = np.nan
            logging.info("  %s (%d imgs): AUC=%.4f Acc=%.4f F1=%.4f",
                         h, mask.sum(), auc_h, accuracy_score(th, prh),
                         f1_score(th, prh, zero_division=0))

        return {'auc': m['auc'], 'acc': m['acc'], 'f1': m['f1'], 'ece': m['ece']}

    else:
        m, p, pr = tc.multi_metrics(targets, logits, 0.5)
        logging.info("=== MULTILABEL (%d images, %d label instances) ===",
                     len(targets), int(targets.sum()))
        logging.info("F1_macro=%.4f F1_micro=%.4f AUC_macro=%.4f Exact=%.4f",
                     m['f1_macro'], m['f1_micro'], m['auc_macro'], m['exact_match'])

        tc.plot_all_visualizations(targets, p, pr, en_labels, False, out_dir)
        tc.per_label_report(targets, pr, p, en_labels, out_dir)

        # Save predictions
        pred_dict = {'image_id': frame['image_id'].values,
                     'hospital': frame['hospital'].values,
                     'patient_id': frame['patient_id'].values}
        for i, name in enumerate(en_labels):
            pred_dict[f'true_{name}'] = targets[:, i].astype(int)
            pred_dict[f'prob_{name}'] = p[:, i]
            pred_dict[f'pred_{name}'] = pr[:, i]
        pd.DataFrame(pred_dict).to_csv(out_dir / 'predictions.csv', index=False, encoding='utf-8-sig')

        # Per-label + per-hospital metrics
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        rows = []
        for i, name in enumerate(en_labels):
            prec, rec, f1, _ = precision_recall_fscore_support(
                targets[:, i], pr[:, i], average='binary', zero_division=0)
            try:
                auc_v = float(roc_auc_score(targets[:, i], p[:, i]))
            except:
                auc_v = np.nan
            rows.append({
                'Class': name, 'Precision': round(prec, 4), 'Recall': round(rec, 4),
                'F1': round(f1, 4), 'AUC': round(auc_v, 4) if not np.isnan(auc_v) else 'N/A',
                'ECE': round(tc._ece(targets[:, i], p[:, i]), 4),
                'Support': int(targets[:, i].sum())
            })
        pd.DataFrame(rows).to_csv(out_dir / 'per_label_metrics.csv', index=False, encoding='utf-8-sig')

        # Per-hospital F1
        for h in sorted(frame['hospital'].dropna().unique()):
            mask = frame['hospital'].values == h
            if mask.sum() == 0:
                continue
            th = targets[mask]
            ph = p[mask]
            prh = pr[mask]
            try:
                f1_h = float(tc.f1_score(th, prh, average='macro', zero_division=0))
                exact_h = float((prh == th).all(axis=1).mean())
            except:
                f1_h, exact_h = float('nan'), float('nan')
            logging.info("  %s (%d imgs, %d labels): F1_macro=%.4f Exact=%.4f",
                         h, mask.sum(), int(th.sum()), f1_h, exact_h)

        return {
            'f1_macro': m['f1_macro'], 'f1_micro': m['f1_micro'],
            'auc_macro': m['auc_macro'], 'exact_match': m['exact_match'],
            'per_label': rows
        }


# ============================================================
#  MULTILABEL
# ============================================================
logging.info("\n" + "=" * 70)
logging.info("  MULTILABEL MODEL — External Validation (4 hospitals)")
logging.info("=" * 70)

ckpt_ml = torch.load(ML_CKPT, map_location=dev, weights_only=False)
model_ml = tc.make_model('resnet50', 6, False).to(dev)
model_ml.load_state_dict(ckpt_ml['model_state'])

res_ml = run_eval(model_ml, merged, False, EN_LABELS, 'multilabel')

# ============================================================
#  BINARY
# ============================================================
logging.info("\n" + "=" * 70)
logging.info("  BINARY MODEL — External Validation (4 hospitals)")
logging.info("=" * 70)

ckpt_bn = torch.load(BN_CKPT, map_location=dev, weights_only=False)
model_bn = tc.make_model('resnet50', 1, False).to(dev)
model_bn.load_state_dict(ckpt_bn['model_state'])

res_bn = run_eval(model_bn, merged, True, ['Disease'], 'binary')

# ============================================================
#  SUMMARY
# ============================================================
print()
print("=" * 70)
print("  EXTERNAL VALIDATION — FINAL RESULTS")
print(f"  {len(merged)} images across 4 hospitals")
print("=" * 70)
print()
print("  MULTILABEL (6 classes):")
print("    F1_macro = %.4f  |  F1_micro = %.4f  |  AUC_macro = %.4f  |  Exact = %.4f" %
      (res_ml['f1_macro'], res_ml['f1_micro'], res_ml['auc_macro'], res_ml['exact_match']))
for r in res_ml['per_label']:
    print("    %-32s F1=%.4f  AUC=%s  Sup=%d" % (r['Class'], r['F1'], str(r['AUC']), r['Support']))
print()
print("  BINARY:")
print("    AUC = %.4f  |  Acc = %.4f  |  F1 = %.4f  |  ECE = %.4f" %
      (res_bn['auc'], res_bn['acc'], res_bn['f1'], res_bn['ece']))
print()
print("=" * 70)

# Save summary JSON
summary = {
    'n_images': len(merged),
    'n_hospitals': merged['hospital'].nunique(),
    'hospitals': sorted(merged['hospital'].dropna().unique().tolist()),
    'multilabel': {k: v for k, v in res_ml.items() if k != 'per_label'},
    'multilabel_per_label': res_ml['per_label'],
    'binary': res_bn,
}
with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

logging.info("All done! → %s", OUT_DIR)
