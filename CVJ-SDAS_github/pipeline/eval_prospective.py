"""Evaluate Binary and Multilabel models on the prospective test set (267 images)."""
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

# ── Paths ──
OUT_DIR = SCRIPT_DIR / 'outputs' / '06_prospective'
OUT_DIR.mkdir(parents=True, exist_ok=True)
BBOX_CSV = OUT_DIR / 'prospective_bboxes.csv'
LABEL_XLSX = SCRIPT_DIR / '前瞻金标准.xlsx'
ML_CKPT = SCRIPT_DIR / 'outputs' / '05_train_multilabel' / 'best_model.pt'
BN_CKPT = SCRIPT_DIR / 'outputs' / '05_train_binary' / 'best_model.pt'
IMAGE_DIR = SCRIPT_DIR / 'prospective_test' / 'image'

# Setup logging
for h in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(h)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(OUT_DIR / 'eval.log', mode='w', encoding='utf-8')])

logging.info("=== Prospective Test Evaluation ===")

# ── Load labels ──
label_df = pd.read_excel(LABEL_XLSX)
id_col = label_df.columns[0]
disease_cols = list(label_df.columns[1:7])

# Rename to standard names
label_df = label_df.rename(columns={
    id_col: 'image_id',
    disease_cols[0]: tc.LABEL_COLS[0],
    disease_cols[1]: tc.LABEL_COLS[1],
    disease_cols[2]: tc.LABEL_COLS[2],
    disease_cols[3]: tc.LABEL_COLS[3],
    disease_cols[4]: tc.LABEL_COLS[4],
    disease_cols[5]: tc.LABEL_COLS[5],
})
label_df['image_id'] = label_df['image_id'].astype(str).str.strip()
logging.info("Labels: %d rows, %d per class:\n%s", len(label_df),
             label_df[tc.LABEL_COLS].sum().sum(), label_df[tc.LABEL_COLS].sum().to_dict())

# ── Load bboxes ──
bbox_df = pd.read_csv(BBOX_CSV)
bbox_df['image'] = bbox_df['image'].astype(str).str.strip()
bbox_df['image_id'] = bbox_df['image'].apply(lambda x: Path(x).stem)
logging.info("BBoxes: %d entries", len(bbox_df))

# ── Build unified DataFrame matching SpineDataset format ──
# Merge labels + bboxes on image_id, add full path
merged = label_df.merge(bbox_df, on='image_id', how='inner')
logging.info("Matched: %d images (labels without bbox: %d, bbox without label: %d)",
             len(merged),
             len(label_df) - len(merged),
             len(bbox_df) - len(merged))

# Add data_source_dir for resolve_path() in SpineDataset
merged['data_source_dir'] = 'prospective_test/image'

# Add disease_present for binary
merged['disease_present'] = (merged[tc.LABEL_COLS].sum(axis=1) > 0).astype(int)

# Verify all images exist via the resolve_path logic
missing_files = []
for _, row in merged.iterrows():
    p = tc.resolve_path(row)
    if p is None:
        missing_files.append(row['image'])
if missing_files:
    logging.warning("Missing files: %d — %s", len(missing_files), missing_files[:5])
    merged = merged[~merged['image'].isin(missing_files)]

logging.info("Final dataset: %d images", len(merged))
logging.info("disease_present: %d positive, %d negative",
             merged['disease_present'].sum(), (merged['disease_present'] == 0).sum())

# ── Transform ──
tf_val = tc.transforms.Compose([
    tc.transforms.Resize((224, 224)), tc.transforms.ToTensor(),
    tc.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_eval(model, frame, is_binary, en_labels, out_subdir):
    """Run evaluation on given DataFrame using SpineDataset."""
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

    # Note: binary_metrics / multi_metrics expect LOGITS, they apply sgm internally
    if is_binary:
        m, p, pr = tc.binary_metrics(targets.ravel(), logits.ravel(), 0.5)
        logging.info("=== BINARY (%d images) ===", len(targets))
        logging.info("AUC=%.4f Acc=%.4f F1=%.4f ECE=%.4f", m['auc'], m['acc'], m['f1'], m['ece'])

        # targets for binary: 1D
        targets_1d = targets.ravel().astype(int)

        tc.plot_all_visualizations(targets_1d, p, pr, ['Normal', 'Disease'], True, out_dir)
        tc.per_label_report(targets_1d.reshape(-1, 1), pr.reshape(-1, 1), p.reshape(-1, 1),
                           ['Disease'], out_dir)

        # Save predictions with image IDs
        pred_df = frame[['image_id']].copy()
        pred_df['true'] = targets_1d
        pred_df['prob'] = p  # p is probs returned by binary_metrics (1D array)
        pred_df['pred'] = pr.astype(int)
        pred_df.to_csv(out_dir / 'predictions.csv', index=False, encoding='utf-8-sig')

        return {'auc': m['auc'], 'acc': m['acc'], 'f1': m['f1'], 'ece': m['ece']}

    else:
        m, p, pr = tc.multi_metrics(targets, logits, 0.5)  # pass LOGITS not probs!
        logging.info("=== MULTILABEL (%d images, %d label instances) ===",
                     len(targets), int(targets.sum()))
        logging.info("F1_macro=%.4f F1_micro=%.4f AUC_macro=%.4f Exact=%.4f",
                     m['f1_macro'], m['f1_micro'], m['auc_macro'], m['exact_match'])

        tc.plot_all_visualizations(targets, p, pr, en_labels, False, out_dir)
        tc.per_label_report(targets, pr, p, en_labels, out_dir)

        # Save predictions
        pred_dict = {'image_id': frame['image_id'].values}
        for i, name in enumerate(en_labels):
            pred_dict[f'true_{name}'] = targets[:, i].astype(int)
            pred_dict[f'prob_{name}'] = p[:, i]   # p = probs from multi_metrics
            pred_dict[f'pred_{name}'] = pr[:, i]
        pd.DataFrame(pred_dict).to_csv(out_dir / 'predictions.csv', index=False, encoding='utf-8-sig')

        # Per-label metrics
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

        return {
            'f1_macro': m['f1_macro'], 'f1_micro': m['f1_micro'],
            'auc_macro': m['auc_macro'], 'exact_match': m['exact_match'],
            'per_label': rows
        }


# ============================================================
#  MULTILABEL
# ============================================================
logging.info("\n" + "=" * 70)
logging.info("  MULTILABEL MODEL")
logging.info("=" * 70)

ckpt_ml = torch.load(ML_CKPT, map_location=dev, weights_only=False)
model_ml = tc.make_model('resnet50', 6, False).to(dev)
model_ml.load_state_dict(ckpt_ml['model_state'])
logging.info("Loaded multilabel from epoch %d (val F1_macro=%.4f)",
             ckpt_ml.get('epoch', -1), ckpt_ml.get('val_f1_macro', -1))

res_ml = run_eval(model_ml, merged, False, EN_LABELS, 'multilabel')

# ============================================================
#  BINARY
# ============================================================
logging.info("\n" + "=" * 70)
logging.info("  BINARY MODEL")
logging.info("=" * 70)

ckpt_bn = torch.load(BN_CKPT, map_location=dev, weights_only=False)
model_bn = tc.make_model('resnet50', 1, False).to(dev)
model_bn.load_state_dict(ckpt_bn['model_state'])
logging.info("Loaded binary from epoch %d (val F1=%.4f)",
             ckpt_bn.get('epoch', -1), ckpt_bn.get('val_f1', -1))

res_bn = run_eval(model_bn, merged, True, ['Disease'], 'binary')

# ============================================================
#  SUMMARY
# ============================================================
print()
print("=" * 70)
print("  PROSPECTIVE TEST SET — FINAL RESULTS (N=%d images)" % len(merged))
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
    'multilabel': {k: v for k, v in res_ml.items() if k != 'per_label'},
    'multilabel_per_label': res_ml['per_label'],
    'binary': res_bn,
}
with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

logging.info("All done! → %s", OUT_DIR)
