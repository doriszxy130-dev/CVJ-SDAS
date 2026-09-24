"""Compute and plot per-patient (not just per-image) evaluation for both models."""
import sys, logging, json, numpy as np, torch
from pathlib import Path
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                              f1_score, precision_recall_fscore_support)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_classifier as tc

dev = torch.device('cuda')
tc.LABEL_COLS = ['寰椎前脱位', '寰椎后脱位', '颅底凹陷', '齿突不连', '寰椎枕化', '颈2-3分节不全']
en = ['Anterior AAD', 'Posterior AAD', 'Basilar Invagination',
      'Os Odontoideum', 'Occipitalization of Atlas', 'C2-3 Non-segmentation']

def setup_log(out_dir):
    for h in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_dir / 'train.log', mode='a', encoding='utf-8')])

def aggregate_patient(df, probs, preds, labels):
    """Group images by patient_id → patient-level predictions by max."""
    df = df.reset_index(drop=True)
    df['__prob0'] = probs[:, 0]
    df['__prob1'] = probs[:, 1]
    df['__prob2'] = probs[:, 2]
    df['__prob3'] = probs[:, 3]
    df['__prob4'] = probs[:, 4]
    df['__prob5'] = probs[:, 5]
    df['__pred0'] = preds[:, 0]
    df['__pred1'] = preds[:, 1]
    df['__pred2'] = preds[:, 2]
    df['__pred3'] = preds[:, 3]
    df['__pred4'] = preds[:, 4]
    df['__pred5'] = preds[:, 5]
    # labels
    for i, c in enumerate(tc.LABEL_COLS):
        df[f'__lab{i}'] = labels[:, i]

    grouped = df.groupby('patient_id')
    n_pat = grouped.ngroups
    pat_probs = np.zeros((n_pat, 6))
    pat_preds = np.zeros((n_pat, 6), dtype=int)
    pat_true = np.zeros((n_pat, 6), dtype=int)

    for pi, (pid, grp) in enumerate(grouped):
        for i in range(6):
            pat_probs[pi, i] = grp[f'__prob{i}'].max()
            pat_preds[pi, i] = int(grp[f'__pred{i}'].max())
            pat_true[pi, i] = int(grp[f'__lab{i}'].max())

    # Clean up temp cols
    return pat_probs, pat_preds, pat_true

def per_patient_binary_metrics(t, pred, p):
    """Binary patient-level metrics."""
    from sklearn.metrics import accuracy_score
    auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else np.nan
    return {
        'acc': float(accuracy_score(t, pred)),
        'f1': float(f1_score(t, pred, zero_division=0)),
        'auc': auc,
        'ece': tc._ece(t, p),
    }

# ──────────────────────────────────────────────────────────────
#  1. MULTILABEL — per-image + per-patient
# ──────────────────────────────────────────────────────────────
OUT_ML = SCRIPT_DIR / 'outputs' / '05_train_multilabel'
setup_log(OUT_ML)

retro = pd.read_csv(SCRIPT_DIR/'outputs'/'04_unified_metadata'/'retrospective_7830_bbox_labels.csv')
tc.seed_all(42)
df = tc.split_patients(retro, 0.15, 42)
train_df, val_df = df[0], df[1]

tf_val = tc.transforms.Compose([
    tc.transforms.Resize((224, 224)), tc.transforms.ToTensor(),
    tc.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds_vl = tc.SpineDataset(val_df, tf_val, 'crop', False, label_cols=tc.LABEL_COLS)
dl_vl = tc.DataLoader(ds_vl, 32, shuffle=False, num_workers=0, pin_memory=True)

ckpt = torch.load(OUT_ML / 'best_model.pt', map_location=dev)
model = tc.make_model('resnet50', 6, False).to(dev)
model.load_state_dict(ckpt['model_state'])
pos = train_df[tc.LABEL_COLS].sum(axis=0).astype(np.float32).values
pw = torch.tensor((len(train_df) - pos) / np.maximum(pos, 1.0), dtype=torch.float32, device=dev)
crit = tc.MultilabelLoss(pos_weight=pw, mutex_lambda=0.5)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=True)
vl, vt, vl_, _ = tc.run_epoch(model, dl_vl, crit, opt, scaler, dev,
                              is_train=False, epoch_num=0, log_batch_interval=0,
                              is_binary=False, amp_enabled=True)
m_img, p_img, pr_img = tc.multi_metrics(vt, vl_, 0.5)

# --- Per-image plots (already done, regenerate with clear title) ---
logging.info('=== PER-IMAGE (N=%d images, %d patients) ===', len(vt), val_df['patient_id'].nunique())
tc.plot_all_visualizations(vt, p_img, pr_img, en, False, OUT_ML)
tc.per_label_report(vt, pr_img, p_img, en, OUT_ML)

# --- Per-patient ---
p_pat, pr_pat, t_pat = aggregate_patient(val_df, p_img, pr_img, vt)
n_pat = len(t_pat)
logging.info('=== PER-PATIENT (N=%d patients) ===', n_pat)

# Compute per-patient metrics
m_pat, _, _ = tc.multi_metrics(t_pat, pr_pat, 0.5)
exact_pat = (pr_pat == t_pat).all(axis=1).sum()
logging.info('Per-Patient F1_macro=%.4f F1_micro=%.4f AUC=%.4f Exact=%.4f (%d/%d)',
             m_pat['f1_macro'], m_pat['f1_micro'], m_pat['auc_macro'],
             exact_pat/n_pat, int(exact_pat), n_pat)

# Per-patient confusion matrices — save separately
tc.plot_confusion_matrices(t_pat, pr_pat, en, False, OUT_ML)
# Rename to avoid overwriting per-image
import shutil
shutil.copy(OUT_ML / 'confusion_matrix.png', OUT_ML / 'confusion_matrix_patient.png')
tc._plot_combined_confusion(t_pat, pr_pat, en, OUT_ML)
shutil.copy(OUT_ML / 'confusion_matrix_combined.png', OUT_ML / 'confusion_matrix_combined_patient.png')
tc._plot_error_decomposition(t_pat, pr_pat, en, OUT_ML)
shutil.copy(OUT_ML / 'error_decomposition.png', OUT_ML / 'error_decomposition_patient.png')

# Now regenerate per-image with "_image" suffix
tc.plot_confusion_matrices(vt, pr_img, en, False, OUT_ML)
shutil.copy(OUT_ML / 'confusion_matrix.png', OUT_ML / 'confusion_matrix_image.png')
tc._plot_combined_confusion(vt, pr_img, en, OUT_ML)
shutil.copy(OUT_ML / 'confusion_matrix_combined.png', OUT_ML / 'confusion_matrix_combined_image.png')
tc._plot_error_decomposition(vt, pr_img, en, OUT_ML)
shutil.copy(OUT_ML / 'error_decomposition.png', OUT_ML / 'error_decomposition_image.png')

# Per-patient per-label report
tc.per_label_report(t_pat, pr_pat, p_pat, en, OUT_ML)
pr_pat_df = pd.DataFrame(tc.per_label_report(t_pat, pr_pat, p_pat, en, OUT_ML) if False else None)
# Actually call directly and save
rows = []
for i, name in enumerate(en):
    ti = t_pat[:, i]; pi = p_img[:, i]  # need to recompute p_pat properly
    prec, rec, f1, _ = precision_recall_fscore_support(t_pat[:, i], pr_pat[:, i], average='binary', zero_division=0)
    auc_v = float(roc_auc_score(t_pat[:, i], p_pat[:, i])) if len(np.unique(t_pat[:, i])) > 1 else np.nan
    rows.append({'Class': name, 'Precision': round(prec, 4), 'Recall': round(rec, 4),
                 'F1': round(f1, 4), 'AUC': round(auc_v, 4) if not np.isnan(auc_v) else 'N/A',
                 'ECE': round(tc._ece(t_pat[:, i], p_pat[:, i]), 4),
                 'Support': int(t_pat[:, i].sum())})
pd.DataFrame(rows).to_csv(OUT_ML / 'per_label_metrics_patient.csv', index=False, encoding='utf-8-sig')

logging.info('=== PER-PATIENT METRICS ===')
for _, r in pd.DataFrame(rows).iterrows():
    logging.info('  %-28s Prec=%.4f Rec=%.4f F1=%.4f AUC=%s ECE=%.4f Sup=%d',
                 r['Class'], r['Precision'], r['Recall'], r['F1'],
                 str(r['AUC']), r['ECE'], int(r['Support']))


# ──────────────────────────────────────────────────────────────
#  2. BINARY — per-image + per-patient
# ──────────────────────────────────────────────────────────────
OUT_BN = SCRIPT_DIR / 'outputs' / '05_train_binary'
setup_log(OUT_BN)

public = pd.read_csv(SCRIPT_DIR/'outputs'/'04_unified_metadata'/'public_normal_3054_bbox_labels.csv')
for c in tc.LABEL_COLS:
    if c not in public.columns: public[c] = 0
public['disease_present'] = 0; public['patient_id'] = 'pub_' + public.index.astype(str)
public['position'] = 'unknown'; retro['source'] = 'r'; public['source'] = 'p'
df_b = pd.concat([retro, public], ignore_index=True)
tc.seed_all(42)
df_b_s = tc.split_patients(df_b, 0.15, 42)
train_b, val_b = df_b_s[0], df_b_s[1]

ds_vb = tc.SpineDataset(val_b, tf_val, 'crop', True, label_cols=tc.LABEL_COLS)
dl_vb = tc.DataLoader(ds_vb, 32, shuffle=False, num_workers=0, pin_memory=True)

ckpt_b = torch.load(OUT_BN / 'best_model.pt', map_location=dev)
mb = tc.make_model('resnet50', 1, False).to(dev)
mb.load_state_dict(ckpt_b['model_state'])
pwb = torch.tensor([1.0], device=dev)
crit_b = tc.nn.BCEWithLogitsLoss(pos_weight=pwb)
vl_b, vt_b, vl_b_, _ = tc.run_epoch(mb, dl_vb, crit_b, torch.optim.AdamW(mb.parameters()),
                                     scaler, dev, is_train=False, epoch_num=0, log_batch_interval=0,
                                     is_binary=True, amp_enabled=True)
m_b, p_b, pr_b = tc.binary_metrics(vt_b, vl_b_, 0.5)

# Per-image binary
n_img_b = len(vt_b)
n_pat_b = val_b['patient_id'].nunique()
logging.info('=== BINARY PER-IMAGE (N=%d images, %d patients) ===', n_img_b, n_pat_b)
logging.info('AUC=%.4f Acc=%.4f F1=%.4f ECE=%.4f', m_b['auc'], m_b['acc'], m_b['f1'], m_b['ece'])
tc.plot_all_visualizations(vt_b, p_b, pr_b, ['Normal', 'Disease'], True, OUT_BN)
tc.per_label_report(vt_b, pr_b, p_b, ['Disease'], OUT_BN)

# Per-patient binary
val_b_reset = val_b.reset_index(drop=True)
val_b_reset['__disease_true'] = vt_b.reshape(-1).astype(int)
val_b_reset['__disease_pred'] = pr_b.astype(int)
val_b_reset['__disease_prob'] = p_b

pat_grp = val_b_reset.groupby('patient_id')
n_pat_binary = pat_grp.ngroups
pat_true_b = np.zeros(n_pat_binary, dtype=int)
pat_pred_b = np.zeros(n_pat_binary, dtype=int)
pat_prob_b = np.zeros(n_pat_binary)

for pi, (pid, grp) in enumerate(pat_grp):
    pat_true_b[pi] = int(grp['__disease_true'].max())
    pat_pred_b[pi] = int(grp['__disease_pred'].max())
    pat_prob_b[pi] = grp['__disease_prob'].max()

m_b_pat = per_patient_binary_metrics(pat_true_b, pat_pred_b, pat_prob_b)
logging.info('=== BINARY PER-PATIENT (N=%d patients) ===', n_pat_binary)
logging.info('AUC=%.4f Acc=%.4f F1=%.4f ECE=%.4f',
             m_b_pat['auc'], m_b_pat['acc'], m_b_pat['f1'], m_b_pat['ece'])

# Per-patient confusion matrix for binary
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cm_pat = confusion_matrix(pat_true_b, pat_pred_b, labels=[0, 1])
tn, fp, fn, tp = cm_pat.ravel()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_pat, cmap='Blues')
for i in range(2):
    for j in range(2):
        val = cm_pat[i, j]
        pct = val / n_pat_binary * 100
        ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if val > cm_pat.max()/2 else 'black')
ax.set_xticks([0,1]); ax.set_xticklabels(['Normal','Disease'], fontsize=12)
ax.set_yticks([0,1]); ax.set_yticklabels(['Normal','Disease'], fontsize=12)
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Binary — Patient-Level ({n_pat_binary} patients)\n'
             f'TP={tp} FP={fp} FN={fn} TN={tn}  Acc={(tp+tn)/n_pat_binary:.3f}',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); plt.tight_layout()
fig.savefig(OUT_BN / 'confusion_matrix_patient.png', dpi=150, bbox_inches='tight'); plt.close()

# Per-image for binary (rename)
shutil.copy(OUT_BN / 'confusion_matrix.png', OUT_BN / 'confusion_matrix_image.png')

# ROC for binary patient-level
if len(np.unique(pat_true_b)) > 1:
    fpr, tpr, _ = roc_curve(pat_true_b, pat_prob_b)
    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.plot(fpr, tpr, lw=2.5, label=f'AUC={m_b_pat["auc"]:.4f}')
    ax2.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
    ax2.set_xlim([0,1]); ax2.set_ylim([0,1.05])
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
    ax2.set_title(f'Binary — Patient-Level ROC ({n_pat_binary} patients)')
    ax2.legend(loc='lower right')
    plt.tight_layout()
    fig2.savefig(OUT_BN / 'roc_curves_patient.png', dpi=150); plt.close()

# Summary
print()
print('=' * 70)
print('  FINAL SUMMARY — Image-Level vs Patient-Level')
print('=' * 70)
print()
print('  MULTILABEL (6 classes):')
print(f'    Image-level  ({len(vt)} images,  {val_df["patient_id"].nunique()} patients):  F1_macro={m_img["f1_macro"]:.4f}  AUC={m_img["auc_macro"]:.4f}  Exact={m_img["exact_match"]:.4f}')
print(f'    Patient-level({n_pat} patients):  F1_macro={m_pat["f1_macro"]:.4f}  AUC={m_pat["auc_macro"]:.4f}  Exact={exact_pat/n_pat:.4f}')
print()
print('  BINARY:')
print(f'    Image-level  ({n_img_b} images,  {n_pat_b} patients):  AUC={m_b["auc"]:.4f}  Acc={m_b["acc"]:.4f}  F1={m_b["f1"]:.4f}')
print(f'    Patient-level({n_pat_binary} patients):  AUC={m_b_pat["auc"]:.4f}  Acc={m_b_pat["acc"]:.4f}  F1={m_b_pat["f1"]:.4f}')
print('=' * 70)
