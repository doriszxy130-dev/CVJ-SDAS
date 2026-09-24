"""
Generate binary model figures for External Validation + Prospective Test.
Fig 7B: External Binary (CM + ROC)
Fig 8B: Prospective Binary (CM + ROC)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'outputs' / 'paper_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# ── External Validation Binary ──
print("Generating External Validation Binary...")
df_eb = pd.read_csv(SCRIPT_DIR / 'outputs/06_prospective/external_val/binary/predictions.csv')
T_eb = df_eb['true'].values.astype(int)
P_eb = df_eb['prob'].values
D_eb = (P_eb >= 0.5).astype(int)

cm_eb = confusion_matrix(T_eb, D_eb, labels=[0, 1])
tn, fp, fn_bn, tp_bn = cm_eb.ravel()
total_eb = len(T_eb)
acc_eb = accuracy_score(T_eb, D_eb)
f1_eb = f1_score(T_eb, D_eb, zero_division=0)
fpr_eb, tpr_eb, _ = roc_curve(T_eb, P_eb)
auc_eb = auc(fpr_eb, tpr_eb)

fig7 = plt.figure(figsize=(14, 6.5))
gs7 = GridSpec(1, 2, figure=fig7, wspace=0.25, left=0.05, right=0.97, top=0.88, bottom=0.12)

ax7a = fig7.add_subplot(gs7[0, 0])
im7 = ax7a.imshow(cm_eb, cmap='Blues')
for i in range(2):
    for j in range(2):
        val = cm_eb[i, j]; pct = val / total_eb * 100
        ax7a.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white' if val > cm_eb.max()/2 else 'black')
ax7a.set_xticks([0, 1]); ax7a.set_xticklabels(['Normal', 'Disease'], fontsize=12)
ax7a.set_yticks([0, 1]); ax7a.set_yticklabels(['Normal', 'Disease'], fontsize=12)
ax7a.set_xlabel('Predicted', fontsize=11); ax7a.set_ylabel('True', fontsize=11)
ax7a.set_title(f'A  Binary Confusion Matrix\n(N={total_eb}, TP={tp_bn} FP={fp} FN={fn_bn} TN={tn}, Acc={acc_eb:.4f})',
              fontsize=11, fontweight='bold', loc='left', pad=8)
plt.colorbar(im7, ax=ax7a, shrink=0.78)

ax7b = fig7.add_subplot(gs7[0, 1])
ax7b.plot(fpr_eb, tpr_eb, lw=3, color='#e74c3c', label=f'Binary ResNet50 (AUC={auc_eb:.4f})')
ax7b.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.3)
ax7b.set_xlim([0, 1]); ax7b.set_ylim([0, 1.02])
ax7b.set_xlabel('1 - Specificity', fontsize=11); ax7b.set_ylabel('Sensitivity', fontsize=11)
ax7b.set_title(f'B  ROC Curve (AUC={auc_eb:.4f}, Acc={acc_eb:.4f}, F1={f1_eb:.4f})',
              fontsize=11, fontweight='bold', loc='left')
ax7b.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax7b.set_aspect('equal'); ax7b.grid(alpha=0.15)

fig7.suptitle('Figure 9. External multi-center validation — binary disease screening (643 images, 4 hospitals)',
             fontsize=13, fontweight='bold', y=0.98)
fig7.savefig(OUT_DIR / 'Figure_9_External_Binary.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
fig7.savefig(OUT_DIR / 'Figure_9_External_Binary.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.close()
print(f'  Fig 9 done (AUC={auc_eb:.4f} Acc={acc_eb:.4f})')

# ── Prospective Binary ──
print("Generating Prospective Binary...")
df_pb = pd.read_csv(SCRIPT_DIR / 'outputs/06_prospective/binary/predictions.csv')
T_pb = df_pb['true'].values.astype(int)
P_pb = df_pb['prob'].values
D_pb = (P_pb >= 0.5).astype(int)

cm_pb = confusion_matrix(T_pb, D_pb, labels=[0, 1])
tn2, fp2, fn2, tp2 = cm_pb.ravel()
total_pb = len(T_pb)
acc_pb = accuracy_score(T_pb, D_pb)
f1_pb = f1_score(T_pb, D_pb, zero_division=0)
fpr_pb, tpr_pb, _ = roc_curve(T_pb, P_pb)
auc_pb = auc(fpr_pb, tpr_pb)

fig8 = plt.figure(figsize=(14, 6.5))
gs8 = GridSpec(1, 2, figure=fig8, wspace=0.25, left=0.05, right=0.97, top=0.88, bottom=0.12)

ax8a = fig8.add_subplot(gs8[0, 0])
im8 = ax8a.imshow(cm_pb, cmap='Blues')
for i in range(2):
    for j in range(2):
        val = cm_pb[i, j]; pct = val / total_pb * 100
        ax8a.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white' if val > cm_pb.max()/2 else 'black')
ax8a.set_xticks([0, 1]); ax8a.set_xticklabels(['Normal', 'Disease'], fontsize=12)
ax8a.set_yticks([0, 1]); ax8a.set_yticklabels(['Normal', 'Disease'], fontsize=12)
ax8a.set_xlabel('Predicted', fontsize=11); ax8a.set_ylabel('True', fontsize=11)
ax8a.set_title(f'A  Binary Confusion Matrix\n(N={total_pb}, TP={tp2} FP={fp2} FN={fn2} TN={tn2}, Acc={acc_pb:.4f})',
              fontsize=11, fontweight='bold', loc='left', pad=8)
plt.colorbar(im8, ax=ax8a, shrink=0.78)

ax8b = fig8.add_subplot(gs8[0, 1])
ax8b.plot(fpr_pb, tpr_pb, lw=3, color='#e74c3c', label=f'Binary ResNet50 (AUC={auc_pb:.4f})')
ax8b.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.3)
ax8b.set_xlim([0, 1]); ax8b.set_ylim([0, 1.02])
ax8b.set_xlabel('1 - Specificity', fontsize=11); ax8b.set_ylabel('Sensitivity', fontsize=11)
ax8b.set_title(f'B  ROC Curve (AUC={auc_pb:.4f}, Acc={acc_pb:.4f}, F1={f1_pb:.4f})',
              fontsize=11, fontweight='bold', loc='left')
ax8b.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax8b.set_aspect('equal'); ax8b.grid(alpha=0.15)

fig8.suptitle('Figure 10. Prospective test — binary disease screening (266 images)',
             fontsize=13, fontweight='bold', y=0.98)
fig8.savefig(OUT_DIR / 'Figure_10_Prospective_Binary.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
fig8.savefig(OUT_DIR / 'Figure_10_Prospective_Binary.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.close()
print(f'  Fig 10 done (AUC={auc_pb:.4f} Acc={acc_pb:.4f})')

print(f"\nDone. → {OUT_DIR}")
