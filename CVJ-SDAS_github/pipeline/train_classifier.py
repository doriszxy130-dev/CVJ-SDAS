"""
寰枢椎分类训练 — 并行独立训练两个模型:
  python train_classifier.py --mode binary       # 是否有病 (回顾性+公共无病)
  python train_classifier.py --mode multilabel   # 6类疾病 (回顾性, 含互斥约束)

可同时启动两个终端分别跑，互不依赖。

输出 (以 binary 为例, outputs/05_train_binary/):
  train.log              — 详细训练日志 (actionformer风格, 含batch级记录)
  history.csv            — 每epoch指标汇总
  best_model.pt          — 最佳验证指标模型
  final_model.pt         — 最后一轮模型
  checkpoint_epochNNN.pt — 每隔N轮保存
  confusion_matrix.png   — 混淆矩阵
  roc_curves.png         — ROC曲线
  calibration_curves.png — 校准曲线
  confidence_dist.png    — 置信度分布
  per_label_metrics.csv  — 逐标签Precision/Recall/F1/AUC/ECE
  val_report.txt         — sklearn classification_report
  deploy/best_model.pt   — 部署用模型权重
  deploy/config.json     — 部署配置
"""
import argparse, csv, io, json, logging, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_recall_fscore_support, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SCRIPT_DIR = Path(__file__).resolve().parent

# ── 全局 ──────────────────────────────────────────────────────────
LOG_FILE = None
LABEL_COLS = None   # 6个中文标签列名, 由 load_metadata 检测


# ═══════════════════════════════════════════════════════════════════
#  命令行
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="寰枢椎分类训练 — 二分类 / 多标签并行")
    p.add_argument("--mode", choices=["binary", "multilabel"], required=True)
    p.add_argument("--retro-csv", type=Path,
                   default=SCRIPT_DIR/"outputs"/"04_unified_metadata"/"retrospective_7830_bbox_labels.csv")
    p.add_argument("--public-csv", type=Path,
                   default=SCRIPT_DIR/"outputs"/"04_unified_metadata"/"public_normal_3054_bbox_labels.csv")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model", default="resnet50",
                   choices=["resnet18","resnet50","densenet121","efficientnet_b0"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--roi-mode", default="both", choices=["crop","full","both"])
    p.add_argument("--save-interval", type=int, default=5,
                   help="Save checkpoint every N epochs (default 5)")
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience on val loss plateau")
    p.add_argument("--mutex-lambda", type=float, default=0.5,
                   help="Mutual exclusion loss weight (multilabel only)")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision")
    p.add_argument("--log-batch-interval", type=int, default=0,
                   help="Log every N batches (0=auto, ~10 per epoch)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from checkpoint .pt")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  日志 & 工具
# ═══════════════════════════════════════════════════════════════════

def setup(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = out_dir / "train.log"
    # 每次都新建 log 文件 (不追加)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); root.addHandler(ch)


def flush_log():
    for h in logging.getLogger().handlers:
        h.flush()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def sgm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def format_time(s):
    if s < 60: return f"{s:.0f}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m:.0f}m{s:.0f}s"
    h, m = divmod(m, 60); return f"{h:.0f}h{m:.0f}m"


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════

def load_metadata(args):
    global LABEL_COLS
    retro = pd.read_csv(args.retro_csv)

    # 自动检测6个中文标签列
    skip = {"image","xmin","ymin","xmax","ymax","bbox_source","bbox_confidence",
            "patient_id","position","data_source_dir","disease_present",
            "source","filename","image_path","file_stem","_p"}
    LABEL_COLS = [c for c in retro.columns if c not in skip]
    if len(LABEL_COLS) != 6:
        # fallback: 硬编码（与统一CSV严格一致）
        LABEL_COLS = ["寰椎前脱位","寰椎后脱位","颅底凹陷","齿突不连","寰椎枕化","颈2-3分节不全"]
    logging.info("标签列: %s", LABEL_COLS)

    if args.mode == "binary":
        public = pd.read_csv(args.public_csv) if args.public_csv.exists() else pd.DataFrame()
        if "patient_id" not in public.columns:
            public["patient_id"] = "public_" + public.index.astype(str)
        if "position" not in public.columns:
            public["position"] = "unknown"
        for c in LABEL_COLS:
            if c not in public.columns:
                public[c] = 0
        public["disease_present"] = 0
        retro["source"] = "retrospective"
        public["source"] = "public"
        df = pd.concat([retro, public], ignore_index=True)
        if "patient_id" not in df.columns:
            df["patient_id"] = df["image"].apply(lambda x: Path(str(x)).stem)
        logging.info("二元分类总数据: %d (有病=%d, 无病=%d)",
                     len(df), int((df["disease_present"]==1).sum()),
                     int((df["disease_present"]==0).sum()))
    else:
        df = retro.copy()
        logging.info("多标签总数据: %d (有病=%d, 无病=%d)",
                     len(df), int((df["disease_present"]==1).sum()),
                     int((df["disease_present"]==0).sum()))
    return df


def resolve_path(row):
    d = str(row.get("data_source_dir", ""))
    fname = str(row.get("image", ""))
    if not d or not fname:
        return None
    p = SCRIPT_DIR / d / fname
    if p.exists():
        return str(p)
    for ext in [".bmp",".png",".jpg",".jpeg",".tif",".tiff"]:
        alt = SCRIPT_DIR / d / (Path(fname).stem + ext)
        if alt.exists():
            return str(alt)
    return None


def split_patients(df, val_size, seed):
    sp = GroupShuffleSplit(1, test_size=val_size, random_state=seed)
    groups = df["patient_id"].values if "patient_id" in df.columns else df.index
    ti, vi = next(sp.split(df, groups=groups))
    return df.iloc[ti].reset_index(drop=True), df.iloc[vi].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class SpineDataset(Dataset):
    def __init__(self, frame, transform, roi_mode, is_binary, label_cols=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.roi_mode = roi_mode
        self.is_binary = is_binary
        self.label_cols = label_cols or LABEL_COLS
        self.paths = []
        self.bboxes = []
        for _, r in frame.iterrows():
            self.paths.append(resolve_path(r))
            if pd.notna(r.get("xmin")) and pd.notna(r.get("xmax")):
                self.bboxes.append((float(r["xmin"]), float(r["ymin"]),
                                    float(r["xmax"]), float(r["ymax"])))
            else:
                self.bboxes.append(None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = None
        if self.paths[idx]:
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
        if img is None:
            img = Image.new("RGB", (224, 224))

        b = self.bboxes[idx]
        if b is not None:
            if self.roi_mode == "crop":
                img = self._crop(img, b)
            elif self.roi_mode == "both" and random.random() < 0.5:
                img = self._crop(img, b)

        img = self.transform(img)
        if self.is_binary:
            y = torch.tensor([float(row["disease_present"])], dtype=torch.float32)
        else:
            y = torch.tensor(row[self.label_cols].astype(np.float32).values,
                           dtype=torch.float32)
        return img, y

    @staticmethod
    def _crop(img, b):
        w, h = img.size
        xmin, ymin, xmax, ymax = b
        bw, bh = xmax - xmin, ymax - ymin
        left = max(0, int(xmin - bw * 0.1))
        top = max(0, int(ymin - bh * 0.1))
        right = min(w, int(xmax + bw * 0.1))
        bottom = min(h, int(ymax + bh * 0.1))
        if right > left and bottom > top:
            return img.crop((left, top, right, bottom))
        return img


# ═══════════════════════════════════════════════════════════════════
#  模型
# ═══════════════════════════════════════════════════════════════════

def make_model(name, nc, pretrained):
    if name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, nc)
    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w); m.fc = nn.Linear(m.fc.in_features, nc)
    elif name == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=w); m.classifier = nn.Linear(m.classifier.in_features, nc)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    else:
        raise ValueError(name)
    return m


# ═══════════════════════════════════════════════════════════════════
#  损失函数 (含互斥约束)
# ═══════════════════════════════════════════════════════════════════

class MultilabelLoss(nn.Module):
    """BCEWithLogitsLoss + 寰椎前脱位/寰椎后脱位互斥惩罚"""
    def __init__(self, pos_weight, mutex_lambda=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.mutex_lambda = mutex_lambda

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        # 互斥惩罚: sigmoid(logits[:,0]) * sigmoid(logits[:,1]) 的平均
        if self.mutex_lambda > 0:
            p0 = torch.sigmoid(logits[:, 0])
            p1 = torch.sigmoid(logits[:, 1])
            mutex_loss = (p0 * p1).mean()
            return bce_loss + self.mutex_lambda * mutex_loss, {
                "bce": bce_loss.item(), "mutex": mutex_loss.item()
            }
        return bce_loss, {"bce": bce_loss.item(), "mutex": 0.0}


# ═══════════════════════════════════════════════════════════════════
#  训练 & 验证 单epoch (含batch级日志)
# ═══════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, opt, scaler, dev, is_train,
              epoch_num, log_batch_interval, is_binary, amp_enabled):
    """训练/验证 一个epoch, 返回 (avg_loss, targets, logits, loss_components)"""
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_t, all_l = [], []
    comp_sums = defaultdict(float)
    n_batches = len(loader)
    t_start = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    phase = "Train" if is_train else "Val"

    with ctx:
        for bi, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(dev), labels.to(dev)
            bs = imgs.size(0)

            if is_train:
                opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled and dev.type == "cuda"):
                logits = model(imgs)
                if is_binary:
                    loss = criterion(logits, labels)
                    comps = {"loss": loss.item()}
                else:
                    loss, comps = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            total_loss += loss.item() * bs
            n += bs
            for k, v in comps.items():
                comp_sums[k] += v * bs

            all_t.append(labels.detach().cpu().numpy())
            all_l.append(logits.detach().cpu().numpy())

            # ── batch级日志 (actionformer风格) ──
            if log_batch_interval > 0 and ((bi + 1) % log_batch_interval == 0 or bi == n_batches - 1):
                avg_l = total_loss / n
                elapsed = time.time() - t_start
                batch_time = elapsed / (bi + 1)
                # 组件平均值
                comp_str = ""
                for k in sorted(comp_sums.keys()):
                    comp_str += f"  {k} {comp_sums[k]/n:.4f} ({comp_sums[k]/n:.4f})"
                logging.info("Epoch: [%03d][%05d/%05d]  Time %.2f (%.2f)  Loss %.4f (%.4f)%s",
                             epoch_num, bi + 1, n_batches, batch_time, batch_time,
                             loss.item(), avg_l, comp_str)

    avg_loss = total_loss / n
    elapsed = time.time() - t_start
    comp_avg = {k: v / n for k, v in comp_sums.items()}

    logging.info("[%s]: Epoch %d finished | Loss=%.6f | Time=%s | Samples=%d",
                 phase, epoch_num, avg_loss, format_time(elapsed), n)

    return avg_loss, np.concatenate(all_t), np.concatenate(all_l), comp_avg


# ═══════════════════════════════════════════════════════════════════
#  验证指标计算
# ═══════════════════════════════════════════════════════════════════

def binary_metrics(t, l, th):
    """t:(N,1)  l:(N,1)  返回 metrics, probs, preds"""
    p = sgm(l).reshape(-1)
    t = t.reshape(-1).astype(int)
    pred = (p >= th).astype(int)
    auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else np.nan
    return {
        "acc": float(accuracy_score(t, pred)),
        "f1": float(f1_score(t, pred, zero_division=0)),
        "auc": auc,
        "ece": _ece(t, p),
    }, p, pred


def multi_metrics(t, l, th):
    """t:(N,6)  l:(N,6)  返回 metrics, probs, preds"""
    p = sgm(l)
    pred = (p >= th).astype(int)
    n_cls = t.shape[1]
    aucs = []
    for i in range(n_cls):
        if len(np.unique(t[:, i])) > 1:
            aucs.append(float(roc_auc_score(t[:, i], p[:, i])))
        else:
            aucs.append(np.nan)
    f1_macro = float(f1_score(t, pred, average="macro", zero_division=0))
    f1_micro = float(f1_score(t, pred, average="micro", zero_division=0))
    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "exact_match": float((pred == t).all(axis=1).mean()),
        "auc_macro": float(np.nanmean(aucs)),
        "auc_per_class": aucs,
        "ece": _ece_multi(t, p),
    }, p, pred


def _ece(t, p, B=10):
    """Expected Calibration Error (binary)"""
    t = t.reshape(-1); p = p.reshape(-1)
    e = 0.0
    for lo in np.linspace(0, 1, B + 1)[:-1]:
        hi = lo + 1.0 / B
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() > 0:
            e += m.sum() / len(t) * abs(p[m].mean() - t[m].mean())
    return float(e)


def _ece_multi(t, p, B=10):
    return float(np.mean([_ece(t[:, i], p[:, i], B) for i in range(t.shape[1])]))


def per_label_report(t, pred, p, label_names, out_dir):
    """逐标签详细报告 (CSV + 打印)"""
    # 确保2D数组 (binary模式时p/pred为1D)
    if t.ndim == 1: t = t.reshape(-1, 1)
    if p.ndim == 1: p = p.reshape(-1, 1)
    if pred.ndim == 1: pred = pred.reshape(-1, 1)
    rows = []
    for i, name in enumerate(label_names):
        ti = t[:, i]; pi = p[:, i]; predi = pred[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(ti, predi, average="binary", zero_division=0)
        auc = float(roc_auc_score(ti, pi)) if len(np.unique(ti)) > 1 else np.nan

        # 混淆矩阵
        cm = confusion_matrix(ti, predi, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        rows.append({
            "Class": name,
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "AUC": round(auc, 6) if not np.isnan(auc) else "N/A",
            "ECE": round(_ece(ti, pi), 6),
            "Support": int(ti.sum()),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_label_metrics.csv", index=False, encoding="utf-8-sig")

    logging.info("=" * 75)
    logging.info("  Per-Label Evaluation Report")
    logging.info("=" * 75)
    logging.info("  %-18s %8s %8s %8s %8s %8s %8s",
                 "Class", "Prec", "Recall", "F1", "AUC", "ECE", "Support")
    logging.info("  " + "-" * 73)
    for _, r in df.iterrows():
        logging.info("  %-18s %8.4f %8.4f %8.4f %8s %8.4f %8d",
                     r["Class"], r["Precision"], r["Recall"], r["F1"],
                     f'{r["AUC"]:.4f}' if isinstance(r["AUC"], float) else r["AUC"],
                     r["ECE"], int(r["Support"]))
    logging.info("=" * 75)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Label mapping: Chinese → English (for matplotlib rendering)
# ═══════════════════════════════════════════════════════════════════

CHINESE_TO_ENGLISH = {
    "寰椎前脱位": "Anterior AAD",
    "寰椎后脱位": "Posterior AAD",
    "颅底凹陷": "Basilar Invagination (BI)",
    "齿突不连": "Os Odontoideum (OO)",
    "寰椎枕化": "Occipitalization of Atlas",
    "颈2-3分节不全": "C2-3 Non-segmentation",
}

def _en_labels(names):
    """Translate label names to English (fallback to original if no mapping)."""
    return [CHINESE_TO_ENGLISH.get(n, n) for n in names]

def _set_matplotlib_font():
    """Try to configure a font that supports both Chinese and English."""
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        # Try common CJK fonts on Windows
        for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
            available = [f.name for f in fm.fontManager.ttflist]
            if font_name in available:
                matplotlib.rcParams["font.family"] = font_name
                return
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
#  可视化: 混淆矩阵 / ROC / 校准 / 置信度
# ═══════════════════════════════════════════════════════════════════

def plot_all_visualizations(t, p, pred, label_names, is_binary, out_dir):
    """绘制并保存所有评估图"""
    _set_matplotlib_font()
    plot_confusion_matrices(t, pred, label_names, is_binary, out_dir)
    plot_roc_curves(t, p, label_names, is_binary, out_dir)
    plot_calibration_curves(t, p, label_names, is_binary, out_dir)
    plot_confidence_distributions(p, t, label_names, is_binary, out_dir)


def plot_confusion_matrices(t, pred, label_names, is_binary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    en_names = _en_labels(label_names)

    if is_binary:
        # ── Binary: 2x2 with total ──
        cm = confusion_matrix(t.reshape(-1).astype(int), pred.astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = int(tn + fp + fn + tp)
        acc = (tp + tn) / total if total > 0 else 0

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                cell_text = f"{val}\n({pct:.1f}%)"
                ax.text(j, i, cell_text, ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if val > cm.max() / 2 else "black")

        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Disease"], fontsize=12)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Disease"], fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"Binary Classification — Confusion Matrix\n"
                     f"(N={total}  |  TP={tp}  FP={fp}  FN={fn}  TN={tn}  |  Acc={acc:.3f})",
                     fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        # ── Label-level 6×6: each cell = count of labels (true=i, pred=j) ──
        n = len(label_names)
        mat = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                mat[i, j] = int(((t[:, i] == 1) & (pred[:, j] == 1)).sum())
        row_sums = np.array([int(t[:, i].sum()) for i in range(n)])
        total_label_instances = int(row_sums.sum())

        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

        for i in range(n):
            for j in range(n):
                pct = mat[i, j] / row_sums[i] * 100 if row_sums[i] > 0 else 0
                if i == j:
                    label = f"TP\n{mat[i,j]}\n({pct:.0f}%)"
                    color = "white" if pct > 50 else "black"
                    fontsize = 12
                else:
                    label = f"{mat[i,j]}"
                    color = "white" if pct > 30 else "black"
                    fontsize = 9
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=fontsize, fontweight="bold", color=color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(en_names, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(en_names, fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Label-Level Confusion Matrix — {total_label_instances} label instances in {len(t)} images\n"
                     f"Cell (i,j) = count of TRUE disease i instances that were PREDICTED as disease j\n"
                     f"Diagonal = Correctly predicted (TP, Recall%)  |  "
                     f"Off-diagonal = disease i also predicted as j (comorbidity, not misclassification)\n"
                     f"Each row sums to total true-positive instances of that disease",
                     fontsize=11, fontweight="bold")

        # Right-side: per-class true positive count
        for i in range(n):
            ax.text(n + 0.6, i, f"N={row_sums[i]}", va="center", fontsize=10, color="dimgray")

        plt.colorbar(im, ax=ax, shrink=0.80, label="Label-instance count")
        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("  标签级6×6混淆矩阵(N=%d个标签实例)已保存: %s", total_label_instances, out_dir / "confusion_matrix.png")

        # ── (b) Label-level error decomposition ──
        _plot_error_decomposition(t, pred, en_names, out_dir)

        # ── (c) Image-level: 3-category diagnostic ──
        _plot_image_level_diagnostic(t, pred, en_names, out_dir)

    logging.info("  混淆矩阵已保存: %s", out_dir / "confusion_matrix.png")


def _plot_image_level_diagnostic(t, pred, label_names, out_dir):
    """Image-level 3-category diagnostic: Complete Match / Miss / Error.
    Evaluated per image (1153 images), not per label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = len(t)
    n_labels = len(label_names)

    # Binary detection: any disease present?
    gt_binary = t.max(axis=1).astype(int)
    pred_binary = pred.max(axis=1).astype(int)

    # Per-image correct label count
    correct_per_image = (pred == t).sum(axis=1)  # 0..6

    # Three categories
    complete = (correct_per_image == n_labels)  # all 6 correct

    miss_detect = (pred_binary == 0) & (gt_binary == 1)  # FN detection
    miss_label = ((pred_binary == 1) & (gt_binary == 1) &
                  ((pred == 0) & (t == 1)).any(axis=1))  # detection OK, but missed label
    miss_total = miss_detect | miss_label

    error_detect = (pred_binary == 1) & (gt_binary == 0)  # FP detection
    error_label = ((pred_binary == 1) & (gt_binary == 1) &
                   ((pred == 1) & (t == 0)).any(axis=1))  # detection OK, but extra label
    error_total = error_detect | error_label

    # Priority: if both miss AND error, count as error
    miss_only = miss_total & ~error_total & ~complete
    error_only = error_total & ~complete
    # Remaining
    other = ~complete & ~miss_only & ~error_only

    cm = complete.sum()
    miss_n = miss_only.sum()
    err_n = error_only.sum()

    # ── Plot: left = 3-category pie/donut, right = per-image correct label distribution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: 3-category bar
    cats = ['Complete\nMatch', 'Misdiagnosis\n(Miss)', 'Misdiagnosis\n(Error)']
    counts = [cm, miss_n, err_n]
    colors_bar = ['#4CAF50', '#FF9800', '#F44336']
    bars = ax1.bar(cats, counts, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        pct = cnt / n_images * 100
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                 f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    ax1.set_ylabel('Number of images', fontsize=11)
    ax1.set_title(f"Image-Level Diagnostic Assessment\n(N={n_images} images)",
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.grid(axis='y', alpha=0.3)

    # Sub-text for each category
    det_fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())
    det_fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
    ax1.text(0, max(counts) * 1.18,
             f'Binary correct | All 6 labels exact',
             ha='center', fontsize=7.5, color='dimgray', style='italic')
    ax1.text(1, max(counts) * 1.18,
             f'Missed disease present ({det_fn} FN detection) | Missed label',
             ha='center', fontsize=7.5, color='dimgray', style='italic')
    ax1.text(2, max(counts) * 1.18,
             f'Predicted disease not present ({det_fp} FP detection) | Extra label',
             ha='center', fontsize=7.5, color='dimgray', style='italic')

    # Right: per-image correct label distribution
    dist = [int((correct_per_image == k).sum()) for k in range(n_labels + 1)]
    colors_dist = ['#F44336'] + ['#FF9800'] * 1 + ['#FFEB3B'] * 1 + ['#8BC34A'] * 1 + ['#4CAF50'] * 2 + ['#2E7D32']
    # simple clean bar chart
    x_labels = [f'{k}/6' for k in range(n_labels + 1)]
    bars2 = ax2.bar(x_labels, dist, color=['#d32f2f','#e64a19','#f57c00','#689f38','#388e3c','#2e7d32','#1b5e20'],
                    alpha=0.85, edgecolor='black', linewidth=0.3)

    for bar, cnt in zip(bars2, dist):
        if cnt > 0:
            pct = cnt / n_images * 100
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    ax2.set_xlabel('Correct labels per image', fontsize=11)
    ax2.set_ylabel('Number of images', fontsize=11)
    ax2.set_title(f'Per-Image Correct Label Distribution (N={n_images} images)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(dist) * 1.25)
    ax2.grid(axis='y', alpha=0.3)

    # Summary annotation
    avg_correct = correct_per_image.mean()
    ax2.text(0.5, 0.95,
             f'Mean: {avg_correct:.2f}/6 correct  |  '
             f'≥5/6: {(correct_per_image >= 5).sum()}/{n_images} ({(correct_per_image >= 5).sum()/n_images*100:.1f}%)',
             transform=ax2.transAxes, ha='center', fontsize=9, color='dimgray')

    plt.tight_layout()
    fig.savefig(out_dir / "diagnostic_image_level.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  图像级诊断评估已保存: %s", out_dir / "diagnostic_image_level.png")


def _plot_combined_confusion(t, pred, label_names, out_dir):
    """6x6 co-occurrence matrix: rows=True label, cols=Predicted positive.
    Cell (i,j): patients with TRUE disease i AND PREDICTED disease j.
    Diagonal = correctly identified. Off-diagonal = comorbidity overlap.
    NOTE: High off-diagonals DO NOT mean errors — they reflect that
    many patients truly have multiple diseases simultaneously."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(label_names)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            mat[i, j] = int(((t[:, i] == 1) & (pred[:, j] == 1)).sum())
    row_sums = np.array([int(t[:, i].sum()) for i in range(n)])

    mat_pct = np.zeros((n, n))
    for i in range(n):
        if row_sums[i] > 0:
            mat_pct[i] = mat[i] / row_sums[i] * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            pct_val = mat_pct[i, j]
            if i == j:
                text = f"{mat[i,j]}\n({pct_val:.1f}%)"
                color = "white" if pct_val > 50 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
            else:
                text = f"{mat[i,j]}"
                color = "white" if pct_val > 50 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(label_names, fontsize=11)
    ax.set_xlabel("Predicted Positive", fontsize=13)
    ax.set_ylabel("True Positive", fontsize=13)
    ax.set_title("Co-occurrence Matrix: True Disease vs Predicted Positive\n"
                 f"[Count + Row%] N={len(t)}  |  Diagonal = Recall  |  "
                 "Off-diagonal reflects comorbidity",
                 fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("% of True Cases", fontsize=11)

    for i in range(n):
        ax.text(n + 0.6, i, f"N={row_sums[i]}", va="center",
                fontsize=9, color="dimgray", style="italic")

    # Footnote
    fig.text(0.5, 0.01,
             "Off-diagonal: patients with disease i also predicted as j (many have both diseases). "
             "See error_decomposition.png for genuine false positives.",
             ha="center", fontsize=7.5, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_dir / "confusion_matrix_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  混淆矩阵(合并)已保存: %s", out_dir / "confusion_matrix_combined.png")


def _plot_error_decomposition(t, pred, label_names, out_dir):
    """Clean 6x6 error matrix for multilabel classification.
    Diagonal (green): correctly identified (TP + Recall%).
    Off-diagonal (i,j): genuine FP — patients with TRUE disease i, who do NOT
      actually have disease j, but were predicted as j. Comorbidity excluded.

    Right side: per-class Precision / Recall / F1 bar chart with FN count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(label_names)
    total_samples = len(t)  # images
    total_label_instances = int(sum(int(t[:, i].sum()) for i in range(n)))

    tp_vec = np.array([int(((t[:, i] == 1) & (pred[:, i] == 1)).sum()) for i in range(n)])
    fn_vec = np.array([int(((t[:, i] == 1) & (pred[:, i] == 0)).sum()) for i in range(n)])
    fp_vec = np.array([int(((t[:, i] == 0) & (pred[:, i] == 1)).sum()) for i in range(n)])
    row_sums = tp_vec + fn_vec

    # Error matrix: true i=1, true j=0, predicted j=1  (genuine FP, comorbidity excluded)
    err_mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                err_mat[i, j] = int(((t[:, i] == 1) & (t[:, j] == 0) & (pred[:, j] == 1)).sum())

    fig = plt.figure(figsize=(18, 7.5))
    gs = GridSpec(1, 26, figure=fig, wspace=0.05)

    # ── LEFT: 6×6 clean matrix (no extra rows) ──
    ax_left = fig.add_subplot(gs[0, :12])
    ax_left.set_facecolor("#f8f8f8")

    vmax = max(err_mat.max(), 1)
    # Use a masked array: diagonal = vmax (green), off-diagonal = err counts
    disp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            disp[i, j] = err_mat[i, j] if i != j else np.nan

    im_left = ax_left.imshow(disp, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    for i in range(n):
        # Diagonal: green patch + TP + recall%
        recall_pct = tp_vec[i] / row_sums[i] * 100 if row_sums[i] > 0 else 0
        ax_left.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                         fill=True, facecolor="#4CAF50", alpha=0.75, zorder=0))
        ax_left.text(i, i, f"TP={tp_vec[i]}\n({recall_pct:.0f}%)",
                     ha="center", va="center", fontsize=10, fontweight="bold", color="white")

        # Off-diagonal error cells
        for j in range(n):
            if i != j and err_mat[i, j] > 0:
                ax_left.text(j, i, str(err_mat[i, j]),
                             ha="center", va="center", fontsize=8.5, fontweight="bold",
                             color="darkred")

    ax_left.set_xticks(range(n))
    ax_left.set_xticklabels(label_names, rotation=25, ha="right", fontsize=9)
    ax_left.set_yticks(range(n))
    ax_left.set_yticklabels(label_names, fontsize=9)
    ax_left.set_xlabel("Predicted Label  (off-diag = FP: predicted j but patient does NOT have j)", fontsize=9)
    ax_left.set_ylabel("True Disease", fontsize=10)
    ax_left.set_title(f"Label-Level Error Decomposition — {total_label_instances} label instances in {total_samples} images\n"
                      "Green diag = Correct (TP+Recall%)  |  Red off-diag = Genuine False Positives (comorbidity excluded)",
                      fontsize=11, fontweight="bold")
    plt.colorbar(im_left, ax=ax_left, shrink=0.78, label="FP count")

    # ── RIGHT: Per-class metrics + FN info ──
    ax_right = fig.add_subplot(gs[0, 14:])

    prec_list, rec_list, f1_list = [], [], []
    for i in range(n):
        dp = tp_vec[i] + fp_vec[i]; dr = tp_vec[i] + fn_vec[i]
        prec = tp_vec[i] / dp if dp > 0 else 0
        rec = tp_vec[i] / dr if dr > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

    x = np.arange(n); w = 0.25
    ax_right.bar(x - w, prec_list, w, label="Precision", color="#2196F3", alpha=0.85)
    ax_right.bar(x, rec_list, w, label="Recall", color="#4CAF50", alpha=0.85)
    ax_right.bar(x + w, f1_list, w, label="F1", color="#FF9800", alpha=0.85)

    for i in range(n):
        ax_right.text(i - w, prec_list[i] + 0.02, f"{prec_list[i]:.3f}",
                      ha="center", fontsize=7, rotation=90)
        ax_right.text(i, rec_list[i] + 0.02, f"{rec_list[i]:.3f}",
                      ha="center", fontsize=7, rotation=90)
        ax_right.text(i + w, f1_list[i] + 0.02, f"{f1_list[i]:.3f}",
                      ha="center", fontsize=7, rotation=90)

    # Add FN numbers below bars as text
    for i in range(n):
        ax_right.text(i, 0.05, f"FN={fn_vec[i]}", ha="center", fontsize=7.5,
                      color="darkred", style="italic", fontweight="bold")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(label_names, rotation=30, ha="right", fontsize=8.5)
    ax_right.set_ylim(0, 1.18)
    ax_right.set_ylabel("Score", fontsize=10)
    ax_right.set_title("Per-Class Precision / Recall / F1  (FN=missed below bars)", fontsize=11, fontweight="bold")
    ax_right.legend(loc="lower right", fontsize=8)
    ax_right.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "error_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  错误分解矩阵已保存: %s", out_dir / "error_decomposition.png")


def plot_roc_curves(t, p, label_names, is_binary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    en_names = _en_labels(label_names)
    fig, ax = plt.subplots(figsize=(9, 7.5))

    if is_binary:
        ti = t.reshape(-1).astype(int); pi = p.reshape(-1)
        if len(np.unique(ti)) > 1:
            fpr, tpr, _ = roc_curve(ti, pi)
            auc_val = roc_auc_score(ti, pi)
            ax.plot(fpr, tpr, lw=2.5, label=f"Disease (AUC={auc_val:.4f})")
    else:
        for i, name in enumerate(en_names):
            ti = t[:, i]; pi = p[:, i]
            if len(np.unique(ti)) > 1:
                fpr, tpr, _ = roc_curve(ti, pi)
                auc_val = roc_auc_score(ti, pi)
                ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc_val:.4f})")
        # macro-average
        from sklearn.metrics import auc as sk_auc
        all_fpr = np.linspace(0, 1, 100)
        tprs = []
        for i in range(len(en_names)):
            ti = t[:, i]; pi = p[:, i]
            if len(np.unique(ti)) > 1:
                fpr, tpr, _ = roc_curve(ti, pi)
                tprs.append(np.interp(all_fpr, fpr, tpr))
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            macro_auc = sk_auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, lw=3, linestyle="--", color="black",
                    label=f"Macro-avg (AUC={macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classes Combined", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  ROC曲线已保存: %s", out_dir / "roc_curves.png")


def plot_calibration_curves(t, p, label_names, is_binary, out_dir):
    """Reliability diagrams — all classes overlaid in one figure + per-class subplots"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    en_names = _en_labels(label_names)
    B = 10

    if is_binary:
        targets = [(t.reshape(-1).astype(int), p.reshape(-1), "Disease")]
    else:
        targets = [(t[:, i], p[:, i], name) for i, name in enumerate(en_names)]

    # ── Per-class subplots ──
    n = len(targets)
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ti, pi, name) in enumerate(targets):
        ax = axes[i]
        bin_edges = np.linspace(0, 1, B + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        accs, confs, counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (pi >= lo) & ((pi < hi) if hi < 1 else (pi <= hi))
            if mask.sum() >= 5:
                accs.append(ti[mask].mean())
                confs.append(pi[mask].mean())
            else:
                accs.append(np.nan)
                confs.append(bin_centers[len(accs) - 1] if accs else 0.5)
            counts.append(mask.sum())

        ax.bar(bin_centers, accs, width=0.08, alpha=0.6, label="Actual", color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Perfect")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (ECE={_ece(ti, pi):.4f})", fontsize=10)
        for j, (bc, acc, cnt) in enumerate(zip(bin_centers, accs, counts)):
            if not np.isnan(acc) and cnt > 0:
                ax.text(bc, acc + 0.03, str(cnt), ha="center", fontsize=7, color="gray")
        ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Calibration Curves (Reliability Diagrams)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Combined: all classes overlaid ──
    if not is_binary:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10.colors
        for i, (ti, pi, name) in enumerate(targets):
            bin_edges = np.linspace(0, 1, B + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            accs = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (pi >= lo) & ((pi < hi) if hi < 1 else (pi <= hi))
                accs.append(ti[mask].mean() if mask.sum() >= 5 else np.nan)
            ax2.plot(bin_centers, accs, "o-", lw=2, markersize=5,
                     color=colors[i % len(colors)], label=f"{name} (ECE={_ece(ti, pi):.4f})")
        ax2.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.3, label="Perfect")
        ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
        ax2.set_xlabel("Confidence"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Calibration Curves — All Classes Combined", fontsize=13)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(out_dir / "calibration_curves_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("  校准曲线(合并)已保存: %s", out_dir / "calibration_curves_combined.png")

    logging.info("  校准曲线已保存: %s", out_dir / "calibration_curves.png")


def plot_confidence_distributions(p, t, label_names, is_binary, out_dir):
    """置信度分布: 正/负样本的预测概率直方图 — 全部合并 + 单独子图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    en_names = _en_labels(label_names)

    if is_binary:
        targets = [(t.reshape(-1).astype(int), p.reshape(-1), "Disease")]
    else:
        targets = [(t[:, i], p[:, i], name) for i, name in enumerate(en_names)]

    # ── Per-class subplots ──
    n = len(targets)
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ti, pi, name) in enumerate(targets):
        ax = axes[i]
        pos_mask = ti == 1; neg_mask = ti == 0
        bins = np.linspace(0, 1, 21)
        ax.hist(pi[pos_mask], bins=bins, alpha=0.6, label=f"Positive (n={pos_mask.sum()})",
                color="coral", edgecolor="darkred", linewidth=0.5)
        ax.hist(pi[neg_mask], bins=bins, alpha=0.6, label=f"Negative (n={neg_mask.sum()})",
                color="lightblue", edgecolor="darkblue", linewidth=0.5)
        ax.axvline(x=0.5, color="black", linestyle="--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
        ax.set_title(f"Confidence Distribution — {name}", fontsize=10)
        ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Confidence Score Distributions (Blue=Negative, Red=Positive)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "confidence_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Combined: overlay all classes ──
    if not is_binary:
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        axes2 = axes2.flatten()
        for i, (ti, pi, name) in enumerate(targets):
            ax = axes2[i]
            pos_mask = ti == 1; neg_mask = ti == 0
            bins = np.linspace(0, 1, 21)
            ax.hist(pi[pos_mask], bins=bins, alpha=0.7, label=f"Positive (n={pos_mask.sum()})",
                    color="coral", edgecolor="darkred", linewidth=0.5)
            ax.hist(pi[neg_mask], bins=bins, alpha=0.7, label=f"Negative (n={neg_mask.sum()})",
                    color="lightblue", edgecolor="darkblue", linewidth=0.5)
            ax.axvline(x=0.5, color="black", linestyle="--", lw=1, alpha=0.5)
            ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
            ax.set_title(name, fontsize=11)
            ax.legend(fontsize=8)

        # Per-class metrics summary in the 6th slot (or extra)
        for j in range(n, len(axes2)):
            axes2[j].set_visible(False)

        fig2.suptitle("Confidence Score Distributions — All Classes", fontsize=14, y=1.01)
        plt.tight_layout()
        fig2.savefig(out_dir / "confidence_dist_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("  置信度分布(合并)已保存: %s", out_dir / "confidence_dist_combined.png")

    logging.info("  置信度分布已保存: %s", out_dir / "confidence_dist.png")


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    is_binary = args.mode == "binary"

    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "outputs" / (
            "05_train_binary" if is_binary else "05_train_multilabel")

    # ── 1. 设置 ──
    setup(args.output_dir)
    seed_all(args.seed)

    # ── 2. 配置dump ──
    config = {
        "mode": args.mode,
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "threshold": args.threshold,
        "roi_mode": args.roi_mode,
        "save_interval": args.save_interval,
        "patience": args.patience,
        "amp": args.amp,
        "mutex_lambda": args.mutex_lambda if not is_binary else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    logging.info("=" * 58)
    logging.info("  Configuration")
    logging.info("=" * 58)
    for k, v in config.items():
        logging.info("  %-20s %s", k, v)
    logging.info("=" * 58)

    # ── 3. 数据 ──
    df = load_metadata(args)
    train_df, val_df = split_patients(df, args.val_size, args.seed)
    logging.info("Split: train=%d (patients=%d) | val=%d (patients=%d)",
                 len(train_df), train_df["patient_id"].nunique(),
                 len(val_df), val_df["patient_id"].nunique())

    nc = 1 if is_binary else 6
    label_names = ["Normal", "Disease"] if is_binary else list(LABEL_COLS)

    if args.dry_run:
        logging.info("DRY RUN — 标签分布:")
        if is_binary:
            logging.info("  %s", train_df["disease_present"].value_counts().to_dict())
        else:
            logging.info("  %s", train_df[LABEL_COLS].sum().to_dict())
        return

    # ── 4. Device ──
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", dev)

    # ── 5. DataLoaders ──
    tf_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.12, 0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_tr = SpineDataset(train_df, tf_train, args.roi_mode, is_binary, label_cols=LABEL_COLS)
    ds_vl = SpineDataset(val_df, tf_val, "crop", is_binary, label_cols=LABEL_COLS)
    dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=dev.type == "cuda")
    dl_vl = DataLoader(ds_vl, args.batch_size, shuffle=False, num_workers=args.num_workers,
                       pin_memory=dev.type == "cuda")

    # batch日志间隔 (auto: ~10条/epoch)
    log_interval = args.log_batch_interval
    if log_interval <= 0:
        log_interval = max(1, len(dl_tr) // 10)

    logging.info("Train batches: %d | Val batches: %d | Log interval: %d",
                 len(dl_tr), len(dl_vl), log_interval)

    # ── 6. 模型 & 损失 ──
    model = make_model(args.model, nc, args.pretrained).to(dev)

    if is_binary:
        pos = max((train_df["disease_present"] == 1).sum(), 1.0)
        neg = max((train_df["disease_present"] == 0).sum(), 1.0)
        pw = torch.tensor([neg / pos], device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        score_name = "auc"
        is_better = lambda new, best: not np.isnan(new) and new > best
    else:
        pos = train_df[LABEL_COLS].sum(axis=0).values.astype(np.float32)
        neg = len(train_df) - pos
        pw = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=dev)
        criterion = MultilabelLoss(pos_weight=pw, mutex_lambda=args.mutex_lambda)
        score_name = "f1_macro"
        is_better = lambda new, best: new > best

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")

    best_score = -1.0
    history = []
    t0 = time.time()
    start_epoch = 1
    best_epoch = 0
    patience_counter = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model_state"])
        if "epoch" in ck:
            start_epoch = ck["epoch"] + 1
        if "best_score" in ck:
            best_score = ck["best_score"]
        if "best_epoch" in ck:
            best_epoch = ck["best_epoch"]
        if "optimizer_state" in ck:
            opt.load_state_dict(ck["optimizer_state"])
        if "history" in ck:
            history = ck["history"]
        logging.info("Resumed from %s, start_epoch=%d, best_%s=%.6f",
                     args.resume, start_epoch, score_name, best_score)

    # ── 7. 训练循环 ──
    logging.info("=" * 58)
    logging.info("  Start training — %s | %d epochs | %d train samples",
                 "Binary (disease present)" if is_binary else "Multilabel (6 classes)",
                 args.epochs, len(train_df))
    logging.info("=" * 58)

    for ep in range(start_epoch, args.epochs + 1):
        current_lr = opt.param_groups[0]["lr"]
        ep_t0 = time.time()

        # ── Train ──
        logging.info("[Train]: Epoch %d started (lr=%.8f)", ep, current_lr)
        tl, _, _, tcomps = run_epoch(model, dl_tr, criterion, opt, scaler, dev,
                                     is_train=True, epoch_num=ep,
                                     log_batch_interval=log_interval,
                                     is_binary=is_binary, amp_enabled=args.amp)
        logging.info("[Train]: Epoch %d finished with lr=%.8f", ep, current_lr)

        # ── Val ──
        logging.info("-" * 55)
        logging.info("[Val]: Epoch %d started", ep)
        vl, vt, vl_, vcomps = run_epoch(model, dl_vl, criterion, opt, scaler, dev,
                                        is_train=False, epoch_num=ep,
                                        log_batch_interval=0,
                                        is_binary=is_binary, amp_enabled=args.amp)
        logging.info("[Val]: Epoch %d finished", ep)

        sch.step()

        # ── 指标 ──
        if is_binary:
            m, p, pr = binary_metrics(vt, vl_, args.threshold)
            score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        else:
            m, p, pr = multi_metrics(vt, vl_, args.threshold)
            score = m["f1_macro"]

        ep_time = time.time() - ep_t0

        # ── 历史记录 ──
        record = {"epoch": ep, "train_loss": round(tl, 6), "val_loss": round(vl, 6),
                  "lr": round(current_lr, 8), "time_s": round(ep_time, 1)}
        for k, v in m.items():
            if isinstance(v, (int, float, bool)):
                record[k] = round(v, 6) if isinstance(v, float) else v
        history.append(record)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

        # ── Epoch 汇总日志 ──
        if is_binary:
            logging.info("Epoch %03d/%d | t_loss=%.4f v_loss=%.4f | acc=%.4f f1=%.4f auc=%.4f ece=%.4f | lr=%.2e | %s",
                         ep, args.epochs, tl, vl, m["acc"], m["f1"], m["auc"], m["ece"],
                         current_lr, format_time(ep_time))
        else:
            logging.info("Epoch %03d/%d | t_loss=%.4f v_loss=%.4f | f1_m=%.4f f1_μ=%.4f auc=%.4f ex=%.4f ece=%.4f | lr=%.2e | %s",
                         ep, args.epochs, tl, vl,
                         m["f1_macro"], m["f1_micro"], m["auc_macro"],
                         m["exact_match"], m["ece"],
                         current_lr, format_time(ep_time))
            # 逐标签F1
            per_f1 = " | ".join(
                [f"{LABEL_COLS[i]}:{f1_score(vt[:, i], pr[:, i], zero_division=0):.3f}"
                 for i in range(6)])
            logging.info("         per-label F1: %s", per_f1)

        flush_log()

        # ── 保存最佳模型 ──
        if is_better(score, best_score):
            best_score = score
            best_epoch = ep
            ckpt = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "epoch": ep,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "labels": label_names,
                "val_metrics": m,
                "optimizer_state": opt.state_dict(),
                "history": history,
                "config": config,
            }
            torch.save(ckpt, args.output_dir / "best_model.pt")
            logging.info("  >>> best_model.pt saved (epoch %d, %s=%.6f)", ep, score_name, best_score)
            patience_counter = 0
        else:
            patience_counter += 1

        # ── 定期保存 checkpoint ──
        if ep % args.save_interval == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "epoch": ep,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "labels": label_names,
                "val_metrics": m,
                "optimizer_state": opt.state_dict(),
                "history": history,
                "config": config,
            }
            torch.save(ckpt, args.output_dir / f"checkpoint_epoch{ep:03d}.pt")
            logging.info("  >>> checkpoint_epoch%03d.pt saved", ep)

        # ── Early stopping ──
        if patience_counter >= args.patience:
            logging.info("Early stopping triggered! val_loss hasn't improved for %d epochs.", args.patience)
            logging.info("Best epoch: %d | Best %s: %.6f", best_epoch, score_name, best_score)
            break

        # ── 过拟合检测 ──
        if vl > tl * 1.8 and ep > 20:
            logging.info("Overfitting detected (v_loss %.4f > 1.8 × t_loss %.4f), consider early stop.", vl, tl)

    # ═══════════════════════════════════════════════════════════════
    #  8. 最终评估 (使用最佳模型)
    # ═══════════════════════════════════════════════════════════════
    logging.info("")
    logging.info("=" * 58)
    logging.info("  Training complete. Final evaluation with best model (epoch %d)...", best_epoch)
    logging.info("=" * 58)

    # 加载最佳模型
    best_ckpt = torch.load(args.output_dir / "best_model.pt", map_location=dev)
    model.load_state_dict(best_ckpt["model_state"])

    vl, vt, vl_, vcomps = run_epoch(model, dl_vl, criterion, opt, scaler, dev,
                                    is_train=False, epoch_num=0,
                                    log_batch_interval=0,
                                    is_binary=is_binary, amp_enabled=args.amp)

    if is_binary:
        m, p, pr = binary_metrics(vt, vl_, args.threshold)
        report = classification_report(vt.reshape(-1).astype(int), pr.astype(int),
                                       target_names=["Normal", "Disease"], zero_division=0)
    else:
        m, p, pr = multi_metrics(vt, vl_, args.threshold)
        report = classification_report(vt, pr, target_names=list(LABEL_COLS),
                                       zero_division=0)

    # 保存分类报告
    (args.output_dir / "val_report.txt").write_text(report, encoding="utf-8")
    logging.info("Validation report:\n%s", report)

    # 逐标签报告
    per_label_report(vt, pr, p, label_names[1:] if is_binary else label_names, args.output_dir)

    # 所有可视化
    logging.info("")
    logging.info("Generating evaluation visualizations...")
    try:
        plot_all_visualizations(vt, p, pr, label_names, is_binary, args.output_dir)
    except Exception as e:
        logging.warning("Visualization error: %s", e)

    # ── 9. 保存最终模型 ──
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "labels": label_names,
        "config": config,
    }, args.output_dir / "final_model.pt")
    logging.info("final_model.pt saved.")

    # ── 10. 部署目录 ──
    deploy_dir = args.output_dir / "deploy"
    deploy_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "labels": label_names,
        "is_binary": is_binary,
        "image_size": args.image_size,
        "config": config,
    }, deploy_dir / "best_model.pt")

    deploy_config = {
        "model": args.model,
        "mode": "binary" if is_binary else "multilabel",
        "labels": label_names,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "roi_mode": args.roi_mode,
        "input_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "val_metrics": {k: v for k, v in m.items() if isinstance(v, (int, float, bool))},
        "n_classes": nc,
    }
    json.dump(deploy_config, (deploy_dir / "config.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    logging.info("Deploy package saved to: %s", deploy_dir)

    # ── 总结 ──
    total_time = time.time() - t0
    logging.info("")
    logging.info("=" * 58)
    logging.info("  Training Summary")
    logging.info("=" * 58)
    logging.info("  Mode:          %s", "Binary" if is_binary else "Multilabel")
    logging.info("  Total epochs:  %d (best=%d)", ep, best_epoch)
    logging.info("  Total time:    %s", format_time(total_time))
    logging.info("  Best %s:    %.6f", score_name, best_score)
    if is_binary:
        logging.info("  Best acc=%.4f  f1=%.4f  auc=%.4f  ece=%.4f",
                     m["acc"], m["f1"], m["auc"], m["ece"])
    else:
        logging.info("  Best f1_m=%.4f  f1_μ=%.4f  auc=%.4f  ece=%.4f",
                     m["f1_macro"], m["f1_micro"], m["auc_macro"], m["ece"])
    logging.info("  Output:        %s", args.output_dir)
    logging.info("=" * 58)


if __name__ == "__main__":
    main()
