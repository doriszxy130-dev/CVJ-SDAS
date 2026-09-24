"""Re-run binary final evaluation from saved best_model.pt"""
import sys, logging, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import *

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs" / "05_train_binary"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(OUT_DIR / "train.log", mode="a", encoding="utf-8")])

# Load best model
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=dev)
config = ckpt.get("config", {})
label_names = ckpt["labels"]
is_binary = True

logging.info("Re-running binary final evaluation using best_model.pt (epoch %d)", ckpt["epoch"])

# Re-load data
args = type('Args', (), {})()
args.mode = "binary"
args.retro_csv = SCRIPT_DIR/"outputs"/"04_unified_metadata"/"retrospective_7830_bbox_labels.csv"
args.public_csv = SCRIPT_DIR/"outputs"/"04_unified_metadata"/"public_normal_3054_bbox_labels.csv"
args.val_size = 0.15
args.seed = 42
args.batch_size = 32
args.num_workers = 0
args.image_size = 224
args.roi_mode = "both"
args.lr = 1e-4
args.weight_decay = 1e-4
args.amp = True
args.mutex_lambda = 0
args.threshold = 0.5
args.epochs = 100

df = load_metadata(args)
train_df, val_df = split_patients(df, args.val_size, args.seed)

tf_val = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ds_vl = SpineDataset(val_df, tf_val, "crop", is_binary, label_cols=LABEL_COLS)
dl_vl = DataLoader(ds_vl, args.batch_size, shuffle=False, num_workers=0, pin_memory=dev.type=="cuda")

model = make_model("resnet50", 1, False).to(dev)
model.load_state_dict(ckpt["model_state"])

pos = 1.0; neg = 1.0
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], device=dev))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=dev.type=="cuda")

vl, vt, vl_, vcomps = run_epoch(model, dl_vl, crit, opt, scaler, dev,
                                is_train=False, epoch_num=0, log_batch_interval=0,
                                is_binary=True, amp_enabled=True)
m, p, pr = binary_metrics(vt, vl_, 0.5)
report = classification_report(vt.reshape(-1).astype(int), pr.astype(int),
                               target_names=["Normal", "Disease"], zero_division=0)
logging.info("Validation report:\n%s", report)

per_label_report(vt, pr, p, label_names[1:], OUT_DIR)

logging.info("Generating visualizations...")
try:
    plot_all_visualizations(vt, p, pr, label_names, is_binary, OUT_DIR)
except Exception as e:
    logging.warning("Visualization error: %s", e)

# save final_model.pt
torch.save({"model_state": model.state_dict(), "model_name": "resnet50",
            "labels": label_names, "config": config}, OUT_DIR / "final_model.pt")

# deploy
deploy_dir = OUT_DIR / "deploy"
deploy_dir.mkdir(exist_ok=True)
torch.save({"model_state": model.state_dict(), "model_name": "resnet50",
            "labels": label_names, "is_binary": True, "image_size": 224,
            "config": config}, deploy_dir / "best_model.pt")
dconfig = {
    "model": "resnet50", "mode": "binary",
    "labels": label_names, "image_size": 224, "threshold": 0.5,
    "input_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "val_metrics": {k: v for k, v in m.items() if isinstance(v, (int, float, bool))},
    "n_classes": 1,
}
import json
json.dump(dconfig, (deploy_dir / "config.json").open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

logging.info("Binary final evaluation complete.")
logging.info("Best auc=%.4f f1=%.4f acc=%.4f ece=%.4f", m["auc"], m["f1"], m["acc"], m["ece"])
