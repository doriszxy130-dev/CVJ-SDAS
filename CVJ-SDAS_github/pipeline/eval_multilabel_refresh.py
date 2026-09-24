"""Re-run multilabel evaluation with English labels + combined plots."""
import sys, logging, time, numpy as np, torch, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import *

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs" / "05_train_multilabel"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(OUT_DIR / "train.log", mode="a", encoding="utf-8")])

# Load best model (epoch 73, f1_macro=0.8926)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=dev)
config = ckpt.get("config", {})
is_binary = False
en_labels = ["Anterior AAD", "Posterior AAD", "Basilar Invagination (BI)",
             "Os Odontoideum (OO)", "Occipitalization of Atlas", "C2-3 Non-segmentation"]

logging.info("Re-running multilabel evaluation with English labels (best epoch %d)", ckpt["epoch"])

# Re-load data
args = type('Args', (), {})()
args.mode = "multilabel"
args.retro_csv = SCRIPT_DIR/"outputs"/"04_unified_metadata"/"retrospective_7830_bbox_labels.csv"
args.public_csv = SCRIPT_DIR/"outputs"/"04_unified_metadata"/"public_normal_3054_bbox_labels.csv"
args.val_size = 0.15; args.seed = 42; args.batch_size = 32; args.num_workers = 0
args.image_size = 224; args.roi_mode = "both"; args.amp = True
args.mutex_lambda = 0.5; args.threshold = 0.5; args.epochs = 100
args.lr = 1e-4; args.weight_decay = 1e-4

df = load_metadata(args)
train_df, val_df = split_patients(df, args.val_size, args.seed)

tf_val = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ds_vl = SpineDataset(val_df, tf_val, "crop", is_binary, label_cols=LABEL_COLS)
dl_vl = DataLoader(ds_vl, 32, shuffle=False, num_workers=0, pin_memory=dev.type=="cuda")

model = make_model("resnet50", 6, False).to(dev)
model.load_state_dict(ckpt["model_state"])

pos = train_df[LABEL_COLS].sum(axis=0).values.astype(np.float32)
neg = len(train_df) - pos
pw = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=dev)
crit = MultilabelLoss(pos_weight=pw, mutex_lambda=0.5)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=dev.type=="cuda")

vl, vt, vl_, vcomps = run_epoch(model, dl_vl, crit, opt, scaler, dev,
                                is_train=False, epoch_num=0, log_batch_interval=0,
                                is_binary=False, amp_enabled=True)
m, p, pr = multi_metrics(vt, vl_, 0.5)

# Re-generate all visualizations with English labels
logging.info("Generating evaluation visualizations with English labels...")
try:
    plot_all_visualizations(vt, p, pr, en_labels, is_binary, OUT_DIR)
except Exception as e:
    logging.exception("Visualization error: %s", e)

# Re-generate per-label report with English labels
per_label_report(vt, pr, p, en_labels, OUT_DIR)

# Update deploy config with English labels
deploy_dir = OUT_DIR / "deploy"
dconfig = json.loads((deploy_dir / "config.json").read_text(encoding="utf-8"))
dconfig["labels_en"] = en_labels
json.dump(dconfig, (deploy_dir / "config.json").open("w", encoding="utf-8"),
         indent=2, ensure_ascii=False)

logging.info("Multilabel evaluation refresh complete.")
