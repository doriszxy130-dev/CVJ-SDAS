# Atlantoaxial Disease Classification System

> 寰枢椎疾病自动分类系统 — YOLOv8 椎体检测 → ResNet50 疾病分类

## 1. 项目概述 / Overview

本系统用于颈椎 X 线片的自动化分析：先检测脊柱 ROI 区域，再判断是否存在寰枢椎相关疾病，并对 6 类疾病进行多标签分类。

**训练数据**: 7,778 张回顾性图像 (2,802 患者) + 3,054 张公开正常图像  
**外部验证**: 643 张 (4 家多中心医院)  
**前瞻测试**: 267 张 (1 家新医院)

---

## 2. 模型架构 / Architecture

```
输入 X 线片
    │
    ▼
┌─────────────────────────────────┐
│ ① YOLOv8s                       │  目标检测 → 脊柱 ROI bbox
│    bbox 可手动修正              │
└─────────────────────────────────┘
    │ crop + 10% padding + Resize(224×224)
    ▼
      ┌──────────────────────────┐
      │ ② Binary ResNet50 (1 out)│  → 是否有病? + 置信度
      └──────────────────────────┘
      ┌──────────────────────────┐
      │ ③ Multilabel ResNet50    │  → 6 类疾病概率分布
      │    (6 outputs)           │
      └──────────────────────────┘
    │
    ▼
输出: { disease_present: bool, diseases: [{name, confidence}, ...] }
```

### 模型配置

| 组件 | 模型 | 输入尺寸 | 输出 | 权重文件 |
|---|---|---|---|---|
| BBox 检测 | YOLOv8s | 640×640 | 1 bbox + conf | `yolo_best.pt` (22 MB) |
| 二分类 | ResNet50 | 224×224 | 1 logit | `binary_best.pt` (90 MB) |
| 多标签 | ResNet50 | 224×224 | 6 logits | `multilabel_best.pt` (91 MB) |

---

## 3. 疾病类别 / Disease Classes

| # | 英文 | 中文 | 缩写 |
|---|---|---|---|
| 1 | Anterior Atlantoaxial Dislocation | 寰椎前脱位 | Anterior AAD |
| 2 | Posterior Atlantoaxial Dislocation | 寰椎后脱位 | Posterior AAD |
| 3 | Basilar Invagination | 颅底凹陷 | BI |
| 4 | Os Odontoideum | 齿突不连 | OO |
| 5 | Occipitalization of Atlas | 寰椎枕化 | — |
| 6 | C2-3 Non-segmentation | 颈 2-3 分节不全 | C2-3 NS |

> ⚠️ **互斥约束**: 寰椎前脱位 与 寰椎后脱位 不能同时为阳性（训练时加入 mutex penalty λ=0.5）

---

## 4. 模型性能 / Performance

### 内部验证集 (Internal Validation)
1,153 张图 / 2,530 个标签实例 (来自同一医院，患者级划分)

#### Multilabel (6 类)

| 疾病 | F1 | AUC | 支持数 |
|---|---|---|---|
| Anterior AAD | 0.9485 | 0.9545 | 876 |
| Posterior AAD | 0.8528 | 0.9845 | 129 |
| Basilar Invagination | 0.8453 | 0.9499 | 344 |
| Os Odontoideum | 0.8911 | 0.9548 | 511 |
| Occipitalization of Atlas | 0.9119 | 0.9659 | 414 |
| C2-3 Non-segmentation | 0.9059 | 0.9792 | 256 |
| **Macro 平均** | **0.8926** | **0.9648** | — |

| 指标 | 值 |
|---|---|
| F1_macro | 0.8926 |
| AUC_macro | 0.9648 |
| Exact Match (全对) | 70.60% |
| ECE (校准误差) | 0.0544 |

#### Binary (有病/无病)

| 指标 | 值 |
|---|---|
| AUC | **0.9938** |
| Accuracy | 0.9684 |
| F1 | 0.9779 |
| ECE | 0.0210 |

### 前瞻测试集 (Prospective, N=266)
另一家新医院，完全不同数据域

| Multilabel | F1_macro | AUC_macro | Exact |
|---|---|---|---|
| 内部验证 | 0.8926 | 0.9648 | 70.6% |
| **前瞻测试** | **0.8868** | **0.9612** | **65.4%** |
| Δ | −0.006 | −0.004 | −5.2% |

| Binary | AUC | Acc |
|---|---|---|
| 内部验证 | 0.9938 | 0.9684 |
| **前瞻测试** | **0.9119** | **0.9362** |

### 外部多中心验证 (External Validation, N=643)
4 家独立医院

| Multilabel | F1_macro | AUC_macro | Exact |
|---|---|---|---|
| 内部验证 | 0.8926 | 0.9648 | 70.6% |
| **外部验证** | **0.8091** | **0.9329** | **55.8%** |
| Δ | −0.083 | −0.032 | −14.8% |

| 分医院 | 图数 | ML F1_macro | ML Exact | Binary AUC |
|---|---|---|---|---|
| 浙大二院 | 380 | 0.8258 | 56.3% | 0.8735 |
| 盛京医院 | 57 | 0.8699 | 77.2% | 0.9340 |
| 河北三院 | 71 | 0.7460 | 40.9% | 0.7575 |
| 吉大三院 | 135 | 0.7169 | 53.3% | 0.6953 |

---

## 5. 部署使用 / Deployment

### 环境依赖

```
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
ultralytics >= 8.0 (用于 YOLOv8)
pillow
numpy
```

```bash
pip install torch torchvision ultralytics pillow numpy
```

### 快速开始

```bash
# 单张图推理
python deploy/inference.py --image chest_xray.jpg

# 指定 bbox (跳过 YOLO)
python deploy/inference.py --image chest_xray.jpg --bbox "100,200,300,400"

# 批量推理
python deploy/inference.py --dir /path/to/images/ --output results.csv

# GPU
python deploy/inference.py --image xray.jpg --device cuda
```

### 输出格式

```json
{
  "success": true,
  "image_size": [628, 707],
  "bbox": {
    "xmin": 241.2, "ymin": 162.5, "xmax": 391.6, "ymax": 277.9,
    "confidence": 0.871, "source": "yolo_predicted"
  },
  "disease_present": true,
  "disease_present_confidence": 0.9999,
  "diseases_found": [
    {"name_en": "Anterior AAD", "name_cn": "寰椎前脱位", "confidence": 0.9995},
    {"name_en": "Basilar Invagination (BI)", "name_cn": "颅底凹陷", "confidence": 0.9887}
  ],
  "timing_ms": {"yolo_ms": 487, "binary_ms": 53, "multilabel_ms": 11, "total_ms": 561}
}
```

### 平台集成要点

1. **用户可修正 bbox**: 调用 `engine.predict(image, bbox=(xmin,ymin,xmax,ymax))` 跳过了 YOLO 检测
2. **结果展示**: `disease_present` 判断有无病，`diseases_found` 列出所有发现的疾病及置信度
3. **若无病**: `diseases_found=[]`, `disease_present=false`, 显示 `disease_present_confidence`
4. **若有病**: `diseases_found` 按置信度从高到低排列

### Python API

```python
from deploy.inference import SpineInference

engine = SpineInference("deploy/", device="cuda")

# 自动 YOLO 检测
result = engine.predict("xray.jpg")

# 手动 bbox 覆盖
result = engine.predict("xray.jpg", bbox=(100, 150, 300, 350))

print(result["disease_present"])
for d in result["diseases_found"]:
    print(f"  {d['name_cn']}: {d['confidence']:.4f}")
```

---

## 6. 目录结构 / Project Structure

```
Atlantoaxial_Disease_Classifier/
├── deploy/
│   ├── inference.py          # 推理脚本 (独立, 无项目依赖)
│   ├── config.json            # 模型配置 & 性能指标
│   ├── yolo_best.pt           # YOLOv8s 脊柱检测 (22 MB)
│   ├── binary_best.pt         # ResNet50 二分类 (90 MB)
│   ├── multilabel_best.pt     # ResNet50 多标签 (91 MB)
│   └── README.md              # 本文档
└── results/
    ├── internal_val/          # 内部验证集结果
    │   ├── multilabel/        # 6×6 混淆矩阵, ROC, 校准曲线等
    │   └── binary/            # 2×2 混淆矩阵, ROC 等
    ├── prospective_test/      # 前瞻测试集结果 (266张)
    │   ├── summary.json
    │   ├── multilabel/
    │   └── binary/
    └── external_val/          # 外部多中心验证 (643张, 4医院)
        ├── summary.json
        ├── multilabel/
        └── binary/
```

---

## 7. 训练配置 / Training Details

| 参数 | 值 |
|---|---|
| Backbone | ResNet50 (ImageNet 预训练) |
| 优化器 | AdamW (lr=1e-4, wd=1e-4) |
| 调度器 | CosineAnnealingLR (T_max=100) |
| 损失函数 (Binary) | BCEWithLogitsLoss |
| 损失函数 (Multilabel) | BCEWithLogitsLoss + pos_weight + mutex_penalty (λ=0.5) |
| 图像尺寸 | 224×224 (crop 后) |
| ROI 模式 | 50% crop / 50% 原图 |
| Batch Size | 32 |
| AMP | ✅ |
| 早停 | patience=15 |
| 数据划分 | GroupShuffleSplit (按患者, 15% 验证) |

---

## 8. 结果图说明 / Plot Guide

| 文件 | 级别 | 含义 |
|---|---|---|
| `confusion_matrix.png` | 标签级 | 6×6 矩阵: 真实病种→预测病种计数 |
| `error_decomposition.png` | 标签级 | 真实假阳性矩阵 (排除共病重叠) + P/R/F1 柱状图 |
| `diagnostic_image_level.png` | 图像级 | 3类评估 (全对/漏诊/误诊) + 正确标签数分布 |
| `roc_curves.png` | 标签级 | 6条 ROC + macro-avg |
| `calibration_curves.png` | 标签级 | 可靠性图 (校准度) |
| `confidence_dist.png` | 标签级 | 正/负样本的置信度分布 |

---

## 9. 版本历史 / Changelog

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-06-17 | 初始发布: 内部验证 + 前瞻测试 + 多中心外部验证 |

---

*如有问题请联系项目负责人。*
