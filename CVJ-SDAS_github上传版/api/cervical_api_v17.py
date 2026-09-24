#!/usr/bin/env python3
"""
寰枢椎诊断API v17.0 — 真实模型推理版
  YOLOv8s ROI检测 → ResNet50 Binary → ResNet50 Multilabel (6类)

接口:
  1. POST /cervical_upload     — 上传图片，返回 image_id + 检测框 + 疾病判断
  2. POST /cervical_rediagnose  — 传新框坐标 + image_id，重新推理
  3. POST /get_detailed_diagnosis — 传 image_id，返回6类详细诊断
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
import time
import os
import sys
import io
import cgi
import traceback
import tempfile
import base64
import shutil
from datetime import datetime
from urllib.parse import urlsplit
from uuid import uuid4
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# ── Force real-time output ──
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Default paths are relative to this script's own location, so the
#    server works out-of-the-box on any machine. Override via env vars
#    (MODEL_DIR / UPLOAD_DIR / STORE_FILE) for a production deployment. ──
BASE_DIR = Path(__file__).resolve().parent

PORT = int(os.getenv('PORT', '8090'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', str(BASE_DIR / 'models')))

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DISEASE_NAMES_EN = [
    "Anterior AAD", "Posterior AAD", "Basilar Invagination",
    "Os Odontoideum", "Occipitalization of Atlas", "C2-3 Non-segmentation",
]
DISEASE_NAMES_CN = [
    "寰椎前脱位", "寰椎后脱位", "颅底凹陷",
    "齿突不连", "寰椎枕化", "颈2-3分节不全",
]

BINARY_LABELS = ["No common CVJ deformity", "Disease"]
NEGATIVE_LABEL = "未检测到脱位、颅底凹陷、齿突小骨、寰椎枕化或分节不全畸形"

# ═══════════════════════════════════════════════════════════════
#  Model loader (lazy, loaded once on first request)
# ═══════════════════════════════════════════════════════════════
_model_cache = {}

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def build_resnet50(n_classes):
    from torchvision import models
    try:
        w = models.ResNet50_Weights.DEFAULT
    except:
        w = None
    m = models.resnet50(weights=w)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    return m

def load_models():
    """Load all three models. Cached after first call."""
    if _model_cache:
        return _model_cache

    device = get_device()
    print(f"[INIT] Device: {device}", flush=True)

    # YOLOv8
    yolo_path = MODEL_DIR / 'yolo_best.pt'
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    print(f"[INIT] YOLO loaded: {yolo_path}", flush=True)

    # Binary ResNet50
    binary_path = MODEL_DIR / 'binary_best.pt'
    ckpt_b = torch.load(binary_path, map_location=device, weights_only=False)
    binary_model = build_resnet50(1).to(device)
    binary_model.load_state_dict(ckpt_b['model_state'])
    binary_model.eval()
    print(f"[INIT] Binary model loaded (epoch {ckpt_b.get('epoch',-1)})", flush=True)

    # Multilabel ResNet50
    multilabel_path = MODEL_DIR / 'multilabel_best.pt'
    ckpt_m = torch.load(multilabel_path, map_location=device, weights_only=False)
    multilabel_model = build_resnet50(6).to(device)
    multilabel_model.load_state_dict(ckpt_m['model_state'])
    multilabel_model.eval()
    print(f"[INIT] Multilabel model loaded (epoch {ckpt_m.get('epoch',-1)})", flush=True)

    # Transform
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    _model_cache.update({
        'yolo': yolo,
        'binary': binary_model,
        'multilabel': multilabel_model,
        'transform': tf,
        'device': device,
    })
    return _model_cache


# ═══════════════════════════════════════════════════════════════
#  Inference functions
# ═══════════════════════════════════════════════════════════════

def detect_roi(yolo, image_path):
    """Run YOLOv8. Returns (xmin,ymin,xmax,ymax,confidence) or None."""
    results = yolo(str(image_path), conf=0.25, imgsz=640, device=get_device(), verbose=False)
    if isinstance(results, list):
        results = results[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    xmin, ymin, xmax, ymax = boxes[best_idx].tolist()
    confidence = float(confs[best_idx])
    return (xmin, ymin, xmax, ymax, confidence)


def bbox_xyxy_to_x_y_w_h(xmin, ymin, xmax, ymax):
    """Convert xyxy format to {x, y, width, height} for API output."""
    return {
        "x": float(xmin),
        "y": float(ymin),
        "width": float(xmax - xmin),
        "height": float(ymax - ymin),
    }


def bbox_x_y_w_h_to_xyxy(box_dict, img_w, img_h):
    """Convert API {x, y, width, height} to (xmin, ymin, xmax, ymax) clamped."""
    x = float(box_dict.get('x', 0))
    y = float(box_dict.get('y', 0))
    w = float(box_dict.get('width', 100))
    h = float(box_dict.get('height', 100))
    xmin = max(0, x)
    ymin = max(0, y)
    xmax = min(img_w, x + w)
    ymax = min(img_h, y + h)
    return (xmin, ymin, xmax, ymax)


def crop_with_padding(img, xmin, ymin, xmax, ymax):
    """Crop with 10% padding, matching training."""
    w, h = img.size
    bw = xmax - xmin; bh = ymax - ymin
    left = max(0, int(xmin - bw * 0.1))
    top = max(0, int(ymin - bh * 0.1))
    right = min(w, int(xmax + bw * 0.1))
    bottom = min(h, int(ymax + bh * 0.1))
    if right > left and bottom > top:
        return img.crop((left, top, right, bottom))
    return img


def run_inference(image_path, bbox_override=None):
    """
    Full inference pipeline.
    Returns dict with: success, has_disease, detection_box, diseases, timing_ms, error
    """
    models = load_models()
    yolo = models['yolo']
    binary = models['binary']
    multilabel = models['multilabel']
    transform = models['transform']
    device = models['device']

    t_start = time.time()
    timing = {}

    # Load image
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img_w, img_h = img.size

    # Step 1: BBox
    t0 = time.time()
    bbox_source = 'full_image_fallback'
    bbox_conf = 0.0
    if bbox_override is not None:
        bbox = bbox_x_y_w_h_to_xyxy(bbox_override, img_w, img_h)
        bbox_source = 'manual'
        bbox_conf = 1.0
    else:
        detected = detect_roi(yolo, image_path)
        if detected is not None:
            xmin, ymin, xmax, ymax, bbox_conf = detected
            bbox = (xmin, ymin, xmax, ymax)
            bbox_source = 'yolo_predicted'
        else:
            # Fallback: YOLO failed → use central region of image
            bw, bh = img_w / 3, img_h / 3
            bbox = (img_w/2 - bw/2, img_h/2 - bh/2, img_w/2 + bw/2, img_h/2 + bh/2)
            bbox_source = 'full_image_fallback'
            bbox_conf = 0.0
    timing['yolo_ms'] = (time.time() - t0) * 1000

    # Step 2: Crop + classify
    img_cropped = crop_with_padding(img, *bbox)
    img_tensor = transform(img_cropped).unsqueeze(0).to(device)

    # Binary
    t0 = time.time()
    with torch.no_grad():
        binary_logit = binary(img_tensor)
    binary_prob = float(torch.sigmoid(binary_logit).cpu().item())
    has_disease = binary_prob >= 0.5
    timing['binary_ms'] = (time.time() - t0) * 1000

    # Multilabel
    t0 = time.time()
    with torch.no_grad():
        ml_logits = multilabel(img_tensor)
    ml_probs = torch.sigmoid(ml_logits).cpu().numpy()[0]
    timing['multilabel_ms'] = (time.time() - t0) * 1000

    # Build disease list
    diseases = []
    for i in range(6):
        prob = float(ml_probs[i])
        present = prob >= 0.5
        diseases.append({
            "disease_id": i + 1,
            "disease_name": DISEASE_NAMES_CN[i],
            "disease_name_en": DISEASE_NAMES_EN[i],
            "confidence": round(prob, 4),
            "is_detected": present,
            "threshold": 0.5,
        })

    # Disease gate: ML primary (proven more robust on domain-shifted data)
    # ML any > 0.5 → diseased. If ML says nothing, trust ML — binary can be overconfident.
    ml_any_disease = any(d['is_detected'] for d in diseases)

    # BBox in API format
    detection_box = bbox_xyxy_to_x_y_w_h(*bbox)

    timing['total_ms'] = (time.time() - t_start) * 1000

    return {
        'success': True,
        'image_size': (img_w, img_h),
        'has_disease': ml_any_disease,             # ML primary gate (more robust)
        'binary_confidence': round(binary_prob, 4),
        'detection_box': detection_box,
        'detection_box_source': bbox_source,
        'detection_confidence': round(bbox_conf, 4),
        'diseases': diseases,
        'detected_diseases': [d for d in diseases if d['is_detected']],
        'timing_ms': timing,
    }


# ═══════════════════════════════════════════════════════════════
#  Storage (persistent — survives restarts)
# ═══════════════════════════════════════════════════════════════
image_store = {}
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', str(BASE_DIR / 'data' / 'uploads')))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORE_FILE = Path(os.getenv('STORE_FILE', str(BASE_DIR / 'data' / 'image_store.json')))

def _load_store():
    """Load image_store from disk on startup."""
    if STORE_FILE.exists():
        try:
            with open(STORE_FILE, 'r') as f:
                data = json.load(f)
            image_store.update(data)
            print(f'[STORAGE] Loaded {len(data)} records from {STORE_FILE}')
        except:
            print('[STORAGE] Failed to load store, starting fresh')

def _save_store():
    """Persist image_store to disk."""
    try:
        safe = {}
        for k, v in image_store.items():
            safe[k] = {
                'has_disease': v.get('has_disease'),
                'binary_confidence': v.get('binary_confidence'),
                'detection_box': v.get('detection_box'),
                'diseases': v.get('diseases'),
                'image_path': v.get('image_path'),
                'upload_time': v.get('upload_time'),
                'timing_ms': v.get('timing_ms'),
            }
        with open(STORE_FILE, 'w') as f:
            json.dump(safe, f, ensure_ascii=False)
    except Exception as e:
        print(f'[STORAGE] Save failed: {e}')

# Load existing store on module import
_load_store()


# ═══════════════════════════════════════════════════════════════
#  HTTP Handler
# ═══════════════════════════════════════════════════════════════

class CervicalAPI(BaseHTTPRequestHandler):

    def _get_path(self):
        raw_path = urlsplit(self.path).path
        if raw_path != '/' and raw_path.endswith('/'):
            return raw_path.rstrip('/')
        return raw_path

    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{ts}] {msg}", flush=True)

    def _parse_request(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self._log("_parse_request: content_length=0")
            return {}

        raw_bytes = self.rfile.read(content_length)
        content_type = self.headers.get('Content-Type', '').lower()
        self._log(f"_parse_request: content_type={content_type[:60]}, content_length={content_length}")

        if 'multipart/form-data' in content_type:
            try:
                pdict = cgi.parse_header(self.headers.get('Content-Type', ''))[1]
                boundary = pdict.get('boundary')
                if not boundary:
                    return {}
                pdict['boundary'] = boundary.encode('utf-8')
                pdict['CONTENT-LENGTH'] = content_length
                form_data = cgi.parse_multipart(io.BytesIO(raw_bytes), pdict)
                normalized = {}
                for key, values in form_data.items():
                    if len(values) == 1:
                        normalized[key] = values[0]
                    else:
                        normalized[key] = values
                self._log(f"Multipart parsed: keys={list(normalized.keys())[:10]}")
                return normalized
            except:
                self._log("Multipart parse failed")
                return {}

        # Try JSON
        try:
            decoded = raw_bytes.decode('utf-8')
        except:
            try:
                decoded = raw_bytes.decode('gbk')
            except:
                decoded = raw_bytes.decode('utf-8', errors='ignore')

        try:
            result = json.loads(decoded)
            if isinstance(result, dict):
                return result
            # If result is a string, treat it as a base64 image in 'image' field
            return {'image': result}
        except:
            # Not JSON — maybe raw image bytes? Return as 'raw_bytes' for _save_image
            if len(raw_bytes) > 100:
                self._log("Treating raw body as image bytes")
                return {'raw_bytes': raw_bytes}
            return {}

    def _send_response(self, status_code, data):
        json_str = json.dumps(data, ensure_ascii=False)
        utf8_bytes = json_str.encode('utf-8')
        allow_headers = self.headers.get(
            'Access-Control-Request-Headers',
            'Content-Type, Authorization, X-Requested-With'
        )
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', allow_headers)
        self.send_header('Content-Length', str(len(utf8_bytes)))
        self.end_headers()
        self.wfile.write(utf8_bytes)

    def do_OPTIONS(self):
        allow_headers = self.headers.get(
            'Access-Control-Request-Headers',
            'Content-Type, Authorization, X-Requested-With'
        )
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', allow_headers)
        self.end_headers()

    def do_GET(self):
        path = self._get_path()
        if path == '/':
            self._send_response(200, {
                "service": "Cervical Spine Diagnosis API",
                "version": "17.0 (Real Models)",
                "status": "running",
                "device": str(get_device()),
                "port": PORT,
            })
        elif path == '/debug/store':
            safe_store = {}
            for k, v in image_store.items():
                safe_store[k] = {
                    'has_disease': v.get('has_disease'),
                    'detection_box': v.get('detection_box'),
                    'upload_time': v.get('upload_time'),
                }
            self._send_response(200, {"code": 200, "data": safe_store})
        else:
            self._send_response(404, {"code": 404, "message": f"Not found: {path}"})

    def do_POST(self):
        path = self._get_path()
        self._log(f"POST {path}")

        try:
            request_data = self._parse_request()

            if path == '/cervical_upload':
                self._handle_upload(request_data)
            elif path == '/cervical_rediagnose':
                self._handle_rediagnose(request_data)
            elif path == '/get_detailed_diagnosis':
                self._handle_detailed(request_data)
            elif path == '/debug/clear':
                global image_store
                image_store = {}
                self._send_response(200, {"code": 200, "message": "store cleared"})
            else:
                self._send_response(404, {"code": 404, "message": f"Not found: {path}"})
        except Exception as e:
            self._log(f"ERROR: {e}")
            traceback.print_exc()
            self._send_response(500, {"code": 500, "message": str(e)})

    def _save_image(self, request_data):
        """Extract image from request. Tries ALL fields — accepts any key name."""
        self._log(f"Request keys: {list(request_data.keys())[:10]}")
        if not request_data:
            self._log("WARNING: empty request_data (check Content-Type header)")

        # ── Try ALL fields ──
        for key, val in list(request_data.items()):
            # Case 1: bytes from multipart upload
            if isinstance(val, bytes) and len(val) > 100:
                self._log(f"Found bytes in key '{key}', len={len(val)}")
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                tmp.write(val)
                tmp.close()
                return tmp.name

            # Case 2: long base64 string
            if isinstance(val, str) and len(val) > 100:
                clean = val
                if ';base64,' in clean:
                    clean = clean.split(';base64,', 1)[1]
                try:
                    img_bytes = base64.b64decode(clean)
                    if len(img_bytes) > 100:
                        self._log(f"Found base64 in key '{key}', decoded={len(img_bytes)} bytes")
                        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        tmp.write(img_bytes)
                        tmp.close()
                        return tmp.name
                except:
                    pass

            # Case 3: direct file path
            if isinstance(val, str) and len(val) < 500 and os.path.exists(val):
                self._log(f"Found file path in key '{key}': {val}")
                return val

            # Case 4: nested dict (e.g. {"image": "base64..."})
            if isinstance(val, dict):
                result = self._save_image(val)
                if result:
                    return result

        self._log("WARNING: No image data found in any request field")
        return None

    def _handle_upload(self, request_data):
        """POST /cervical_upload — initial diagnosis with image upload."""
        self._log("Processing upload + inference...")

        # Save image
        img_path = self._save_image(request_data)
        if img_path is None:
            self._send_response(400, {
                "code": 400,
                "message": "No valid image provided. Send 'image' (base64 string) or 'image_file' (file upload).",
                "data": None
            })
            return

        try:
            result = run_inference(img_path)
        finally:
            pass  # keep temp file for now, will copy below

        if not result.get('success'):
            if img_path.startswith(tempfile.gettempdir()):
                try: os.unlink(img_path)
                except: pass
            self._send_response(400, {
                "code": 400,
                "message": result.get('error', 'Inference failed'),
                "data": None
            })
            return

        # Generate image_id and store image persistently
        image_id = f"img_{int(time.time()) % 100000:05d}_{uuid4().hex[:8]}"
        saved_path = UPLOAD_DIR / f"{image_id}.jpg"
        # Copy temp file to persistent location
        shutil.copy(img_path, str(saved_path))

        # Clean up temp file
        if img_path.startswith(tempfile.gettempdir()):
            try: os.unlink(img_path)
            except: pass

        global image_store
        image_store[image_id] = {
            "has_disease": result['has_disease'],
            "binary_confidence": result['binary_confidence'],
            "detection_box": result['detection_box'],
            "diseases": result['diseases'],
            "image_path": str(saved_path),
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timing_ms": result['timing_ms'],
        }
        _save_store()

        detected_list = result.get('detected_diseases', [])
        all_diseases = result.get('diseases', [])
        if not detected_list:
            detected_list = [{
                "disease_id": 7,
                "disease_name": NEGATIVE_LABEL,
                "confidence": round(1.0 - max(d['confidence'] for d in all_diseases), 4),
            }]
            next_step = "can_adjust_box"
        else:
            next_step = "get_detailed_diagnosis"

        self._send_response(200, {
            "code": 200,
            "message": "success",
            "data": {
                "image_id": image_id,
                "has_disease": result['has_disease'],
                "binary_confidence": result['binary_confidence'],
                "detection_box": result['detection_box'],
                "detection_box_source": result['detection_box_source'],
                "detected_diseases": detected_list,
                "all_diseases": all_diseases,
                "timing_ms": result['timing_ms'],
                "next_step": next_step,
            }
        })

    def _handle_rediagnose(self, request_data):
        """POST /cervical_rediagnose — user adjusts box, re-inference (same API as v16)"""
        self._log("Processing rediagnose...")

        image_id = request_data.get('image_id')
        new_box = request_data.get('detection_box')

        if not image_id:
            self._send_response(400, {"code": 400, "message": "image_id required"})
            return
        if not new_box:
            self._send_response(400, {"code": 400, "message": "detection_box required"})
            return

        required = ['x', 'y', 'width', 'height']
        missing = [f for f in required if f not in new_box]
        if missing:
            self._send_response(400, {"code": 400, "message": f"Box missing: {missing}"})
            return

        global image_store
        if image_id not in image_store:
            self._send_response(400, {"code": 400, "message": f"image_id not found: {image_id}"})
            return

        # Load saved image from disk
        saved_path = image_store[image_id].get('image_path')
        if not saved_path or not os.path.exists(saved_path):
            self._send_response(400, {
                "code": 400,
                "message": "Saved image expired. Please re-upload.",
            })
            return

        # Re-run inference with new bbox (on saved image, no re-upload needed)
        result = run_inference(saved_path, bbox_override=new_box)

        if not result.get('success'):
            self._send_response(400, {"code": 400, "message": result.get('error')})
            return

        # Update store with new results
        old_data = image_store[image_id]
        image_store[image_id] = {
            "has_disease": result['has_disease'],
            "binary_confidence": result['binary_confidence'],
            "detection_box": result['detection_box'],
            "diseases": result['diseases'],
            "image_path": saved_path,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timing_ms": result['timing_ms'],
        }
        _save_store()

        detected_list = result.get('detected_diseases', [])
        if not detected_list:
            detected_list = [{"disease_id": 7, "disease_name": NEGATIVE_LABEL, "confidence": 0.95}]

        old_diseases = old_data.get('diseases', [])
        old_detected = [{"disease_id": d["disease_id"], "disease_name": d["disease_name"],
                         "confidence": d["confidence"]} for d in old_diseases if d.get('is_detected')]
        if not old_detected:
            old_detected = [{"disease_id": 7, "disease_name": NEGATIVE_LABEL, "confidence": 0.95}]

        old_result_data = {
            "has_disease": old_data.get('has_disease'),
            "detection_box": old_data.get('detection_box', {}),
            "detected_diseases": old_detected,
        }
        new_result_data = {
            "has_disease": result['has_disease'],
            "detection_box": result['detection_box'],
            "detected_diseases": detected_list,
            "all_diseases": result.get('diseases', []),
        }

        self._send_response(200, {
            "code": 200,
            "message": "success",
            "data": {
                "image_id": image_id,
                "old_data": old_result_data,
                "new_data": new_result_data,
                "old_result": old_result_data,
                "new_result": new_result_data,
                "has_disease": result['has_disease'],
                "adjusted_box": result['detection_box'],
                "confidence": result['binary_confidence'],
                "all_diseases": result.get('diseases', []),
                "rediagnosis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "next_step": "get_detailed_diagnosis" if result['has_disease'] else "cervical_upload",
            }
        })

    def _handle_detailed(self, request_data):
        """POST /get_detailed_diagnosis — return full 6-class breakdown."""
        self._log("Processing detailed diagnosis...")

        image_id = request_data.get('image_id')
        if not image_id:
            self._send_response(400, {"code": 400, "message": "image_id required"})
            return

        global image_store
        if image_id not in image_store:
            self._send_response(400, {"code": 400, "message": f"image_id not found: {image_id}"})
            return

        stored = image_store[image_id]
        diseases = stored.get('diseases', [])

        if not stored.get('has_disease'):
            self._send_response(200, {
                "code": 200,
                "message": "No disease detected",
                "data": {
                    "image_id": image_id,
                    "has_disease": False,
                    "detailed_results": diseases,
                    "summary": {"total_diseases": 6, "detected_count": 0},
                }
            })
            return

        detected_count = sum(1 for d in diseases if d.get('is_detected'))

        self._send_response(200, {
            "code": 200,
            "message": "success",
            "data": {
                "image_id": image_id,
                "has_disease": True,
                "detection_box": stored.get('detection_box'),
                "binary_confidence": stored.get('binary_confidence'),
                "detailed_results": diseases,
                "summary": {
                    "total_diseases": 6,
                    "detected_count": detected_count,
                },
                "timing_ms": stored.get('timing_ms'),
            }
        })


def main():
    print()
    print("=" * 70)
    print("  CVJ-SDAS Diagnosis API v17.0 — Real Model Inference")
    print("=" * 70)
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Device   : {get_device()}")
    print(f"  Port     : {PORT}")
    print(f"  Endpoints:")
    print(f"    POST /cervical_upload      — initial image + diagnosis")
    print(f"    POST /cervical_rediagnose   — adjust bbox + re-infer")
    print(f"    POST /get_detailed_diagnosis — full 6-class breakdown")
    print("=" * 70)
    print()

    # Pre-load models at startup
    print("[STARTUP] Loading models...")
    try:
        load_models()
        print("[STARTUP] All models loaded successfully!")
    except Exception as e:
        print(f"[STARTUP] WARNING: Model loading failed: {e}")
        print("[STARTUP] Models will be loaded on first request instead.")
        traceback.print_exc()
    print()

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedServer(('0.0.0.0', PORT), CervicalAPI)
    print(f'[STARTUP] Threaded server listening on port {PORT}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAPI stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
