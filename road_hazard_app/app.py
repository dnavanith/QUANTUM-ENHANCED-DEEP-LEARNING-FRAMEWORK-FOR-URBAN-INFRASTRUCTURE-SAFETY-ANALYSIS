"""
Road Hazard Detection — Dual Model Comparison Server
=====================================================
Classical YOLOv8n  vs  Quantum-Enhanced (ResNet18 + VQC)
"""

import os, io, base64, warnings
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████████████████████
#                       ★  LABEL CONFIGURATION  ★
#
#  If your training labels got mixed up, fix them HERE by editing LABEL_REMAP.
#
#  During training, class indices were assigned as:
#    0 → cracks      (verify this is correct)
#    1 → open_manhole
#    2 → pothole
#    3 → garbage
#
#  CLASS_NAMES is the ORDER your model learned (indices 0–3).
#  LABEL_REMAP lets you reassign what each index DISPLAYS as.
#
#  Example — if index 0 was accidentally trained as "pothole" not "cracks":
#    LABEL_REMAP = {0: "pothole", 1: "open_manhole", 2: "cracks", 3: "garbage"}
#
#  If no remapping is needed, leave LABEL_REMAP = {} (identity mapping).
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['pothole', 'cracks', 'open_manhole', 'garbage']

# ▼▼▼  EDIT THIS DICT TO FIX LABEL MIXUPS  ▼▼▼
LABEL_REMAP = {
    # index : "correct_label_name"
    # 0: "pothole",          # ← uncomment & edit as needed
    # 1: "open_manhole",
    # 2: "cracks",
    # 3: "garbage",
}
# ▲▲▲  END OF LABEL CONFIG  ▲▲▲

# ─────────────────────────────────────────────────────────────────────────────
#   MODEL PATHS  — point these to your local .pt / .pth files
# ─────────────────────────────────────────────────────────────────────────────
YOLO_WEIGHTS_PATH    = "weights/best_m1.pt"        # YOLOv8n model weights
QUANTUM_WEIGHTS_PATH = "weights/best_m3.pth"       # QuantumDetectionRefiner weights

# ─────────────────────────────────────────────────────────────────────────────
#   VISUAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Per-class colours  (R, G, B)
CLASS_COLORS = {
    "cracks":       (255, 80,  80),
    "open_manhole": (80,  200, 120),
    "pothole":      (80,  160, 255),
    "garbage":      (255, 180, 50),
}
DEFAULT_COLOR = (200, 200, 200)

# ─────────────────────────────────────────────────────────────────────────────
#   HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def resolve_label(idx: int) -> str:
    """Return display label for a class index, honouring LABEL_REMAP."""
    if LABEL_REMAP and idx in LABEL_REMAP:
        return LABEL_REMAP[idx]
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class_{idx}"


def img_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def draw_detections(pil_img, detections):
    """
    Draw bounding boxes on a PIL image.
    detections: list of {"label": str, "conf": float, "box": [x1,y1,x2,y2]}
    Returns a new PIL image.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        label = det["label"]
        conf  = det["conf"]
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # Box
        lw = max(2, int(min(img.width, img.height) / 200))
        for i in range(lw):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # Label background
        text = f"{label}  {conf:.0%}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        bbox_t = draw.textbbox((x1, y1), text, font=font)
        tw = bbox_t[2] - bbox_t[0]
        th = bbox_t[3] - bbox_t[1]
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), text, fill=(0, 0, 0), font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
#   MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
yolo_model    = None
quantum_model = None
torch_device  = None

def load_models():
    global yolo_model, quantum_model, torch_device

    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import pennylane as qml
    import torch.nn as nn
    from ultralytics import YOLO

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {torch_device}")

    # ── YOLO ──────────────────────────────────────────────────────────────
    if os.path.exists(YOLO_WEIGHTS_PATH):
        yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        print(f"[INFO] YOLO loaded from {YOLO_WEIGHTS_PATH}")
    else:
        print(f"[WARN] YOLO weights not found at '{YOLO_WEIGHTS_PATH}'. "
              "Returning empty detections for YOLO.")

    # ── Quantum Model ──────────────────────────────────────────────────────
    if os.path.exists(QUANTUM_WEIGHTS_PATH):
        N_QC_QUBITS = 4
        N_QC_LAYERS = 3
        NUM_CLASSES = len(CLASS_NAMES)

        dev = qml.device("default.qubit", wires=N_QC_QUBITS)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _vqc(inputs, weights):
            thetas = (inputs + 1.0) * (np.pi / 2.0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.ctrl(qml.RY, control=[0,1], control_values=[0,0])(2.0 * thetas[...,0], wires=2)
            qml.ctrl(qml.RY, control=[0,1], control_values=[0,1])(2.0 * thetas[...,1], wires=2)
            qml.ctrl(qml.RY, control=[0,1], control_values=[1,0])(2.0 * thetas[...,2], wires=2)
            qml.ctrl(qml.RY, control=[0,1], control_values=[1,1])(2.0 * thetas[...,3], wires=2)
            qml.CNOT(wires=[2, 3])
            qml.Hadamard(wires=3)
            qml.StronglyEntanglingLayers(weights, wires=range(N_QC_QUBITS))
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QC_QUBITS)]

        _VQC_WEIGHT_SHAPES = {"weights": (N_QC_LAYERS, N_QC_QUBITS, 3)}

        class QuantumDetectionRefiner(nn.Module):
            def __init__(self, n_classes=4):
                super().__init__()
                resnet = tv_models.resnet18(pretrained=False)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.pre_q = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512, 64), nn.ReLU(),
                    nn.Linear(64, N_QC_QUBITS), nn.Tanh(),
                )
                self.qlayer = qml.qnn.TorchLayer(_vqc, _VQC_WEIGHT_SHAPES)
                self.post_q = nn.Sequential(
                    nn.Linear(N_QC_QUBITS, 32), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(32, n_classes),
                )

            def forward(self, x):
                feats  = self.backbone(x)
                q_in   = self.pre_q(feats)
                q_out  = self.qlayer(q_in)
                return self.post_q(q_out)

        qmodel = QuantumDetectionRefiner(n_classes=NUM_CLASSES)
        state  = torch.load(QUANTUM_WEIGHTS_PATH, map_location=torch_device)
        # Handle both raw state_dict and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        qmodel.load_state_dict(state)
        qmodel.to(torch_device).eval()
        quantum_model = (qmodel, T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ]))
        print(f"[INFO] Quantum model loaded from {QUANTUM_WEIGHTS_PATH}")
    else:
        print(f"[WARN] Quantum weights not found at '{QUANTUM_WEIGHTS_PATH}'.")


# ─────────────────────────────────────────────────────────────────────────────
#   INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo(pil_img: Image.Image, conf_threshold: float):
    """Run YOLO and return list of detection dicts."""
    if yolo_model is None:
        return []

    results = yolo_model.predict(pil_img, conf=conf_threshold, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            idx  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "label": resolve_label(idx),
                "conf":  round(conf, 4),
                "box":   [round(x1), round(y1), round(x2), round(y2)],
            })
    return dets


def run_quantum_on_crops(pil_img: Image.Image, yolo_dets: list, conf_threshold: float):
    """
    For each YOLO bounding box, crop the region and classify with the quantum model.
    Returns a list of dicts with quantum-refined labels.
    """
    import torch, torch.nn.functional as F

    if quantum_model is None or not yolo_dets:
        return []

    qmodel, crop_tf = quantum_model
    refined = []

    with torch.no_grad():
        for det in yolo_dets:
            x1, y1, x2, y2 = det["box"]
            crop = pil_img.crop((x1, y1, x2, y2))
            if crop.width < 4 or crop.height < 4:
                continue

            tensor = crop_tf(crop).unsqueeze(0).to(torch_device)
            logits = qmodel(tensor)
            probs  = F.softmax(logits, dim=1)[0]
            conf   = float(probs.max())
            idx    = int(probs.argmax())

            if conf < conf_threshold:
                continue

            refined.append({
                "label":       resolve_label(idx),
                "conf":        round(conf, 4),
                "box":         det["box"],
                "class_probs": {resolve_label(i): round(float(p), 4)
                                for i, p in enumerate(probs)},
            })
    return refined


# ─────────────────────────────────────────────────────────────────────────────
#   FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit


@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        label_remap=LABEL_REMAP,
    )


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file         = request.files["image"]
    yolo_conf    = float(request.form.get("yolo_conf",    0.25))
    quantum_conf = float(request.form.get("quantum_conf", 0.25))

    # Load image
    img_bytes = file.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # ── YOLO ─────────────────────────────────────
    yolo_dets = run_yolo(pil_img, yolo_conf)
    yolo_img  = draw_detections(pil_img, yolo_dets)

    # ── Quantum ──────────────────────────────────
    q_dets   = run_quantum_on_crops(pil_img, yolo_dets, quantum_conf)
    q_img    = draw_detections(pil_img, q_dets)

    return jsonify({
        "yolo": {
            "image":      img_to_b64(yolo_img),
            "detections": yolo_dets,
            "count":      len(yolo_dets),
        },
        "quantum": {
            "image":      img_to_b64(q_img),
            "detections": q_dets,
            "count":      len(q_dets),
        },
        "original": img_to_b64(pil_img),
    })


@app.route("/status")
def status():
    return jsonify({
        "yolo_loaded":    yolo_model    is not None,
        "quantum_loaded": quantum_model is not None,
        "classes":        CLASS_NAMES,
        "label_remap":    LABEL_REMAP,
        "yolo_path":      YOLO_WEIGHTS_PATH,
        "quantum_path":   QUANTUM_WEIGHTS_PATH,
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
