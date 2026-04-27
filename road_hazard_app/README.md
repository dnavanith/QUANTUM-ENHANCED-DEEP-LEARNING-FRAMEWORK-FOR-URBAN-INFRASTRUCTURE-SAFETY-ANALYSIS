# Road Hazard Detection — Quantum vs Classical Web Interface

Side-by-side comparison of **YOLOv8n (classical)** and **ResNet18 + VQC (quantum-enhanced)** models.

---

## Setup

### 1. Install requirements
```bash
pip install flask ultralytics pennylane pennylane-lightning torch torchvision pillow opencv-python-headless
```

### 2. Place your model weights
```
road_hazard_app/
├── weights/
│   ├── best_m1.pt      ← YOLOv8n weights (Model 1)
│   └── best_m3.pth     ← QuantumDetectionRefiner weights (Model 3)
├── app.py
└── templates/
    └── index.html
```

Update paths in `app.py` if your weights are stored elsewhere:
```python
YOLO_WEIGHTS_PATH    = "weights/best_m1.pt"
QUANTUM_WEIGHTS_PATH = "weights/best_m3.pth"
```

### 3. Fix label mixups (if needed)
Open `app.py` and find the `LABEL_REMAP` section (clearly marked with ★).

```python
# If index 0 was trained as "pothole" instead of "cracks":
LABEL_REMAP = {
    0: "pothole",
    1: "open_manhole",
    2: "cracks",
    3: "garbage",
}
```

Leave `LABEL_REMAP = {}` if labels are correct as-is.

### 4. Run the server
```bash
cd road_hazard_app
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Features
- Upload any road image (JPG / PNG / WEBP)
- Adjustable confidence thresholds for each model independently
- Bounding boxes drawn on output images per model
- Per-detection confidence bars and quantum class probability breakdown
- Summary table comparing both models side by side
- Status pills showing which models loaded successfully

---

## Architecture reminder
```
Input Image (640×640)
    │
    ├─► YOLOv8n ──────────────────────────────────► Classical detections
    │   (Model 1, full detection)
    │
    └─► YOLOv8n bounding boxes
            │
            ▼
        Crops (64×64) per box
            │
            ▼
        ResNet18 backbone (frozen)  → 512-d features
            │
            ▼
        MLP (512 → 64 → 4)          → latent vector ∈ ℝ⁴
            │
            ▼
        4-qubit VQC (PennyLane)     → Pauli-Z expectations ∈ [-1,1]⁴
            │
            ▼
        Linear head + Softmax       → Quantum-refined class
```
