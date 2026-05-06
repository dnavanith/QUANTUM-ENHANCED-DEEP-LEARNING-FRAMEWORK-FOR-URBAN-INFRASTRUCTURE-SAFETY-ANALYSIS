# Quantum-Enhanced Deep Learning Framework for Urban Infrastructure Safety Analysis

A hybrid quantum-classical deep learning framework for automated road hazard detection and classification using **YOLO**, **ResNet-18**, and a **4-qubit Variational Quantum Circuit (VQC)** implemented with **PennyLane** and **PyTorch**.

This project focuses on improving urban infrastructure safety by detecting and classifying road hazards such as potholes, cracks, open manholes, and garbage through a two-stage intelligent pipeline.

---

## Overview

Urban roads deteriorate over time due to traffic load, weather conditions, and poor maintenance. Manual inspection methods are expensive, slow, and difficult to scale.

This project proposes a **Quantum-Enhanced Deep Learning Framework** that combines classical computer vision with quantum machine learning techniques for efficient road hazard analysis.

The system follows a **two-stage architecture**:

1. Hazard Detection using YOLO-based object detection
2. Hazard Classification using a hybrid quantum-classical model

---

## Features

- Hybrid Quantum-Classical Architecture
- YOLO-based Region of Interest (ROI) Detection
- ResNet-18 Feature Extraction
- MLP-based Dimensionality Reduction
- 4-Qubit Variational Quantum Circuit (VQC)
- End-to-End Training with Parameter-Shift Rule
- Multi-Class Hazard Classification
- PyTorch + PennyLane Integration
- Road Hazard Visualization Pipeline

---

## Hazard Classes

The model classifies road hazards into four categories:

- Pothole
- Crack
- Open Manhole
- Garbage

---

## System Architecture

### Stage 1 — Detection
- YOLO detects hazard regions from road images
- Bounding boxes are generated for potential hazards
- Regions of interest are cropped and passed to the classifier

### Stage 2 — Hybrid Quantum Classification

The classification pipeline consists of:

- **ResNet-18 Backbone**
  - Extracts 512-dimensional feature vectors

- **MLP Dimensionality Reduction**
  - Reduces features from 512 → 64 → 4 dimensions

- **Variational Quantum Circuit**
  - 4-qubit VQC
  - Angle encoding using `RY` rotations
  - Ring-topology entanglement using CNOT gates
  - Pauli-Z expectation measurements

- **Classification Head**
  - Softmax-based final prediction layer

---

## Dataset

The dataset was created by combining multiple publicly available datasets:

- Pothole, Crack & Open Manhole Dataset (Kaggle)
- Garbage Detection Dataset (Roboflow)

### Final Dataset Statistics

| Split | Images |
|---|---|
| Training | 3,340 |
| Validation | 740 |
| Test | 258 |

### Crop Dataset Statistics

| Split | Crops |
|---|---|
| Training | 3,738 |
| Validation | 994 |
| Test | 273 |

---

## Technologies Used

### Languages & Frameworks
- Python
- PyTorch
- PennyLane
- YOLO (Ultralytics)

### Libraries
- NumPy
- torchvision
- matplotlib
- scikit-learn

### Development Environment
- Google Colab
- NVIDIA T4 GPU

---

## Model Configuration

| Hyperparameter | Value |
|---|---|
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Dropout | 0.3 |
| Qubits | 4 |
| VQC Depth | 6 Layers |

---

## Results

### Overall Performance

| Configuration | F1-Score |
|---|---|
| Classical Dataset + Classical Model | 0.8016 |
| Classical Dataset + Quantum Model | **0.8439** |

### Key Improvements
- Improved recall compared to classical baseline
- Better classification performance using quantum-enhanced representation learning
- Stable end-to-end hybrid training

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Pothole | 0.94 | 0.91 | 0.92 |
| Crack | 0.92 | 0.73 | 0.82 |
| Garbage | 0.79 | 0.98 | 0.88 |
| Open Manhole | 0.74 | 0.85 | 0.79 |

---

## Training Pipeline

1. Load and preprocess images
2. Extract hazard regions using bounding boxes
3. Resize images to `64×64`
4. Normalize using ImageNet statistics
5. Extract features using ResNet-18
6. Reduce dimensionality using MLP
7. Encode latent vector into quantum circuit
8. Perform quantum variational processing
9. Measure Pauli-Z expectation values
10. Predict hazard class using softmax classifier

---

## Quantum Circuit Design

The Variational Quantum Circuit uses:

- 4 Qubits
- Angle Encoding (`RY`)
- Trainable Variational Layers
- Ring Entanglement Topology
- Pauli-Z Measurements
- Parameter-Shift Gradient Computation

Implemented using:
- PennyLane `default.qubit` simulator

---
## Future Enhancements

- Deployment on real quantum hardware

- Noise-resilient circuit design

- Real-time smart city integration

- Severity-based hazard classification

- Larger and more diverse datasets

- Explainable AI integration

- Quantum kernel methods

---
