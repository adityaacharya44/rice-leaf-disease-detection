# 🌾 Rice Leaf Disease Detection using Deep Learning

A computer vision project that classifies rice leaf diseases into three distinct categories—**Bacterial leaf blight**, **Brown spot**, and **Leaf smut**—using Transfer Learning with **MobileNetV2**.

---

## 📋 Overview

This project utilizes a transfer learning approach with MobileNetV2, enhanced by data augmentation and regularization to overcome data scarcity within a compact 120-image dataset. By comparing a baseline custom CNN trained from scratch against the pre-trained MobileNetV2 architecture, the project demonstrates the effectiveness of transfer learning for small agricultural datasets, achieving a peak validation accuracy of **86.96%**.

---

## 📊 Model Comparison & Performance

| Feature / Metric | Custom CNN (Baseline) | MobileNetV2 (Final Model) |
| :--- | :--- | :--- |
| **Peak Validation Accuracy** | ~56.52% | **86.96%** |
| **Validation Loss** | ~0.95+ | **~0.4677** |
| **Training Strategy** | Trained from scratch | Transfer learning (frozen base + custom head) |
| **Handling of Data Scarcity** | Prone to overfitting | Leveraged ImageNet feature extractors |
| **Status** | Suboptimal | **Production-Ready Winner 🏆** |

---

## 🧠 Architecture & Dataset

* **Dataset:** 120 images divided across 3 classes (`Bacterial leaf blight`, `Brown spot`, `Leaf smut`).
* **Base Model:** Pre-trained `MobileNetV2` (ImageNet weights, frozen base layers).
* **Classification Head:** Global Average Pooling $\rightarrow$ Dense Layer (256 units, ReLU) $\rightarrow$ Dropout (0.4) $\rightarrow$ Softmax Output (3 classes).
* **Optimizer:** Adam (`lr = 0.001`).
* **Regularization:** Data augmentation (rotations, flips, zooms, shifts) paired with Dropout to prevent overfitting.

---

## 🚀 Installation & Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install tensorflow numpy pillow opencv-python
