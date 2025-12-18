# 🐛 Paddy Field Insect Segmentation using Mask R-CNN

This repository presents an end-to-end **computer vision pipeline for insect detection and instance segmentation in paddy field images** using **Mask R-CNN**. The project focuses on **polygonal annotation preprocessing**, **data augmentation**, **model customization**, and **performance evaluation** to support precision agriculture and automated pest management.

---

## 📌 Project Overview

Accurate identification of pests in paddy fields is critical for improving crop yield and reducing pesticide misuse. This project leverages **deep learning–based instance segmentation** to detect and segment insects with irregular shapes under varying environmental conditions.

Key highlights:

* Polygon-based annotations (CVAT → COCO format)
* Data augmentation for improved generalization
* Mask R-CNN with ResNet-50 backbone
* Quantitative and qualitative model evaluation

---

## 🗂 Dataset

* **Source**: [Paddy Pests Dataset – Kaggle](https://www.kaggle.com/datasets/zeeniye/paddy-pests-dataset)
* **Categories**:

  * Paddy with pests
  * Paddy without pests
* **Annotations**:

  * Polygonal annotations generated using **CVAT**
  * COCO-style JSON format
* **Split**:

  * 80% Training
  * 20% Validation

---

## 🛠️ Technologies & Tools

* **Programming Language**: Python
* **Frameworks & Libraries**:

  * PyTorch
  * Torchvision
  * Mask R-CNN
  * pycocotools
  * Albumentations
  * OpenCV
  * NumPy
* **Annotation Tool**: CVAT
* **Model Backbone**: ResNet-50 (pre-trained on COCO)

---

## 🔄 Project Pipeline

```
Dataset Collection
        ↓
Polygon Annotation (CVAT)
        ↓
Preprocessing & COCO Conversion
        ↓
Data Augmentation
        ↓
Mask R-CNN Model Customization
        ↓
Training
        ↓
Evaluation & Visualization
```

---

## ⚙️ Data Preprocessing

* Validation of image–annotation consistency
* Conversion of polygon annotations to binary masks
* Normalization and resizing
* COCO dataset structure compliance
* Class balancing using random sampling

---

## 🔁 Data Augmentation

Applied to improve robustness and reduce overfitting:

* Horizontal & vertical flips
* Random rotations
* Scaling
* Brightness & contrast adjustments

All transformations maintain synchronization between images and masks.

---

## 🧠 Model Architecture

* **Model**: Mask R-CNN
* **Backbone**: ResNet-50 + FPN
* **Classes**:

  * Background
  * Insect
* **Optimizer**: SGD
* **Hyperparameters**:

  * Learning Rate: `0.005`
  * Momentum: `0.9`
  * Weight Decay: `0.0005`
  * Epochs: `5`

---

## 📊 Evaluation Metrics

The model was evaluated using:

* **Intersection over Union (IoU)**: `0.6177`
* **Mean Average Precision (mAP@0.5)**: `0.0828`
* **Pixel Accuracy**: `80.75%`
* **Overall Classification Accuracy**: `85.11%`

Visual inspections reveal strong performance on clean backgrounds, with challenges in dense or noisy scenes.

---

## 🖼 Sample Results

* Bounding boxes and segmentation masks overlaid on validation images
* Correct localization in simple scenes
* Reduced performance in clustered insect scenarios due to limited NMS tuning

---

## 📁 Repository Structure

```
├── data/
│   ├── train/
│   ├── val/
│   └── annotations/
├── augmentation/
├── model/
├── training/
├── evaluation/
├── utils/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/paddy-insect-segmentation.git
cd paddy-insect-segmentation
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Prepare dataset (COCO format)

```bash
python preprocess_dataset.py
```

4. Train the model

```bash
python train.py
```

5. Evaluate the model

```bash
python evaluate.py
```

---

## 🔮 Future Improvements

* Higher resolution and cleaner datasets
* Advanced data augmentation (elastic distortion, random cropping)
* Increased training epochs
* Hyperparameter tuning
* Improved Non-Maximum Suppression (NMS)
* Multi-class insect segmentation

---

## 📚 References

This work is supported by literature on Mask R-CNN, data augmentation, and instance segmentation, including studies from IEEE, ACM, and CVPR.

---

