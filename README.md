# 🖼️ CNN Multi-Class Image Classification on CIFAR-10 | PPGEEC2318

This repository contains the implementation of the **Final Project – Part 1** for the course **PPGEEC2318 – Machine Learning**, instructed by Professor **Ivanovitch Medeiros**. We build, train and evaluate convolutional neural networks (CNNs) on the CIFAR‑10 dataset, including data preparation, model definition, training/validation, intermediate feature inspections and final test evaluation.

---

## 📌 Project Overview

The goal of this project is to design several CNN architectures for **multi‑class image classification** on the CIFAR‑10 dataset. We follow these steps:

1. **Data Preparation & EDA**

   - Load CIFAR‑10, visualize sample images
   - Class distribution histograms
   - Standard deviation heatmaps per class

       <!-- Replace with your EDA images -->

     ![Class Distribution Histogram](images/class_distribution_cifar10.png)
     ![Mean Images By Class](images/mean_images_by_class.png)
     ![Std Heatmap per Class](images/std_heatmap_by_class.png)

2. **Model Definition (Base V1)**

   - Three convolutional blocks (Conv→BN→ReLU→Conv→BN→ReLU→Pool→Dropout)
   - Fully‑connected head (Flatten→Dense→BN→ReLU→Dropout→Dense)
   - Implementation in `./cnn_model.ipynb` as `CNNV1`

3. **Training & Validation**

   - `Architecture` class manages train/val loops, loss/metric logging
   - Early stopping, learning‑rate scheduling, augmentation

4. **Filter & Feature‑Map Inspection**

   - Visualize first‑layer filters  
     ![Conv1 Filters](images/filter_first_conv2d.png)
   - Attach hooks to layers `conv1`, `relu1`, `pool1`, `flatten`
   - Display feature maps for a batch  
     ![Feature Maps](images/feature_maps.png)

5. **Final Test Evaluation**
   - Evaluate on unseen CIFAR‑10 test set
   - Compute test accuracy
   - Plot confusion matrix  
     ![Confusion Matrix](images/confusion_matrix_test_set.png)

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch** & **torchvision**
- **Scikit‑learn** (metrics, confusion matrix)
- **Matplotlib** & **Seaborn** (visualizations)

---

## 📁 Repository Structure

```

.
├── data/               # CIFAR‑10 download scripts or pointers
├── images/
│   ├── eda\_class\_dist.png
│   ├── eda\_std\_heatmap.png
│   ├── cnn\_filters.png
│   ├── feature\_maps.png
│   └── confusion\_matrix.png
├── eda.ipynb
├── Architecture.py
├── cnn_model.ipynb          # contains CNNV1 class
├── requirements.txt        # project dependencies
└── README.md               # this file

```

---

# 📝 Model Card – CIFAR‑10 CNN Classifier

Model cards provide context, usage guidelines, and limitations of ML models.

---

## 📌 Model Details

- **Author:** Efrain Marcelo
- **Framework:** PyTorch
- **Architecture:** `CNNV1` – three convolutional blocks with batch‑norm, ReLU, pooling and dropout; two‑layer MLP head
- **Training pipeline:** managed by custom `Architecture` class for reproducible train/val loops, early stopping, LR scheduling, hooks

---

## 🎯 Intended Use

This model is intended for **educational** and **experimental** purposes, demonstrating:

- End‑to‑end CNN pipeline on CIFAR‑10
- Data augmentation strategies
- Inspection of learned filters and intermediate activations
- Handling of train/validation/test splits

**Not** for real‑world image classification deployment without further tuning and robustness checks.

---

## 🧪 Training Data

- **Dataset:** CIFAR‑10 (60 000 color images, 10 classes)
- **Split:** 80% train, 20% validation (random split), separate 10 000 test images unseen during training
- **Augmentation:** RandomCrop(32, pad=4), RandomHorizontalFlip

---

## 🧪 Evaluation Data

- **Validation:** used for early stopping and LR scheduling
- **Test:** final unseen set for metrics

---

## 📈 Metrics

- **Test Accuracy:** 0.7422
- **Confusion Matrix:** see above for class‑wise performance

---

## ⚖️ Ethical Considerations

- CIFAR‑10 classes (e.g., “cat”, “dog”, “truck”) are generic and carry minimal sensitive bias.
- Nevertheless, the pipeline illustrates general methods that could be misapplied to sensitive data in production.

---

## ⚠️ Caveats and Recommendations

- **Overfitting risk:** monitor train vs. val loss gap; use early stopping and augmentations.
- **Further improvements:** deeper architectures, transfer learning (e.g., ResNet‑18), advanced augmentations (CutMix, MixUp), extensive hyperparameter search.
- **Interpretability:** use hooks, filter visualizations, and SHAP/LIME for deeper analysis.
