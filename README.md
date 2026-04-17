# 🫁 PneumoNet — Deep Learning for Pneumonia Detection

A deep learning-based medical image classification system that detects **pneumonia** from chest X-ray images, achieving **93% accuracy** on the test set. Built with PyTorch and deployed as an interactive web application using Streamlit.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

---

## Overview

PneumoNet is a custom Convolutional Neural Network (CNN) trained to classify chest X-ray images as either **Normal** or **Pneumonia**. The model was trained on a balanced subset of the [Pneumonia Chest X-Ray Dataset](https://www.kaggle.com/datasets/faisalahmed07/pneumonia-chest-xray-dataset) using PyTorch and is served through a clean, interactive Streamlit application.

**Key highlights:**
- Custom CNN built from scratch (no pretrained backbone)
- Binary classification: `Normal (0)` vs `Pneumonia (1)` — covers both bacterial and viral pneumonia
- Balanced training set to avoid class bias
- Interactive web app with configurable prediction threshold
- Full evaluation suite: accuracy, precision, recall, F1-score, and confusion matrix

---

## Demo

The Streamlit app provides three main panels controlled via the sidebar:

| Panel | Description |
|---|---|
| **Model Details** | View training metrics, loss/accuracy curves, sample predictions, and confusion matrix |
| **Run Model** | Upload one or more chest X-ray images and receive real-time predictions |
| **Info** | Author links and project references |

---

## Model Architecture

The `PneumoniaClassifier` is a custom CNN with the following structure:

```
Input: 200 × 200 × 1 (grayscale)
│
├── Conv2d(1 → 30, kernel=3, stride=2)   → 99 × 99 × 30
├── ReLU
├── Conv2d(30 → 20, kernel=3, stride=2)  → 49 × 49 × 20
├── ReLU
├── MaxPool2d(kernel=3, stride=1)         → 47 × 47 × 20
├── Conv2d(20 → 30, kernel=3, stride=1)  → 45 × 45 × 30
├── ReLU
├── Conv2d(30 → 20, kernel=3, stride=1)  → 43 × 43 × 20
├── ReLU
├── MaxPool2d(kernel=3, stride=1)         → 41 × 41 × 20
│
├── Flatten → 33,620
│
├── Linear(33620 → 100) + ReLU
├── Linear(100 → 50)    + ReLU
├── Linear(50 → 20)     + ReLU
└── Linear(20 → 1)      + Sigmoid
```

**Output:** Scalar in `[0, 1]` — thresholded at `0.5` to produce a binary prediction.

---

## Dataset

- **Source:** [Pneumonia Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/faisalahmed07/pneumonia-chest-xray-dataset)
- **Classes:** `NORMAL`, `Bacterial` (→ Pneumonia), `Virus` (→ Pneumonia)
- **Class mapping:** Bacterial and Viral pneumonia are merged into a single `Pneumonia (1)` class
- **Balancing:** Training set is capped at the size of the minority class to ensure balance

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train | 1,267 | 1,267 | 2,534 |
| Test | varies | varies | ~3,322 |

**Image preprocessing pipeline:**
1. Convert to grayscale
2. Pad to 350 × 350
3. Resize to 200 × 200
4. Center crop to 200 × 200
5. Convert to tensor

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Prediction Threshold | 0.5 |
| Input Size | 200 × 200 × 1 |

**Epoch-by-epoch results:**

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 0.6537 | 76.16% |
| 2 | 0.2951 | 85.07% |
| 3 | 0.2251 | 89.64% |
| 4 | 0.2087 | 91.15% |
| 5 | 0.1952 | 91.63% |
| 6 | 0.1901 | 85.43% |
| 7 | 0.1758 | 84.14% |
| 8 | 0.1794 | 91.93% |
| 9 | 0.1639 | 93.35% |
| 10 | 0.1629 | 93.02% |

---

## Results

### Evaluation Metrics (Test Set)

| Metric | Value |
|---|---|
| **Accuracy** | **93.02%** |
| **Precision** | 99.47% |
| **Recall** | 92.78% |
| **F1-Score** | 96.01% |

### Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 301 (TN) | 15 (FP) |
| **Actual Pneumonia** | 217 (FN) | 2789 (TP) |

> **Note:** The model achieves very high precision (99.47%), meaning nearly all positive predictions are correct. Recall of 92.78% indicates a small number of pneumonia cases are missed — an important consideration for clinical use.

<div style="display: flex; flex-direction: column; align-items: center; gap: 40px;">

  <!-- Image 1 -->
  <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://github.com/user-attachments/assets/ea993489-6777-431e-ad20-2182f860cb03" style="max-width: 80%; border-radius: 10px;" />
    <p style="margin-top: 10px; font-size: 16px;">
      <!-- Write your description here -->
      Description for the first image goes here.
    </p>
  </div>

  <!-- Image 2 -->
  <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://github.com/user-attachments/assets/757e76d2-31d6-45d5-b58a-66b42bf35545" style="max-width: 80%; border-radius: 10px;" />
    <p style="margin-top: 10px; font-size: 16px;">
      Description for the second image goes here.
    </p>
  </div>

  <!-- Image 3 -->
  <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://github.com/user-attachments/assets/41bdb561-a215-426e-87f1-32829190b567" style="max-width: 80%; border-radius: 10px;" />
    <p style="margin-top: 10px; font-size: 16px;">
      Description for the third image goes here.
    </p>
  </div>

</div>


## Project Structure

```
PneumoNet/
│
├── main.py                     # Streamlit application entry point
│
├── Utilities/
│   └── Utilities.py            # Model definition, transforms, and helper functions
│
├── model/
│   └── Pneumonia_model.pk      # Serialized trained model (pickle)
│
├── assets/
│   └── styles.css              # Custom Streamlit CSS styling
│
├── Images/
│   ├── accuracy_loss.png       # Training curves
│   ├── predictions.png         # Sample predictions visualization
│   └── Confusion_matrix.png    # Confusion matrix heatmap
│
├── Metrics/
│   └── metrics.csv             # Evaluation metrics table
│
├── pneumonia-project.ipynb     # Training notebook (Kaggle)
└── requirements.txt            # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/OmarGamalH/PneumoNet-Deep-Learning-for-Pneumonia-Detection.git
cd PneumoNet-Deep-Learning-for-Pneumonia-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run main.py
```

---

## Usage

1. Launch the app with `streamlit run main.py`
2. Open the browser at `http://localhost:8501`
3. Use the **sidebar** to toggle sections:
   - Enable **Show Model Details** to explore training metrics and evaluation results
   - Enable **Run Model** to upload X-ray images and get predictions
4. Upload `.jpg`, `.jpeg`, or `.png` chest X-ray images
5. The model will display both the original and preprocessed image alongside the prediction (`Normal` or `Pneumonia`)

> ⚠️ **Medical Disclaimer:** This tool is intended for research and educational purposes only. It is **not** a substitute for professional medical diagnosis.

---

## Author

**Omar Gamal Hamed**

[![GitHub](https://img.shields.io/badge/GitHub-OmarGamalH-181717?logo=github)](https://github.com/OmarGamalH/PneumoNet-Deep-Learning-for-Pneumonia-Detection)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-omar--gamal--hamed-0A66C2?logo=linkedin)](https://www.linkedin.com/in/omar-gamal-hamed/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
