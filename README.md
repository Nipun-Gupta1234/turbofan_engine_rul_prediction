# 🚀 Turbofan Remaining Useful Life (RUL) Prediction using Deep Learning

Predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the **NASA C-MAPSS FD001 dataset**. This project compares multiple deep learning architectures for predictive maintenance by jointly learning:

- 📈 Remaining Useful Life (Regression)
- ⚠️ Imminent Engine Failure (Binary Classification)

---

## 📌 Overview

Predictive maintenance helps estimate how much useful life an engine has before failure, reducing downtime and maintenance costs.

This project implements a **multi-task learning** framework that simultaneously predicts:

- Remaining Useful Life (RUL)
- Failure within a predefined threshold

Three sequence models are compared:

- LSTM
- GRU
- Temporal Convolutional Network (TCN)

---

## ✨ Features

- NASA C-MAPSS FD001 dataset preprocessing
- Sliding window sequence generation
- Feature normalization using StandardScaler
- Multi-task learning architecture
- Joint regression + classification loss
- Automatic uncertainty weighting
- Comparison of LSTM, GRU and TCN
- Validation and test evaluation
- Training history visualization
- GPU acceleration with PyTorch

---

## 📂 Dataset

Dataset used:

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**

The project uses the **FD001** subset.

Dataset files required:

```
data/
└── cmapss_data/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

## 🧠 Models

### LSTM
Long Short-Term Memory network for sequential degradation modeling.

### GRU
A lightweight recurrent architecture with fewer parameters than LSTM.

### TCN
Temporal Convolutional Network using dilated convolutions for long-range temporal dependencies.

---

## 🏗️ Project Pipeline

```
NASA C-MAPSS Dataset
        │
        ▼
Data Loading
        │
        ▼
RUL Computation
        │
        ▼
Feature Scaling
        │
        ▼
Sliding Window Sequence Generation
        │
        ▼
Train / Validation Split
        │
        ▼
LSTM / GRU / TCN Models
        │
        ▼
Joint Loss
(Regression + Classification)
        │
        ▼
Training
        │
        ▼
Evaluation & Visualization
```

---

## ⚙️ Training Details

- Framework: PyTorch
- Window Size: 30
- Batch Size: 64
- Hidden Dimension: 128
- Number of Layers: 2
- Dropout: 0.3
- Optimizer: Adam
- Loss:
  - Mean Squared Error (RUL)
  - Binary Cross Entropy (Failure Classification)
  - Optional uncertainty-based weighting

---

## 📊 Evaluation Metrics

### Regression

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Classification

- Accuracy
- Precision
- Recall
- F1 Score

---

## 📈 Visualizations

The notebook includes:

- Training & validation loss curves
- Regression loss
- Classification loss
- Class distribution
- Model comparison

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

Launch the notebook

```bash
jupyter notebook turbofan_rul_prediction.ipynb
```

or

```bash
jupyter lab
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📁 Repository Structure

```
.
├── data/
│   └── cmapss_data/
├── turbofan_rul_prediction.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 📚 References

- NASA C-MAPSS Turbofan Engine Dataset
- PyTorch Documentation
- Predictive Maintenance Literature

---

## 👨‍💻 Author

**Nipun Gupta**

If you found this project helpful, consider giving the repository a ⭐.