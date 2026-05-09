# Financial Transaction Category Classification


## 1. Project Overview

This project focuses on classifying personal financial transactions into predefined categories using four different supervised machine learning techniques. The goal is to automate financial categorization to support budgeting, expense tracking, and financial planning.

We used the Personal Transactions dataset and implemented four classification models:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* K-Nearest Neighbors (KNN)

Models were evaluated using accuracy, precision, recall, and F1-score, with emphasis on handling class imbalance.

---

## 2. Repository Structure

```
ProjectFinalReport/
├── data/                          # Dataset
│   ├── cleaned/                   # Finalized dataset used for model training
│   ├── raw/                       # Original data
│   ├── data_cleaning.ipynb        # Initial exploration and cleaning
│   ├── data_loading.py            # Shared preprocessing pipeline
├── models/                        # All files for the four models
│   ├── logistic_regression.ipynb
│   ├── svm.ipynb
│   ├── random_forest.ipynb
│   └── knn.ipynb
├── figures/                       # Outputs (confusion matrices, reports)
├── requirements.txt
└── README.md
```

---

## 3. Environment Setup

* Python version: **3.10**
* Tested on: macOS / Windows
* Hardware: CPU

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries:

* numpy
* pandas
* scikit-learn
* matplotlib
* imbalanced-learn

---

## 4. Dataset Instructions

Dataset used:

* **Personal Transactions Dataset**
* Source: https://www.kaggle.com/datasets/shyakanobledavid/personal-transactions-userid-new-transactions/data

### Note:
The cleaned dataset is already included in this repository under:

data/cleaned/

No additional download is required. The code is configured to use this dataset directly.


---


## 5. Training Setup

* Train/test split: **80/20**
* Cross-validation used for hyperparameter tuning
* Consistent preprocessing pipeline across all models

---

## 6. Evaluation Metrics

Due to class imbalance, the following metrics were used:

* **Weighted F1-score (primary metric)**
* Macro F1-score
* Accuracy
* Precision and Recall

F1-score was emphasized because accuracy alone can be misleading for imbalanced datasets.


---

## 7. How to Run the Project

### Step 1: Ensure dataset is placed in `data/`

### Step 2: Run preprocessing

```bash
python data/data_loading.py
```

### Step 3: Run models

Open and run each notebook in the `models/` folder using Jupyter Notebook or VS Code:

- models/logistic_regression.ipynb  
- models/svm.ipynb  
- models/random_forest.ipynb  
- models/knn.ipynb  

Run all cells in each notebook to generate results.


Outputs (metrics and confusion matrices) will be saved under `figures/`.

---

## 8. Reproducibility Notes

* Random seed: 42
* Same preprocessing pipeline used across all models
* Approximate runtime:

  * Logistic Regression / SVM: fast
  * Random Forest: moderate
  * KNN: slower at prediction time

---

## 9. Individual Contributions

* **Aleksa Ocampo**

  * Developed Random Forest model
  * Model evaluation and comparison
  * Contributed to Discussion section

* **Alisa Crowe**

  * Data preprocessing and feature engineering
  * Developed Logistic Regression model
  * Contributed to Methodology section

* **Angela Lee**

  * Developed SVM model
  * Model evaluation and comparison
  * Wrote Abstract and Introduction

* **Brianna Sengchan**

  * Data preprocessing and encoding
  * Developed KNN model
  * Wrote Conclusion and Future Work

---

