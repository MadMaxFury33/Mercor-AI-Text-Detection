# 🧠 Mercor AI Text Detection  
### Detecting AI-Generated Writing Using XGBoost, Optuna, RoBERTa-Base & RoBERTa-Large


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/XGBoost-1.7+-lightgrey?style=for-the-badge">
  <img src="https://img.shields.io/badge/Optuna-AutoML-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
</p>

---

This repository contains my complete solution for the **Mercor AI Text Detection Competition**, where the objective is to classify whether a writing sample is **human-written (0)** or **AI-generated (1)**.

I built, compared, and optimized several models:

- ✔️ **XGBoost baseline**  
- ✔️ **Optuna hyperparameter-tuned XGBoost**  
- ✔️ **RoBERTa-Base fine-tuned**  
- ✔️ **RoBERTa-Large fine-tuned (best model)**  

Transformers significantly outperformed classical ML models.

---

# 🎯 Problem Overview

The challenge is to detect **cheating** in writing assessments by distinguishing human vs AI-generated text.

| Label | Meaning |
|-------|---------|
| **0** | Human-written |
| **1** | AI-generated |

---

# 🔧 Models Implemented

---

## 1️⃣ XGBoost Baseline

A fast, interpretable baseline using engineered features:

- Text length  
- Unique word ratio  
- Stopword density  
- Punctuation ratios  
- Character-level patterns  
- TF-IDF vectors  

### **Performance**

Validation Accuracy: ~0.66
Validation F1 Score: ~0.77


---

## 2️⃣ XGBoost + Optuna (AutoML)

Used Optuna to tune:

- n_estimators (100–3000)  
- max_depth (3–10)  
- learning_rate (0.001–0.1)  
- min_child_weight (1–10)  
- subsample (0.5–1.0)  
- colsample_bytree  
- gamma  

### **Best Example Parameters**
```json
{
  "subsample": 1.0,
  "n_estimators": 800,
  "min_child_weight": 1,
  "max_depth": 3,
  "learning_rate": 0.2,
  "gamma": 0,
  "colsample_bytree": 1.0
}
```
Improved Score
Best R²: 0.885

## 3️⃣ RoBERTa-Base Fine-Tuning

Fine-tuned HuggingFace’s:

### **Setup**
- HuggingFace Trainer API  
- AdamW optimizer  
- Linear LR scheduler  
- CrossEntropyLoss  
- GPU acceleration  

### **Results**
RoBERTa-Base shows significantly better contextual understanding than all classical ML models, capturing semantics, tone, and coherence more effectively.

---

## 4️⃣ RoBERTa-Large Fine-Tuning (Best Model)


### **Why It Wins**
- 355M parameters  
- Superior contextual depth  
- Better sensitivity to writing style patterns  
- Stronger coherence and fluency detection  

### **Notes**
- Requires more GPU memory  
- Achieved **best validation accuracy & F1 score**  
- Used for **final leaderboard submissions**

---

## 📊 Model Performance Summary

| Model                | Accuracy | F1 Score | Notes                        |
|----------------------|----------|----------|------------------------------|
| **XGBoost**          | ~0.66    | ~0.77    | Good baseline                |
| **Optuna-Tuned XGBoost** | ↑        | ↑        | Stable & optimized           |
| **RoBERTa-Base**     | Higher   | Higher   | Strong Transformer baseline  |
| **RoBERTa-Large**    | **Best** | **Best** | Final model used             |

---

## 🛠️ Installation

### Install dependencies:
```bash

pip install transformers torch scikit-learn xgboost optuna pandas numpy
```

## ▶️ How to Run
1️⃣ XGBoost + Optuna
- jupyter notebook notebooks/Mercor_AI_Text_Detection.ipynb

2️⃣ RoBERTa-Base
- jupyter notebook notebooks/Robertabase.ipynb

3️⃣ RoBERTa-Large
- jupyter notebook notebooks/Robertalarge.ipynb

## 📁 Outputs
- File	Description
- submission.csv	XGBoost predictions
- submission_roberta.csv	RoBERTa-base predictions




