
# 💳 AI-Powered Explainable Fraud Detection System for Banking

# 📌 Project Overview

This project implements an AI-powered fraud detection system for banking transactions using machine learning and explainable AI (XAI) techniques.
The system analyzes transaction data to classify transactions as legitimate or fraudulent, while also providing clear explanations for model decisions.

The project focuses on handling highly imbalanced data, achieving strong predictive performance, and ensuring model transparency through SHAP-based explanations.

# 🧠 Problem Statement
The project aims to detect fraudulent banking transactions with high accuracy while providing explainable insights into model decisions using Explainable AI (XAI), ensuring transparency and trust in fraud detection systems.

# 🎯 Objectives
Detect fraudulent transactions with high accuracy
Handle extreme class imbalance in financial data
Build a robust ML pipeline using XGBoost
Provide explainable predictions using SHAP
Evaluate model performance using multiple metrics and visualizations


# 📊 Dataset Description

Dataset: Credit Card Transactions Dataset
Total records: ~284,000+
Features:
V1 to V28 (anonymized transaction features)
Amount (transaction amount)
Target:
Class = 0 → Legitimate transaction

Class = 1 → Fraudulent transaction

# ⚙️ Data Preprocessing

Removed non-informative features (e.g., Time)
Scaled transaction amount using StandardScaler
Split data using stratified train-test split
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance classes
Class Distribution

Before SMOTE:
Legitimate: 227,451
Fraudulent: 394

After SMOTE:
Legitimate: 227,451
Fraudulent: 227,451

# 🧠 Model Used
XGBoost Classifier

The project uses an XGBoost (Extreme Gradient Boosting) classifier due to its:
High performance on tabular data
Ability to handle non-linear relationships
Robustness to noise
Compatibility with explainability tools

Key Parameters
n_estimators = 100
max_depth = 6
learning_rate = 0.1
eval_metric = AUC
random_state = 42

# 📈 Model Evaluation and Accuracy
🔹 Performance Metrics
Accuracy: ~99.9%
ROC-AUC: ~0.97
Precision, Recall, F1-score: Strong balance between fraud detection and false positives

🔹 Cross-Validation

Used Stratified K-Fold (5 folds)
Mean cross-validation accuracy: ~99.5%
Confirms model stability across different splits

# 📊 Visualizations Included

Feature distribution plots (Fraud vs Legitimate)
Class distribution before and after SMOTE
Confusion Matrix
ROC Curve
Precision–Recall Curve
K-Fold accuracy comparison
Feature importance using SHAP

# 🔍 Explainable AI (XAI) with SHAP

To ensure transparency, SHAP (SHapley Additive exPlanations) is used to explain model predictions.
SHAP Capabilities
Identifies most influential features (e.g., V14, V12, V17, V4)
Explains why a transaction is predicted as fraudulent
Provides both global and instance-level explanations
This makes the model suitable for real-world banking environments, where explainability is critical.


# 🛠️ Technologies Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Matplotlib, Seaborn,  SHAP, Joblib, Google Colab

# 📌 My Conclusion
Contributed to data preprocessing and cleaning for an explainable credit card fraud detection model by handling missing values, removing inconsistencies, and performing outlier analysis on transactional data. Applied feature scaling and addressed class imbalance using SMOTE, while engineering meaningful features to support accurate, stable, and interpretable fraud prediction.
