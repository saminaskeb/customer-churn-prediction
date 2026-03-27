# Customer Churn Prediction

A machine learning pipeline to predict customer churn for a telecommunications company using XGBoost, LightGBM, and Logistic Regression with Bayesian hyperparameter optimization.

---

## Problem Statement

Customer churn — the loss of clients or subscribers — is one of the most costly problems in subscription-based businesses. In the telecom industry, acquiring a new customer can cost 5–25x more than retaining an existing one. Early identification of at-risk customers allows businesses to intervene proactively with targeted retention offers, reducing revenue loss and improving customer lifetime value. This project builds an end-to-end churn prediction system optimized for high recall, ensuring that churners are identified before they leave.

---

## Dataset Overview

- **Source:** IBM Telco Customer Churn dataset ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (binary — 1 = churned, 0 = retained)
- **Class imbalance:** ~26.5% churn rate (~2.8:1 ratio)
- **Features:** Customer demographics (gender, senior citizen, dependents), account details (tenure, contract type, monthly charges, total charges), and subscribed services (phone, internet, security, streaming, etc.)
- **Engineered features:** 6 domain-driven interaction features including `charges_per_tenure`, `is_new_customer`, `no_support_services`, `high_value_at_risk`, `total_services`, and `fiber_no_security`

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Imbalanced learning | `imbalanced-learn` (SMOTE, SMOTEENN, ADASYN) |
| Hyperparameter tuning | `optuna` |
| Explainability | `shap` |
| Model persistence | `joblib` |
| Dataset download | `kagglehub` |

---

## Project Workflow

```
1. Data Cleaning
   ├── Download dataset via kagglehub
   ├── Handle 11 missing values in TotalCharges (new customers, tenure = 0)
   └── Convert target variable to binary (Yes/No → 1/0)

2. Feature Engineering
   ├── charges_per_tenure: cost efficiency signal
   ├── is_new_customer: tenure ≤ 6 months flag
   ├── no_support_services: no TechSupport + no OnlineSecurity + no OnlineBackup
   ├── high_value_at_risk: MonthlyCharges > $70 on month-to-month contract
   ├── total_services: count of active subscribed services (0–6)
   └── fiber_no_security: fiber optic internet without online security

3. Exploratory Data Analysis (EDA)
   ├── Churn distribution and class imbalance analysis
   ├── Feature correlations and churn rate by category
   └── Visualizations of key churn drivers

4. Preprocessing Pipeline
   ├── OneHotEncoder for 15 categorical features
   ├── StandardScaler for 10 numeric features
   └── ColumnTransformer (fit on train, applied to val/test — no leakage)

5. Modeling & Hyperparameter Tuning
   ├── Baseline: Logistic Regression, XGBoost, LightGBM (5-fold Stratified CV)
   ├── Optuna Bayesian search: 60 trials × 3 models = 180 total trials
   ├── Resampling strategies: SMOTE, SMOTE-ENN, ADASYN (applied inside CV folds)
   └── Stacking ensemble evaluated but rejected (marginal AUC gain, recall drop)

6. Model Evaluation
   ├── Threshold optimization: F2-score optimized at 0.136 (recall-prioritized)
   ├── Probability calibration: isotonic regression on validation set
   ├── SHAP feature importance analysis
   └── Fairness audit: recall and FPR gaps across gender, age, and contract type
```

---

## Key Results

**Final Model:** XGBoost (Optuna-tuned) + SMOTE-ENN + Isotonic Calibration

| Metric | Score |
|---|---|
| ROC-AUC | 0.839 |
| Recall | **0.896** |
| Precision | 0.438 |
| F1-Score | 0.589 |
| F2-Score | 0.741 |
| Accuracy | 0.668 |
| Average Precision | 0.633 |
| Brier Score | 0.139 |

**Design priority:** Recall was maximized (target ≥ 0.82, achieved 0.896) over accuracy, reflecting the real-world cost asymmetry where missing a churner is more expensive than a false positive.

**Fairness audit highlights:**
- Gender recall gap: 0.020 (acceptable)
- Senior citizen recall gap: 0.114 (flagged for monitoring)

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Open the notebook**
```bash
jupyter notebook notebook/customer_churn_prediction.ipynb
```

> The dataset is downloaded automatically via `kagglehub`. You will need a Kaggle account and API credentials configured (`~/.kaggle/kaggle.json`).

---

## Future Improvements

- **Target AUC ≥ 0.85:** Explore additional feature interactions and neural network-based models (TabNet, NODE)
- **Reduce senior citizen fairness gap:** Apply fairness-aware learning constraints or post-processing calibration per subgroup
- **Real-time scoring API:** Wrap the saved model artifacts into a FastAPI or Flask endpoint for production inference
- **Monitoring dashboard:** Build a Streamlit app to visualize churn risk scores and SHAP explanations per customer
- **Expanded dataset:** Enrich with behavioral signals (support tickets, usage trends) to improve signal quality

---

## Author

**Saminas Kebebe**

Data science student building end-to-end ML pipelines with a focus on interpretable, business-driven modeling.
