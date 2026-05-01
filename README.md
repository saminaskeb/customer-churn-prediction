# Customer Churn Prediction

---

## 1. Business Problem / Motivation

Customer churn — the loss of clients or subscribers — is one of the most costly problems in subscription-based businesses. In the telecom industry, acquiring a new customer can cost 5–25x more than retaining an existing one. Early identification of at-risk customers allows businesses to intervene proactively with targeted retention offers, reducing revenue loss and improving customer lifetime value.

This project addresses that problem head-on: build a production-ready churn prediction system that catches as many churners as possible before they leave, while remaining explainable to business stakeholders.

The cost asymmetry between false negatives (missing a churner) and false positives (wrongly flagging a loyal customer) was quantified at a **5:1 ratio**, meaning missing one churner costs the business five times more than a wasted retention offer. This cost structure directly drove every modeling decision in this project.

---

## 2. Project Overview

**Goal:** Predict which telecom customers will churn within the next billing period with high recall, enabling proactive retention campaigns.

**Approach:**
- Conduct exploratory data analysis to identify the strongest churn signals
- Engineer 6 domain-driven features to capture interaction effects not present in raw data
- Compare Logistic Regression, XGBoost, and LightGBM baselines using 5-fold stratified cross-validation
- Run Bayesian hyperparameter optimization (Optuna) across 60 trials per model
- Address class imbalance (~26% churn) using SMOTE-ENN resampling
- Calibrate predicted probabilities using isotonic regression
- Optimize the decision threshold using F2-score to reflect the 5:1 FN/FP cost ratio
- Audit model fairness across gender, age, and contract type subgroups
- Explain predictions using SHAP values for both global and individual-level interpretability

**Key Results:**

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.839 |
| Recall | **0.896** |
| Precision | 0.438 |
| F1-Score | 0.589 |
| **F2-Score** | **0.741** |
| Threshold | 0.143 |

The final model is an **XGBoost classifier** (Optuna-tuned) wrapped in a SMOTE-ENN resampling pipeline and isotonic probability calibration. It achieves a recall of 0.896, meaning it catches nearly 9 in 10 churners before they leave.

---

## 3. Data

- **Source:** IBM Telco Customer Churn dataset ([Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Downloaded via:** `kagglehub` (automatic download inside the notebook — requires a Kaggle API key)
- **Type:** Structured tabular data (CSV)
- **Size:** 7,043 customers × 21 features
- **Target variable:** `Churn` (binary — 1 = churned, 0 = retained)
- **Class imbalance:** ~26.5% churn rate (~2.8:1 majority-to-minority ratio)

**Key features:**

| Category | Features |
|----------|----------|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Charges | `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 4. Data Preprocessing

**Cleaning:**
- Converted `TotalCharges` from object to float (contained whitespace strings for 11 new customers with `tenure = 0`)
- Imputed those 11 missing `TotalCharges` values with `0` (business logic: no charges yet)
- Converted target variable from `"Yes"/"No"` strings to binary `1/0`

**Train / Validation / Test Split:**
- Stratified split: 65% train / 15% validation / 20% test
- All preprocessing (encoding, scaling) fit exclusively on training data to prevent data leakage
- Validation set used for threshold optimization and probability calibration only
- Test set locked and evaluated once at the end

**Feature Engineering (6 domain-driven features):**

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `charges_per_tenure` | `MonthlyCharges / (tenure + 1)` | Identifies high-cost new customers — a strong churn signal |
| `is_new_customer` | `tenure ≤ 6 months` | New customers churn at much higher rates |
| `no_support_services` | No TechSupport AND no OnlineSecurity AND no OnlineBackup | Unprotected customers are more likely to leave after a bad experience |
| `high_value_at_risk` | `MonthlyCharges > $70` AND month-to-month contract | High-paying customers with no commitment — highest retention priority |
| `total_services` | Count of active subscribed services (0–6) | More services → higher switching costs → lower churn |
| `fiber_no_security` | Fiber optic internet without online security | Fiber customers may leave due to service quality if unprotected |

**Preprocessing Pipeline:**
- `OneHotEncoder` applied to 15 categorical features
- `StandardScaler` applied to 10 numeric features
- Combined via `ColumnTransformer`

---

## 5. Exploratory Data Analysis

[TO BE UPDATED — visualization screenshots to be added to `images/`]

**Visualizations produced in the notebook:**

1. **Churn distribution bar chart** — Confirms the ~26.5% class imbalance; sets the stage for why recall-optimized modeling is necessary
2. **Churn rate by contract type** — Month-to-month customers churn at dramatically higher rates than one- or two-year contract holders, making `Contract` one of the strongest predictors
3. **Monthly charges distribution by churn** — Churners skew toward higher monthly charges, particularly Fiber Optic users without bundled services
4. **Correlation heatmap / feature-churn rate bar chart** — Highlights `tenure`, `Contract`, `InternetService`, and `TechSupport` as the most correlated features with churn

**Key insights from EDA:**
- Short-tenure customers (< 12 months) represent a disproportionate share of churners
- Customers with no online security or tech support churn significantly more
- Fiber optic customers have the highest churn rate among internet service types
- Month-to-month contracts are the single strongest categorical churn predictor

---

## 6. Modeling Approach

**Baseline models (5-fold stratified cross-validation):**

| Model | CV ROC-AUC |
|-------|-----------|
| Logistic Regression | 0.845 |
| XGBoost | ~0.840 |
| LightGBM | ~0.838 |

Logistic Regression achieved the best baseline AUC, demonstrating that the feature set is largely linearly separable. However, XGBoost was selected for further tuning due to its superior capacity to model feature interactions and its stronger recall after threshold optimization.

**Advanced model — reasoning:**

XGBoost was chosen as the final model because:
- It handles the tabular feature mix (numeric + one-hot encoded categoricals) natively
- Tree-based splits naturally capture the interaction between `Contract`, `tenure`, and `MonthlyCharges`
- It responds well to SMOTE-ENN resampling within the training pipeline
- SHAP values are natively supported, enabling full model explainability

**Stacking ensemble (evaluated but rejected):**
A stacking ensemble (LR + XGBoost + LightGBM with a meta-Logistic Regression) was evaluated. It produced marginal AUC improvement but caused a notable drop in recall. Given the 5:1 FN/FP cost ratio, recall preservation was non-negotiable, and the ensemble was dropped in favor of the single tuned XGBoost.

---

## 7. Model Training

**Tools:**
- `XGBClassifier` (XGBoost)
- `ImbPipeline` (imbalanced-learn) — chains SMOTE-ENN resampling with the classifier inside CV folds to prevent resampling leakage
- `Optuna` — Bayesian hyperparameter optimization (Tree-structured Parzen Estimator)
- `CalibratedClassifierCV` with `method='isotonic'` — fits on the validation set post-training

**Hyperparameter search space (Optuna, 60 trials):**

| Hyperparameter | Search Range |
|----------------|-------------|
| `n_estimators` | 100 – 600 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.01 – 0.3 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `min_child_weight` | 1 – 10 |
| `gamma` | 0 – 5 |
| `reg_alpha` | 1e-8 – 10 (log scale) |
| `reg_lambda` | 1e-8 – 10 (log scale) |

**Training process:**
1. Optuna runs 60 trials, each evaluated via 5-fold stratified CV with ROC-AUC as the objective
2. SMOTE-ENN is applied inside each fold (fit on fold train, not applied to fold validation)
3. Best hyperparameters are used to retrain on the full training set
4. Isotonic calibration is fit on the held-out validation set
5. Decision threshold optimized on the validation set by maximizing F2-score (beta=2), reflecting 5:1 FN/FP cost ratio
6. Final threshold: **0.143**

---

## 8. Results

**Why these metrics?**

- **ROC-AUC:** Measures overall discrimination ability across all thresholds — threshold-independent
- **Recall:** The primary business metric — proportion of actual churners correctly identified. A missed churner (false negative) costs 5x more than a false alarm
- **F2-Score:** Weighs recall twice as heavily as precision, matching the 5:1 cost ratio; used for threshold selection
- **Precision:** Reported for completeness — represents the fraction of flagged customers who actually churn
- **Brier Score:** Measures calibration quality of predicted probabilities

**Final model performance (held-out test set):**

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.839 |
| Recall | **0.896** |
| Precision | 0.438 |
| F1-Score | 0.589 |
| **F2-Score** | **0.741** |
| Accuracy | 0.668 |
| Average Precision | 0.633 |
| Brier Score | 0.139 |
| Decision Threshold | 0.143 |

**Model comparison table:**

| Model | CV AUC | Recall (val) | Notes |
|-------|--------|-------------|-------|
| Logistic Regression (baseline) | 0.845 | — | Strong baseline, linear |
| XGBoost (baseline) | ~0.840 | — | Best for tuning |
| LightGBM (baseline) | ~0.838 | — | Competitive baseline |
| XGBoost + Optuna | 0.8477 | — | Best tuned AUC |
| XGBoost + Optuna + SMOTE-ENN | — | High | Selected for recall |
| **XGBoost + Optuna + SMOTE-ENN + Calibration** | **0.839** | **0.896** | **Final model** |
| Stacking Ensemble | ~0.850 | Lower | Rejected — recall drop |

**Visualizations:** [TO BE UPDATED — add ROC curve, Precision-Recall curve, Confusion Matrix, Calibration plot to `images/`]

---

## 9. Model Interpretation

**SHAP (SHapley Additive exPlanations) analysis:**

SHAP values were computed for the final calibrated XGBoost model to explain both global feature importance and individual predictions.

**Top global predictors (SHAP bar chart):**

[TO BE UPDATED — add SHAP bar chart to `images/`]

From the SHAP beeswarm plot, the strongest drivers of churn prediction are:
1. **`tenure`** — Short tenure strongly pushes predictions toward churn; long-tenured customers are reliably retained
2. **`Contract` (month-to-month)** — The single strongest categorical predictor; month-to-month contract customers have much higher churn probability
3. **`charges_per_tenure`** (engineered) — High cost-per-tenure signals new customers paying a lot relative to their engagement
4. **`InternetService` (Fiber Optic)** — Fiber customers without security services churn at elevated rates
5. **`MonthlyCharges`** — High charges, especially without bundled services or long-term contracts, increase churn probability
6. **`TechSupport` / `OnlineSecurity`** (No) — Absence of support services is a consistent churn driver

**Local explanations (individual predictions):**

The notebook includes SHAP waterfall plots for three representative cases:
- A true positive (correctly flagged churner)
- A false negative (missed churner — high-tenure, low-charge)
- A false positive (incorrectly flagged loyal customer)

These explain exactly which features pushed the model above or below the decision threshold for each individual.

**Error analysis — false negative profile:**

Missed churners (false negatives) tend to be long-tenure, lower-charge customers on month-to-month contracts who churn for non-price reasons (e.g., service quality, life events). These are the hardest cases to capture with tabular features alone, and represent an opportunity for behavioral signal enrichment.

---

## 10. Key Insights

**What worked best:**
- **SMOTE-ENN** outperformed standard SMOTE and ADASYN for recall without excessive precision loss; the cleaning step in SMOTE-ENN removes borderline noisy samples that confuse the classifier
- **Threshold optimization at 0.143** (rather than default 0.5) was the single biggest lever for recall improvement — dropping the threshold from 0.5 to 0.143 increased recall from ~0.6 to 0.896
- **Feature engineering** contributed meaningfully: `charges_per_tenure`, `high_value_at_risk`, and `total_services` all appeared in the top SHAP features
- **Isotonic calibration** improved probability reliability (Brier score), which matters for business applications where predicted probability is used to prioritize outreach budget

**Business / practical impact:**
- At **0.896 recall**, the model catches approximately 9 in 10 churners before they leave
- With a 5:1 cost ratio (FN vs FP), the optimized threshold is expected to produce positive ROI on retention campaigns even at 43.8% precision
- The `high_value_at_risk` flag (high charges + month-to-month contract) provides an immediately actionable customer segment for targeted offers
- **Fairness note:** The model shows a 0.114 recall gap for senior citizens vs. non-seniors — this subgroup should receive additional monitoring and potentially a separate retention strategy

---

## 11. Conclusion

This project delivers a production-ready customer churn prediction pipeline built around a clearly quantified business cost structure. By framing the problem as a 5:1 false-negative-to-false-positive cost ratio and optimizing the decision threshold accordingly, the final XGBoost model achieves 0.896 recall — well above the 0.82 target — while maintaining an ROC-AUC of 0.839.

The combination of Optuna-tuned XGBoost, SMOTE-ENN resampling, isotonic calibration, and F2-optimized thresholding produced a model that is both performant and explainable. SHAP analysis confirms that the model's decisions align with domain intuition: tenure, contract type, and support service absence are the primary drivers of churn.

The model artifacts (`churn_model_v3_calibrated.pkl`, `churn_preprocessor_v3.pkl`, `churn_model_config_v3.json`) are export-ready for deployment into a scoring API or batch inference pipeline.

---

## 12. Future Work

- **Target AUC ≥ 0.85:** Explore additional feature interactions, polynomial features, and neural network-based tabular models (TabNet, NODE)
- **Reduce senior citizen fairness gap (0.114):** Apply fairness-aware learning constraints or post-processing calibration per subgroup; consider a subgroup-specific threshold
- **Real-time scoring API:** Wrap the saved model artifacts into a FastAPI endpoint for production inference; expose `/predict` and `/explain` routes
- **Monitoring dashboard:** Build a Streamlit app to visualize churn risk scores, SHAP explanations per customer, and model drift indicators
- **Behavioral signal enrichment:** Add support ticket frequency, usage trend features, and NPS scores to capture non-price churn drivers that the current model misses
- **Temporal validation:** Evaluate model performance on a time-ordered holdout (train on months 1–18, test on months 19–24) to simulate real deployment conditions

---

## 13. How to Run

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

**4. Configure Kaggle API credentials**

The dataset is downloaded automatically via `kagglehub`. You will need a Kaggle account and API token saved at `~/.kaggle/kaggle.json`. See [Kaggle API setup instructions](https://www.kaggle.com/docs/api).

**5. Open and run the notebook**
```bash
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

Run all cells in order. The notebook is self-contained: it downloads the data, runs all preprocessing, trains and evaluates all models, and exports the final artifacts.

**Output artifacts (written to `models/` and `results/` after running):**
- `models/churn_model_v3_calibrated.pkl` — Calibrated XGBoost model
- `models/churn_preprocessor_v3.pkl` — Fitted preprocessing pipeline
- `models/churn_model_config_v3.json` — Model config and threshold
- `results/churn_risk_table_v3.csv` — Risk scores for all test customers

---

## 14. Repository Structure

```
customer-churn-prediction/
│
├── notebooks/
│   └── customer_churn_prediction.ipynb   # Full ML pipeline (EDA → training → evaluation → SHAP)
│
├── data/                                  # Raw and processed data (populated at runtime)
│
├── models/                                # Saved model artifacts (populated at runtime)
│
├── results/                               # Output predictions and evaluation reports
│
├── images/                                # Saved plots and visualizations
│
├── requirements.txt                       # Python package dependencies
├── .gitignore                             # Excludes data, models, checkpoints, secrets
└── README.md                              # This file
```

---

## 15. Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
optuna
shap
joblib
kagglehub
jupyter
```

---

*Author: Saminas Kebebe — Data science student building end-to-end ML pipelines with a focus on interpretable, business-driven modeling.*
