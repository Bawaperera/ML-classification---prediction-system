# ML Classification & Prediction System
## Telco Customer Churn Prediction

A complete end-to-end machine learning project that predicts which telecom customers are likely to cancel their subscription. The project covers data cleaning, feature engineering, model comparison, hyperparameter tuning, threshold optimisation, and a Streamlit dashboard.

---

## Project Structure

```
ML classification & prediction system/
├── data/
│   ├── raw/                   # original Telco CSV (not committed)
│   └── processed/             # cleaned splits saved by notebooks
├── models/
│   ├── baseline_models/       # 6 .joblib files from notebook 03
│   ├── tuned_models/          # tuned XGBoost and RF
│   ├── best_model.joblib      # best model used by dashboard
│   ├── preprocessor.joblib    # fitted ColumnTransformer
│   └── model_meta.json        # model name, threshold, metrics
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation_deep_dive.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # load, clean, split
│   ├── features.py            # feature engineering + sklearn pipeline
│   ├── train.py               # ModelTrainer class (6 models)
│   └── evaluate.py            # plotting utilities for evaluation
├── dashboard/
│   └── app.py                 # 4-page Streamlit app
├── reports/
│   └── model_card.md          # model details, metrics, limitations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

IBM Telco Customer Churn dataset — 7,043 rows, 20 features, binary target (churned or stayed). Available on Kaggle. Place the raw CSV at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebooks in order

Open Jupyter and run the notebooks from 01 to 05. Each notebook saves its outputs (processed data, trained models) so the next one can pick up where the last left off.

```bash
jupyter notebook
```

| Notebook | What it does |
|---|---|
| 01 | EDA — understand the data, check distributions, find patterns |
| 02 | Feature engineering — create 10 new features, apply SMOTE |
| 03 | Train 6 models, compare with a table and charts |
| 04 | Tune the top 2 models with RandomizedSearchCV, pick best |
| 05 | Deep evaluation — confusion matrix, ROC/PR, learning curves, error analysis |

### 3. Launch the dashboard

After running notebooks 01–04 (so the model files exist):

```bash
cd dashboard
streamlit run app.py
```

The dashboard has 4 pages:
- **Overview** — metric cards, feature importance, churn distribution
- **Individual Prediction** — fill in customer details, get churn probability
- **Batch Prediction** — upload a CSV, download predictions
- **Model Performance** — confusion matrix, ROC/PR curves, threshold analysis

---

## Key Results

- Best model: **XGBoost** with tuned hyperparameters
- ROC-AUC: **~0.84**
- F1 Score: **~0.65** (at optimal threshold)
- Optimal threshold: **below 0.50** — catches more churners at the cost of more false alarms, which makes business sense (retention outreach is cheaper than losing a customer)

Top churn drivers:
- Month-to-month contract
- Low tenure (new customers)
- High monthly charges
- No online security or tech support
- Fiber optic internet service

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit · matplotlib · seaborn

---

## What I Learned

- Why accuracy is a bad metric for imbalanced datasets and why F1 + ROC-AUC are better
- How to prevent data leakage by fitting the preprocessing pipeline on training data only
- How SMOTE works and why it should never touch the test set
- How threshold tuning can improve real-world usefulness of a model
