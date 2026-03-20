# Model Card — Telco Customer Churn Prediction

## Model Details

| Property | Value |
|---|---|
| Model type | XGBoost (XGBClassifier) |
| Version | 1.0 |
| Training date | 2024 |
| Author | Portfolio project |
| Framework | scikit-learn 1.4, XGBoost 2.0 |

## Intended Use

Predict which telecom customers are likely to cancel their subscription in the near future so that the retention team can intervene proactively.

**Primary users:** Customer retention team, business analysts.

**Out-of-scope uses:** Credit scoring, insurance risk, any high-stakes decision where errors have severe personal consequences.

## Dataset

- **Source:** IBM Telco Customer Churn dataset (publicly available on Kaggle)
- **Size:** 7,043 rows, 20 features
- **Target:** `Churn` — binary (0 = stays, 1 = churns)
- **Class balance:** ~73% No Churn / ~27% Churn (imbalanced)
- **Train / test split:** 80 / 20, stratified on target

## Training Procedure

1. Load raw CSV, fix `TotalCharges` string issue, drop `customerID`.
2. Engineer 10 new features (tenure groups, service bundles, contract risk score, etc.).
3. Fit a `ColumnTransformer` pipeline on training data only:
   - `StandardScaler` for numerical columns
   - `OneHotEncoder` for categorical columns
4. Apply SMOTE to the training set only (oversamples the minority class).
5. Train 6 baseline models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM).
6. Select top 2 by F1 score, run `RandomizedSearchCV` with `StratifiedKFold (k=5)`.
7. Tune classification threshold by maximising F1 on validation predictions.

## Performance

Evaluated on the held-out test set (no SMOTE applied to test data).

| Metric | Default threshold (0.50) | Optimal threshold |
|---|---|---|
| F1 | ~0.61 | ~0.65 |
| ROC-AUC | ~0.84 | ~0.84 |
| Precision | ~0.67 | ~0.58 |
| Recall | ~0.56 | ~0.74 |

Exact numbers depend on the random seed and hyperparameter search results in your run.

## Key Features

Top drivers of churn predictions from model analysis:

1. Contract type (Month-to-month contracts strongly predict churn)
2. Tenure (newer customers churn more)
3. Monthly charges (higher bills increase churn risk)
4. Online security (no security = higher risk)
5. Tech support (no support = higher risk)
6. Internet service type (Fiber optic customers churn more)

## Ethical Considerations

- The dataset comes from a fictional IBM scenario and does not contain real customer data.
- Features like `gender` and `SeniorCitizen` are included because they were in the original dataset. In a real deployment, it is worth auditing whether these features introduce bias.
- The model is not intended for any consequential decisions beyond targeted retention outreach.

## Limitations

- Trained on a single snapshot — customer behaviour patterns change over time (concept drift). The model should be retrained periodically.
- SMOTE was used to handle class imbalance. Real-world performance on a different customer base may differ from the reported metrics.
- SVM was included for comparison but is slow on larger datasets.

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run notebooks in order (01 to 06)
jupyter notebook

# launch dashboard (after running notebooks 01-04)
cd dashboard
streamlit run app.py
```
