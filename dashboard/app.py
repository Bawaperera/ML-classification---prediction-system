"""
Streamlit dashboard for the Telco Customer Churn prediction model.
4 pages: Overview, Individual Prediction, Batch Prediction, Model Performance.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# so we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import engineer_features, create_preprocessor, get_feature_names
from src.evaluate import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_threshold_analysis,
    plot_feature_importance,
)

# -------------------------------------------------------------------
# paths (relative to this file)
# -------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH      = os.path.join(MODEL_DIR, "best_model.joblib")
PREP_PATH       = os.path.join(MODEL_DIR, "preprocessor.joblib")
META_PATH       = os.path.join(MODEL_DIR, "model_meta.json")
# notebook 02 saves test.csv with all features + Churn column
TEST_DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "processed", "test.csv")


# -------------------------------------------------------------------
# load everything once and cache it
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    if not os.path.exists(PREP_PATH):
        return None
    return joblib.load(PREP_PATH)


@st.cache_data
def load_meta():
    if not os.path.exists(META_PATH):
        return {"model_name": "Unknown", "optimal_threshold": 0.5, "metrics": {}}
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        return None, None
    df = pd.read_csv(TEST_DATA_PATH)
    # test.csv has all engineered features + Churn column
    y = df["Churn"]
    X = df.drop("Churn", axis=1)
    return X, y


def prep_transform(preprocessor, X):
    """Transform X using the fitted preprocessor.
    Uses feature_names_in_ so the column order always matches what was used at fit time.
    """
    cols = list(preprocessor.feature_names_in_)
    avail = [c for c in cols if c in X.columns]
    return preprocessor.transform(X[avail])


# -------------------------------------------------------------------
# page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon=None,
    layout="wide",
)

model        = load_model()
preprocessor = load_preprocessor()
meta         = load_meta()
threshold    = meta.get("optimal_threshold", 0.5)
model_name   = meta.get("model_name", "Best Model")
metrics      = meta.get("metrics", {})

X_test, y_test = load_test_data()

# -------------------------------------------------------------------
# sidebar navigation
# -------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Individual Prediction", "Batch Prediction", "Model Performance"],
)

if model is None:
    st.warning(
        "Model file not found. Run notebooks 01-04 first to train and save the model."
    )
    st.stop()


# ===================================================================
# PAGE 1: OVERVIEW
# ===================================================================
if page == "Overview":
    st.title("Telco Customer Churn — Prediction Dashboard")
    st.markdown(
        "This dashboard lets you explore the churn model, make individual predictions, "
        "upload a batch of customers, and inspect model performance metrics."
    )
    st.markdown("---")

    # metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model", model_name)
    col2.metric("F1 Score",     f"{metrics.get('f1', 0):.3f}")
    col3.metric("ROC-AUC",      f"{metrics.get('roc_auc', 0):.3f}")
    col4.metric("Precision",    f"{metrics.get('precision', 0):.3f}")
    col5.metric("Recall",       f"{metrics.get('recall', 0):.3f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Distribution (test set)")
        if y_test is not None:
            counts = y_test.value_counts()
            labels = ["No Churn", "Churn"]
            sizes  = [counts.get(0, 0), counts.get(1, 0)]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=["#4a90d9", "#e05252"],
                startangle=90,
            )
            ax.set_title("Test Set Class Distribution")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Test data not found. Run the notebooks first.")

    with col_right:
        st.subheader("Feature Importance (top 15)")
        if hasattr(model, "feature_importances_") and preprocessor is not None:
            if X_test is not None:
                num_cols   = preprocessor.transformers_[0][2]
                cat_cols   = preprocessor.transformers_[1][2]
                feat_names = get_feature_names(preprocessor, num_cols, cat_cols)
                fig, ax = plt.subplots(figsize=(6, 6))
                plot_feature_importance(model, feat_names, top_n=15, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Feature importance not available for this model type.")

    st.markdown("---")
    st.subheader("Threshold Setting")
    st.write(
        f"The model uses an optimal threshold of **{threshold:.2f}** (instead of the "
        f"default 0.50). This was chosen to maximise F1 score on the validation set. "
        f"A lower threshold catches more churners at the cost of more false alarms."
    )


# ===================================================================
# PAGE 2: INDIVIDUAL PREDICTION
# ===================================================================
elif page == "Individual Prediction":
    st.title("Individual Customer Prediction")
    st.markdown(
        "Fill in the customer details below. The model will predict churn probability "
        "for the selected customer."
    )
    st.markdown("---")

    # input form
    with st.form("prediction_form"):
        st.subheader("Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender           = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner          = st.selectbox("Partner", ["Yes", "No"])
            dependents       = st.selectbox("Dependents", ["No", "Yes"])
            tenure           = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup    = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        col4, col5, col6 = st.columns(3)

        with col4:
            contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col5:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method    = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        with col6:
            monthly_charges  = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges    = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                               float(monthly_charges * tenure), step=1.0)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # build a one-row DataFrame matching the raw dataset schema
        raw_row = pd.DataFrame([{
            "gender":            gender,
            "SeniorCitizen":     1 if senior_citizen == "Yes" else 0,
            "Partner":           partner,
            "Dependents":        dependents,
            "tenure":            tenure,
            "PhoneService":      phone_service,
            "MultipleLines":     multiple_lines,
            "InternetService":   internet_service,
            "OnlineSecurity":    online_security,
            "OnlineBackup":      online_backup,
            "DeviceProtection":  device_protection,
            "TechSupport":       tech_support,
            "StreamingTV":       streaming_tv,
            "StreamingMovies":   streaming_movies,
            "Contract":          contract,
            "PaperlessBilling":  paperless_billing,
            "PaymentMethod":     payment_method,
            "MonthlyCharges":    monthly_charges,
            "TotalCharges":      total_charges,
        }])

        # clean SeniorCitizen to match training pipeline
        raw_row["SeniorCitizen"] = raw_row["SeniorCitizen"].map({0: "No", 1: "Yes"})

        # feature engineering + preprocessing
        engineered = engineer_features(raw_row)
        X_proc     = prep_transform(preprocessor, engineered)

        prob   = model.predict_proba(X_proc)[0, 1]
        churn  = int(prob >= threshold)

        # result display
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Prediction Result")
            if churn == 1:
                st.error(f"Prediction: **CHURN** (probability {prob:.1%})")
            else:
                st.success(f"Prediction: **STAY** (probability of churn {prob:.1%})")

            st.write(f"Decision threshold: {threshold:.2f}")

            # simple gauge via progress bar
            st.write("Churn probability:")
            st.progress(float(prob))

        with res_col2:
            st.subheader("Risk Guidance")
            if prob >= 0.6:
                st.warning(
                    "High churn risk. Prioritise this customer for retention outreach, "
                    "plan review, and support follow-up."
                )
            elif prob >= 0.3:
                st.info(
                    "Medium churn risk. Consider proactive engagement and offer review "
                    "to reduce churn likelihood."
                )
            else:
                st.success(
                    "Low churn risk. Continue standard engagement and monitor for changes."
                )


# ===================================================================
# PAGE 3: BATCH PREDICTION
# ===================================================================
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown(
        "Upload a CSV file with multiple customers. The model will predict churn for "
        "each one and let you download the results."
    )
    st.markdown("---")

    st.markdown(
        "The CSV must have the same columns as the original Telco dataset "
        "(excluding `customerID` and `Churn`)."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df_raw)} rows.")
            st.dataframe(df_raw.head(5))

            # fix TotalCharges if it's a string
            if "TotalCharges" in df_raw.columns:
                df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0)

            if "customerID" in df_raw.columns:
                customer_ids = df_raw["customerID"].copy()
                df_raw = df_raw.drop("customerID", axis=1)
            else:
                customer_ids = pd.Series(range(len(df_raw)), name="customerID")

            if "SeniorCitizen" in df_raw.columns and df_raw["SeniorCitizen"].dtype != object:
                df_raw["SeniorCitizen"] = df_raw["SeniorCitizen"].map({0: "No", 1: "Yes"})

            if "Churn" in df_raw.columns:
                df_raw = df_raw.drop("Churn", axis=1)

            engineered = engineer_features(df_raw)
            X_proc     = prep_transform(preprocessor, engineered)

            probs  = model.predict_proba(X_proc)[:, 1]
            preds  = (probs >= threshold).astype(int)

            results = df_raw.copy()
            results.insert(0, "CustomerID",   customer_ids.values)
            results["churn_probability"] = probs.round(4)
            results["prediction"]        = np.where(preds == 1, "Churn", "Stay")
            results["risk_band"]         = pd.cut(
                probs,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
            )

            st.markdown("---")
            st.subheader("Prediction Results")

            # summary stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total customers",  len(results))
            c2.metric("Predicted to churn", int(preds.sum()))
            c3.metric("Churn rate",         f"{preds.mean():.1%}")

            st.dataframe(
                results[["CustomerID", "churn_probability", "prediction", "risk_band"]],
                use_container_width=True,
            )

            # download
            csv_bytes = results.to_csv(index=False).encode()
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ===================================================================
# PAGE 4: MODEL PERFORMANCE
# ===================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    if X_test is None or y_test is None:
        st.warning(
            "Test data not found at `data/processed/test.csv`. "
            "Run notebooks 01-02 to generate the processed data files."
        )
        st.stop()

    # X_test from test.csv already has engineered features — just preprocess
    try:
        X_test_proc = prep_transform(preprocessor, X_test)
        num_cols    = preprocessor.transformers_[0][2]
        cat_cols    = preprocessor.transformers_[1][2]
        feat_names  = get_feature_names(preprocessor, num_cols, cat_cols)

        y_prob = model.predict_proba(X_test_proc)[:, 1]
        y_pred_default = (y_prob >= 0.5).astype(int)
        y_pred_optimal = (y_prob >= threshold).astype(int)

    except Exception as e:
        st.error(f"Could not process test data: {e}")
        st.stop()

    # confusion matrices side by side
    st.subheader("Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.write(f"Default threshold (0.50)")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_confusion_matrix(y_test, y_pred_default, model_name=f"{model_name} (0.50)", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with cm_col2:
        st.write(f"Optimal threshold ({threshold:.2f})")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_confusion_matrix(y_test, y_pred_optimal, model_name=f"{model_name} ({threshold:.2f})", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ROC and PR curves
    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_roc_curves({model_name: model}, X_test_proc, y_test, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with curve_col2:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_pr_curves({model_name: model}, X_test_proc, y_test, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # threshold analysis
    st.subheader("Threshold Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    _, opt_t = plot_threshold_analysis(y_test, y_prob, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance (top 20)")
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_feature_importance(model, feat_names, top_n=20, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # model card
    st.subheader("Model Card")
    card_data = {
        "Model":             model_name,
        "Optimal Threshold": f"{threshold:.2f}",
        "F1 Score":          f"{metrics.get('f1', 'N/A')}",
        "ROC-AUC":           f"{metrics.get('roc_auc', 'N/A')}",
        "Precision":         f"{metrics.get('precision', 'N/A')}",
        "Recall":            f"{metrics.get('recall', 'N/A')}",
        "Accuracy":          f"{metrics.get('accuracy', 'N/A')}",
        "Dataset":           "Telco Customer Churn (Kaggle)",
        "Target":            "Churn (binary: 0 = Stay, 1 = Churn)",
        "Class Imbalance":   "~73% No Churn / ~27% Churn",
        "Imbalance Strategy":"SMOTE on training data only",
        "Preprocessing":     "StandardScaler + OneHotEncoder (sklearn Pipeline)",
        "Feature Count":     str(X_test_proc.shape[1]),
        "Training Rows":     "~5,634 (after SMOTE resampling)",
    }

    card_df = pd.DataFrame(list(card_data.items()), columns=["Property", "Value"])
    st.table(card_df)
