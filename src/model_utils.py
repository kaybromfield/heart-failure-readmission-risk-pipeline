# ============================================
# model_utils.py
# ============================================
# Shared logic for loading trained models from S3 and scoring patient data.
# Used by both batch_scoring.py (scheduled/batch automation) and api.py
# (real-time FastAPI endpoint), so preprocessing logic only lives in one
# place and the two paths can never drift out of sync with each other.
# ============================================

import io
import logging

import boto3
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.iolib.smpickle as smpickle

log = logging.getLogger("model_utils")

# ============================================
# CONFIG
# ============================================
BUCKET = "kb-mimic-data"
MODEL_DATA_PREFIX = "kb-mimic-model-data"

MODELING_DATA_KEY = f"{MODEL_DATA_PREFIX}/modeling_data/final_dataset_cleaned.csv"
XGB_MODEL_KEY = f"{MODEL_DATA_PREFIX}/trained-models/xgboost_model.pkl"
LOGIT_MODEL_KEY = f"{MODEL_DATA_PREFIX}/trained-models/logistic_regression_model.pickle"
COX_MODEL_KEY = f"{MODEL_DATA_PREFIX}/trained-models/cox_model.pkl"
FEATURE_COLUMNS_KEY = f"{MODEL_DATA_PREFIX}/trained-models/model_feature_columns.pkl"

TARGET_COL = "readmitted_30_days"
CATEGORICAL_COLS = ["gender", "insurance_simplified"]
DROP_COLS = ["subject_id", "hadm_id", "admittime", "dischtime", "race_simplified", "total_lab_rows"]

COX_VARS = [
    "length_of_stay_days",
    "married_flag",
    "insurance_private",
    "valvular_hd_flag",
    "ckd_flag",
    "diabetes_flag",
    "depression_flag",
    "cancer_flag",
    "lab_abnormal_ratio",
    "avg_creatinine",
    "avg_sodium",
    "avg_hemoglobin",
]

s3 = boto3.client("s3")


# ============================================
# S3 I/O HELPERS
# ============================================
def load_csv_from_s3(bucket, key):
    log.info(f"Loading CSV from s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    log.info(f"Loaded dataframe with shape {df.shape}")
    return df


def load_pickle_from_s3(bucket, key):
    log.info(f"Loading model artifact from s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(io.BytesIO(obj["Body"].read()))


def load_logit_from_s3(bucket, key):
    log.info(f"Loading logistic regression model from s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return smpickle.load_pickle(io.BytesIO(obj["Body"].read()))


def upload_df_to_s3(df, bucket, key):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    log.info(f"Uploaded results to s3://{bucket}/{key}")


class ModelBundle:
    """Holds all four trained model artifacts, loaded once and reused
    across requests/batches instead of reloading from S3 every call."""

    def __init__(self):
        self.trained_columns = load_pickle_from_s3(BUCKET, FEATURE_COLUMNS_KEY)
        self.xgb_model = load_pickle_from_s3(BUCKET, XGB_MODEL_KEY)
        self.cph = load_pickle_from_s3(BUCKET, COX_MODEL_KEY)
        self.logit_result = load_logit_from_s3(BUCKET, LOGIT_MODEL_KEY)
        log.info("All model artifacts loaded successfully")


# ============================================
# PREPROCESSING (mirrors log_xg.py training pipeline)
# ============================================
def preprocess_batch(df_raw, trained_columns):
    df = df_raw.copy()

    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)

    y_true = None
    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL]
        df = df.drop(columns=[TARGET_COL])

    df = df.apply(pd.to_numeric, errors="coerce")

    # CRITICAL STEP: reindex to match exact training column structure
    df = df.reindex(columns=trained_columns, fill_value=0)
    df = df.astype(float)

    return df, y_true


# ============================================
# SCORING FUNCTIONS
# ============================================
def score_xgboost(xgb_model, X):
    return xgb_model.predict_proba(X)[:, 1]


def score_logistic(logit_result, X):
    X_sm = sm.add_constant(X, has_constant="add")
    return logit_result.predict(X_sm)


def score_cox(cph, df_raw_sample):
    """
    Returns estimated median survival time (days) per patient where available.
    Patients whose predicted survival curve never crosses 0.5 probability
    are reported as inf (median survival not reached within observed follow-up).
    """
    df = df_raw_sample.copy()

    # final_dataset_cleaned.csv uses different column names than
    # final_dataset_cox_visual.csv (what the Cox model was trained on).
    df = df.rename(columns={
        "insurance_Private": "insurance_private",
        "abnormal_lab_ratio": "lab_abnormal_ratio",
    })

    available_vars = [c for c in COX_VARS if c in df.columns]
    missing = set(COX_VARS) - set(available_vars)
    if missing:
        log.warning(f"Cox model missing expected columns in input data: {missing}")

    cox_input = df[available_vars].apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        median_survival = cph.predict_median(cox_input)
    except Exception as e:
        log.warning(f"Cox median survival prediction failed: {e}")
        median_survival = pd.Series([np.nan] * len(cox_input), index=cox_input.index)

    return median_survival


def score_patients(models: ModelBundle, df_raw):
    """
    Full scoring pipeline for a dataframe of one or more patients.
    Returns a results dataframe with risk scores, tier, and predicted
    time-to-readmission for each patient.
    """
    X, y_true = preprocess_batch(df_raw, models.trained_columns)

    xgb_probs = score_xgboost(models.xgb_model, X)
    logit_probs = score_logistic(models.logit_result, X)
    median_survival_days = score_cox(models.cph, df_raw)

    results = pd.DataFrame({
        "patient_index": df_raw.index,
        "xgboost_risk_score": xgb_probs,
        "logistic_risk_score": logit_probs,
        "predicted_median_days_to_readmission": median_survival_days.values
        if hasattr(median_survival_days, "values") else median_survival_days,
    })

    results["risk_tier"] = pd.cut(
        results["xgboost_risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )

    if y_true is not None:
        results["actual_readmitted_30_days"] = y_true.values

    return results
