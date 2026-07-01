# ============================================
# Automated Batch Readmission Risk Scoring Pipeline
# ============================================
# This script simulates an automated production workflow:
# 1. Pulls a random sample of patients from the existing modeling dataset (S3)
# 2. Loads the trained Logistic Regression, XGBoost, and Cox models (S3)
# 3. Applies the same preprocessing used during training
# 4. Scores each patient for readmission risk (probability) and
#    estimated time-to-readmission (Cox survival analysis)
# 5. Writes a structured batch report back to S3
#
# Note: Since the full dataset was used during training, this batch
# simulates a production-style re-scoring workflow (not unseen test data).
# In a live deployment this would score genuinely new patient records.
#
# Shared model loading / preprocessing / scoring logic lives in
# model_utils.py so this script and api.py (the real-time endpoint)
# can never drift out of sync with each other.
# ============================================

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import model_utils as mu

BATCH_SIZE = 100
RANDOM_SEED = None  # set to an int for reproducible sampling, None for true randomness
OUTPUT_PREFIX = f"{mu.MODEL_DATA_PREFIX}/batch-outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_scoring.log"),
    ],
)
log = logging.getLogger("batch_scoring")


def run_batch_scoring():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log.info(f"=== Starting batch scoring run: {run_timestamp} ===")

    full_df = mu.load_csv_from_s3(mu.BUCKET, mu.MODELING_DATA_KEY)
    models = mu.ModelBundle()

    if len(full_df) < BATCH_SIZE:
        raise ValueError(f"Dataset has only {len(full_df)} rows, fewer than requested batch size {BATCH_SIZE}")

    batch_raw = full_df.sample(n=BATCH_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    log.info(f"Sampled batch of {len(batch_raw)} patients")

    log.info("Scoring batch...")
    results = mu.score_patients(models, batch_raw)

    readable_cols = [c for c in ["subject_id", "hadm_id"] if c in batch_raw.columns]
    if readable_cols:
        results = pd.concat([batch_raw[readable_cols].reset_index(drop=True), results], axis=1)

    n_high_risk = (results["risk_tier"] == "high").sum()
    high_risk_days = results.loc[results["risk_tier"] == "high", "predicted_median_days_to_readmission"]
    high_risk_days_finite = high_risk_days.replace([np.inf, -np.inf], np.nan).dropna()
    avg_days_high_risk = high_risk_days_finite.mean() if len(high_risk_days_finite) > 0 else np.nan

    log.info("=== Batch Summary ===")
    log.info(f"Total patients scored: {len(results)}")
    log.info(f"High-risk patients flagged: {n_high_risk} ({n_high_risk/len(results)*100:.1f}%)")
    if pd.notna(avg_days_high_risk):
        log.info(f"Avg predicted days to readmission (high-risk group): {avg_days_high_risk:.1f}")
    else:
        log.info("Avg predicted days to readmission (high-risk group): not available (median survival not reached)")

    detail_key = f"{OUTPUT_PREFIX}/batch_{run_timestamp}_detail.csv"
    mu.upload_df_to_s3(results, mu.BUCKET, detail_key)

    summary_df = pd.DataFrame([{
        "run_timestamp": run_timestamp,
        "batch_size": len(results),
        "n_high_risk": int(n_high_risk),
        "pct_high_risk": round(n_high_risk / len(results) * 100, 1),
        "avg_days_to_readmission_high_risk": round(avg_days_high_risk, 1) if pd.notna(avg_days_high_risk) else None,
    }])
    summary_key = f"{OUTPUT_PREFIX}/batch_{run_timestamp}_summary.csv"
    mu.upload_df_to_s3(summary_df, mu.BUCKET, summary_key)

    log.info(f"=== Batch scoring run complete: {run_timestamp} ===")
    return results, summary_df


if __name__ == "__main__":
    run_batch_scoring()
