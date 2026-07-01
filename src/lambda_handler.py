# ============================================
# lambda_handler.py
# ============================================
# AWS Lambda entry point. Lambda invokes `handler(event, context)` --
# it does not run scripts via `if __name__ == "__main__"` the way a
# normal local script does. This wraps the same batch scoring logic
# used locally so the underlying behavior is identical whether run
# manually or triggered automatically by EventBridge.
#
# Lambda automatically captures anything printed to stdout/stderr and
# sends it to CloudWatch Logs -- so the existing logging setup in
# model_utils.py / batch_scoring.py works here with no changes needed
# for observability.
# ============================================

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import model_utils as mu

# Lambda's runtime captures stdout, so a simple StreamHandler is enough --
# no FileHandler needed here, since Lambda's filesystem is ephemeral and
# CloudWatch is the actual log destination in this environment.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lambda_handler")

BATCH_SIZE = 100
OUTPUT_PREFIX = f"{mu.MODEL_DATA_PREFIX}/batch-outputs"


def run_batch_scoring():
    """Same logic as the local batch_scoring.py script, run here inside
    the Lambda handler instead of a __main__ block."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log.info(f"=== Starting batch scoring run: {run_timestamp} ===")

    full_df = mu.load_csv_from_s3(mu.BUCKET, mu.MODELING_DATA_KEY)
    models = mu.ModelBundle()

    if len(full_df) < BATCH_SIZE:
        raise ValueError(f"Dataset has only {len(full_df)} rows, fewer than requested batch size {BATCH_SIZE}")

    batch_raw = full_df.sample(n=BATCH_SIZE).reset_index(drop=True)
    log.info(f"Sampled batch of {len(batch_raw)} patients")

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
    return run_timestamp, len(results), int(n_high_risk)


def handler(event, context):
    """
    Required AWS Lambda entry point signature.
    `event` contains details about what triggered the function (here, an
    EventBridge scheduled rule). `context` contains Lambda runtime info
    (request ID, remaining execution time, etc.).
    """
    log.info(f"Lambda invoked. Trigger event: {event}")

    try:
        run_timestamp, batch_size, n_high_risk = run_batch_scoring()
        return {
            "statusCode": 200,
            "body": {
                "message": "Batch scoring completed successfully",
                "run_timestamp": run_timestamp,
                "batch_size": batch_size,
                "n_high_risk": n_high_risk,
            },
        }
    except Exception as e:
        log.error(f"Batch scoring failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": {"message": "Batch scoring failed", "error": str(e)},
        }
