# ============================================
# api.py -- Readmission Risk Scoring API
# ============================================
# Wraps the trained Logistic Regression, XGBoost, and Cox models in a
# FastAPI application so they can be called as a real-time service,
# rather than only run as a manual/scheduled batch job.
#
# Models are loaded ONCE at startup (not per-request) for performance.
# Run locally with:
#   uvicorn api:app --reload
# Then visit http://127.0.0.1:8000/docs for interactive API docs.
# ============================================

import logging
import sys
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import model_utils as mu

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log"),
    ],
)
log = logging.getLogger("api")

app = FastAPI(
    title="Heart Failure Readmission Risk API",
    description="Scores patients for 30-day readmission risk using XGBoost, "
                 "Logistic Regression, and Cox survival analysis.",
    version="1.0.0",
)

# Loaded once at startup, reused across all requests
models: Optional[mu.ModelBundle] = None


@app.on_event("startup")
def load_models():
    global models
    log.info("Loading model artifacts from S3 at startup...")
    models = mu.ModelBundle()
    log.info("Models loaded. API ready to serve requests.")


# ============================================
# REQUEST / RESPONSE SCHEMAS (Pydantic)
# ============================================
class PatientFeatures(BaseModel):
    """
    One patient's input features. Field validation happens automatically --
    FastAPI will reject malformed requests (wrong types, missing required
    fields) before this data ever reaches the model.
    """
    length_of_stay_days: float = Field(..., ge=0, description="Length of hospital stay in days")
    married_flag: int = Field(..., ge=0, le=1)
    hypertension_flag: int = Field(0, ge=0, le=1)
    afib_flag: int = Field(0, ge=0, le=1)
    ischemic_hd_flag: int = Field(0, ge=0, le=1)
    valvular_hd_flag: int = Field(0, ge=0, le=1)
    ckd_flag: int = Field(0, ge=0, le=1)
    diabetes_flag: int = Field(0, ge=0, le=1)
    copd_flag: int = Field(0, ge=0, le=1)
    anemia_flag: int = Field(0, ge=0, le=1)
    sleep_disorder_flag: int = Field(0, ge=0, le=1)
    obesity_flag: int = Field(0, ge=0, le=1)
    depression_flag: int = Field(0, ge=0, le=1)
    cancer_flag: int = Field(0, ge=0, le=1)
    thyroid_flag: int = Field(0, ge=0, le=1)
    avg_creatinine: float = Field(..., ge=0)
    avg_sodium: float = Field(..., ge=0)
    avg_hemoglobin: float = Field(..., ge=0)
    abnormal_lab_ratio: float = Field(..., ge=0, le=1)
    insurance_Private: int = Field(0, ge=0, le=1)
    gender: Optional[str] = Field(None, description="e.g. 'M' or 'F'")

    class Config:
        extra = "allow"  # allow additional columns the model may use without rejecting the request


class RiskScoreResponse(BaseModel):
    xgboost_risk_score: float
    logistic_risk_score: float
    predicted_median_days_to_readmission: Optional[float]
    risk_tier: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ============================================
# ENDPOINTS
# ============================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Basic health check -- confirms the API is up and models are loaded."""
    return HealthResponse(status="ok", models_loaded=models is not None)


@app.post("/predict-readmission-risk", response_model=RiskScoreResponse)
def predict_readmission_risk(patient: PatientFeatures):
    """
    Scores a single patient's 30-day readmission risk using all three
    trained models. Returns risk probabilities, a risk tier, and an
    estimated median days-to-readmission where available.
    """
    if models is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Try again shortly.")

    start = time.time()
    log.info(f"Received scoring request: {patient.dict()}")

    try:
        df = pd.DataFrame([patient.dict()])
        results = mu.score_patients(models, df)
        row = results.iloc[0]

        median_days = row["predicted_median_days_to_readmission"]
        median_days_clean = None if pd.isna(median_days) or median_days == float("inf") else float(median_days)

        response = RiskScoreResponse(
            xgboost_risk_score=float(row["xgboost_risk_score"]),
            logistic_risk_score=float(row["logistic_risk_score"]),
            predicted_median_days_to_readmission=median_days_clean,
            risk_tier=str(row["risk_tier"]),
        )

        elapsed_ms = (time.time() - start) * 1000
        log.info(f"Scored successfully in {elapsed_ms:.1f}ms -> risk_tier={response.risk_tier}")

        return response

    except Exception as e:
        log.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "Heart Failure Readmission Risk API",
        "docs": "/docs",
        "health": "/health",
    }
