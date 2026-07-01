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
    Patient input features for readmission risk scoring.
    Accepts the 12 most statistically significant predictors identified
    in the upstream hospital-readmission-risk-model analysis.
    Field validation happens automatically -- FastAPI will reject
    malformed requests before data ever reaches the model.
    """
    # Clinical features
    length_of_stay_days: float = Field(..., ge=0, description="Length of hospital stay in days")
    abnormal_lab_ratio: float = Field(..., ge=0, le=1, description="Proportion of lab results flagged abnormal (0-1)")
    avg_creatinine: float = Field(..., ge=0, description="Average serum creatinine level (mg/dL)")
    avg_hemoglobin: float = Field(..., ge=0, description="Average hemoglobin level (g/dL)")
    avg_sodium: float = Field(..., ge=0, description="Average serum sodium level (mEq/L)")

    # Comorbidity flags (0 = absent, 1 = present)
    cancer_flag: int = Field(0, ge=0, le=1, description="Active cancer diagnosis")
    ckd_flag: int = Field(0, ge=0, le=1, description="Chronic kidney disease")
    depression_flag: int = Field(0, ge=0, le=1, description="Depression diagnosis")
    diabetes_flag: int = Field(0, ge=0, le=1, description="Diabetes diagnosis")
    hypertension_flag: int = Field(0, ge=0, le=1, description="Hypertension diagnosis")
    valvular_hd_flag: int = Field(0, ge=0, le=1, description="Valvular heart disease")

    # Socioeconomic features
    married_flag: int = Field(0, ge=0, le=1, description="Marital status (1 = married)")
    insurance_Private: int = Field(0, ge=0, le=1, description="Private insurance (1 = yes)")

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
