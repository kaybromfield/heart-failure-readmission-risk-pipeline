# Heart Failure Readmission Risk Pipeline

**Production ML pipeline built on top of [hospital-readmission-risk-model](https://github.com/kaybromfield/hospital-readmission-risk-model)**

---

## Overview

The [hospital-readmission-risk-model](https://github.com/kaybromfield/hospital-readmission-risk-model) identified key clinical and socioeconomic predictors of 30-day hospital readmission in heart failure patients using the MIMIC-IV clinical database.

This project operationalizes those findings into a deployable clinical tool — moving from research insight to production-ready infrastructure. It answers the question: *how do we make this model actually useful in a hospital setting?*

---

## What This Project Does

### Real-Time Patient Scoring (FastAPI)
A REST API that accepts individual patient features and immediately returns a readmission risk score, risk tier, and estimated time-to-readmission. Designed for point-of-care use — score a patient at admission or discharge in real time.

### Automated Population Screening (AWS Lambda + EventBridge)
A containerized batch pipeline that automatically runs 3x daily (12AM, 8AM, 4PM UTC) without manual intervention. Each run randomly samples 100 patients, scores them through all three models, and writes structured risk reports to S3.

### Cloud-Native Infrastructure
- Models and data stored in **AWS S3**
- Pipeline containerized with **Docker** and deployed to **AWS Lambda** via **ECR**
- Automated scheduling via **AWS EventBridge**
- Execution logs streamed to **AWS CloudWatch**

---

## Architecture

```
                    ┌─────────────────────┐
                    │   AWS EventBridge   │
                    │  (3x daily schedule)│
                    └────────┬────────────┘
                             │ triggers
                             ▼
┌─────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│  FastAPI    │    │   AWS Lambda        │    │   AWS S3         │
│  Endpoint   │───▶│   (Docker/ECR)      │───▶│  batch-outputs/  │
│  /predict   │    │   lambda_handler.py │    │  detail.csv      │
└─────────────┘    └────────┬────────────┘    │  summary.csv     │
                             │ loads from      └──────────────────┘
                             ▼
                    ┌─────────────────────┐
                    │   AWS S3            │
                    │   trained-models/   │
                    │   xgboost_model.pkl │
                    │   cox_model.pkl     │
                    │   logistic_*.pickle │
                    │   feature_cols.pkl  │
                    └─────────────────────┘
```

---

## Models

All three models were trained in [hospital-readmission-risk-model](https://github.com/kaybromfield/hospital-readmission-risk-model) on the MIMIC-IV heart failure cohort (~80,000 admissions, 32 features):

| Model | Purpose | Output |
|-------|---------|--------|
| XGBoost | 30-day readmission classification | Risk probability (0–1) |
| Logistic Regression | Baseline classification | Risk probability (0–1) |
| Cox Proportional Hazards | Survival analysis | Estimated days to readmission |

Risk is tiered as **low** (0–33%), **medium** (33–66%), or **high** (66–100%) based on XGBoost probability.

---

## Tech Stack

**Application**
- Python 3.11
- FastAPI + Uvicorn
- Pydantic (input validation)
- pandas, NumPy, scikit-learn, XGBoost, statsmodels, lifelines

**Infrastructure**
- AWS S3 (data + model storage)
- AWS ECR (Docker image registry)
- AWS Lambda (serverless compute)
- AWS EventBridge (scheduled triggers)
- AWS CloudWatch (logging + observability)
- Docker

**Development**
- boto3 (AWS SDK)
- joblib (model serialization)
- Git / GitHub

---

## Project Structure

```
heart-failure-readmission-risk-pipeline/
├── src/
│   ├── model_utils.py        # Shared model loading, preprocessing, scoring logic
│   ├── api.py                # FastAPI real-time scoring endpoint
│   ├── batch_scoring.py      # Local batch automation script
│   ├── lambda_handler.py     # AWS Lambda entry point
│   └── sql/                  # See hospital-readmission-risk-model for SQL pipeline
├── Dockerfile                # Lambda-compatible container image
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — confirms API is running and models are loaded |
| POST | `/predict-readmission-risk` | Score a single patient, returns risk scores + tier |
| GET | `/` | Root — links to docs and health check |

**API Example request:**

<img width="758" height="587" alt="input" src="https://github.com/user-attachments/assets/2f70df43-4fe9-4bc3-a517-45ca82e9c48d" />



**API Example response:**

<img width="750" height="135" alt="result" src="https://github.com/user-attachments/assets/a546c956-3cd0-46c9-9a40-89238358bfd8" />



---

## Batch Output Schema

Each automated run produces two files in S3:

**`batch_YYYYMMDD_HHMMSS_detail.csv`** — patient-level results
| Column | Description |
|--------|-------------|
| patient_index | Row identifier from sampled batch |
| xgboost_risk_score | XGBoost readmission probability |
| logistic_risk_score | Logistic regression readmission probability |
| predicted_median_days_to_readmission | Cox model time-to-readmission estimate |
| risk_tier | low / medium / high |
| actual_readmitted_30_days | Ground truth label (where available) |

**`batch_YYYYMMDD_HHMMSS_summary.csv`** — run-level aggregate
| Column | Description |
|--------|-------------|
| run_timestamp | Execution timestamp |
| batch_size | Number of patients scored |
| n_high_risk | Count of high-risk patients |
| pct_high_risk | Percentage flagged high-risk |
| avg_days_to_readmission_high_risk | Average predicted days (high-risk group) |

---

## Data & Privacy

This project uses the **MIMIC-IV** clinical database, a restricted de-identified dataset requiring credentialed access through [PhysioNet](https://physionet.org/). No patient data or PHI-derived files are included in this repository. All modeling data and trained model artifacts are stored privately in AWS S3.

This is a demonstration system built on historical research data. In a production clinical setting, additional requirements would apply including EHR integration, FHIR compliance, clinical validation, and regulatory review.

---

## Related Project

**[hospital-readmission-risk-model](https://github.com/kaybromfield/hospital-readmission-risk-model)** — the upstream research project containing the full SQL data pipeline, feature engineering, model training, and statistical analysis that this pipeline is built on.
