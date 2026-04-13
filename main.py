from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ── Load models once at startup ───────────────────────────────────────────────
try:
    rf_pos   = joblib.load("model_pos.pkl")
    rf_neg   = joblib.load("model_neg.pkl")
    weights  = joblib.load("weights.pkl")
    features = joblib.load("features.pkl")
    W_POS    = weights["w_pos"]
    W_NEG    = weights["w_neg"]
    POS_FEATS = features["pos_feats"]
    NEG_FEATS = features["neg_feats"]
except FileNotFoundError:
    raise RuntimeError("Model files not found. Run train.py first.")

app = FastAPI(
    title="Inflation Prediction API",
    description="Dual Random Forest ensemble trained on 35 years of Federal Reserve macroeconomic data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────────────
class InflationRequest(BaseModel):
    synthetic_target_rate: float = Field(..., example=5.25,
        description="Fed funds target rate (or midpoint of range post-2008)")
    real_gdp_change: float       = Field(..., example=2.1,
        description="Real GDP percent change (annual)")
    unemployment_rate: float     = Field(..., example=3.9,
        description="Annual average unemployment rate (%)")
    deviation: float             = Field(..., example=-0.15,
        description="Effective rate minus synthetic target rate")
    is_post_2008: int            = Field(..., example=1,
        description="1 if year is 2008 or later, else 0")
    is_crisis: int               = Field(..., example=0,
        description="1 for tightening crisis, -1 for easing crisis, 0 for normal")

# ── Response schema ───────────────────────────────────────────────────────────
class InflationResponse(BaseModel):
    predicted_inflation: float
    model_weights: dict
    inputs_received: dict

# ── Helper: build regime3 from is_crisis + deviation ─────────────────────────
def compute_regime3(is_crisis: int, deviation: float) -> int:
    NEG_DEV_EPS = -0.03
    if is_crisis == 1:
        return -1 if deviation <= NEG_DEV_EPS else +1
    return 0

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Inflation Prediction API is running.",
        "usage": "POST /predict with macroeconomic inputs",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=InflationResponse)
def predict(req: InflationRequest):
    try:
        regime3 = compute_regime3(req.is_crisis, req.deviation)

        # Build feature rows matching training column order
        pos_row = pd.DataFrame([{
            "Synthetic_Target_Rate":    req.synthetic_target_rate,
            "Real GDP (Percent Change)": req.real_gdp_change,
            "Deviation":                req.deviation,
            "is_post_2008":             req.is_post_2008,
            "regime3":                  regime3
        }])[POS_FEATS]

        neg_row = pd.DataFrame([{
            "Unemployment Rate": req.unemployment_rate,
            "Deviation":         req.deviation,
            "is_post_2008":      req.is_post_2008,
            "regime3":           regime3
        }])[NEG_FEATS]

        pred_pos = rf_pos.predict(pos_row)[0]
        pred_neg = rf_neg.predict(neg_row)[0]
        prediction = round(float(W_POS * pred_pos + W_NEG * pred_neg), 4)

        return InflationResponse(
            predicted_inflation=prediction,
            model_weights={"positive_model": round(W_POS, 3), "negative_model": round(W_NEG, 3)},
            inputs_received=req.dict()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
