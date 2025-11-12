import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from dotenv import load_dotenv

# --- reuse your package code ---
from analytics.src.score_one import load_bundle, fetch_context, upsert_score, TS_FMT
from analytics.src.common import sql_engine
from analytics.src.features import engineer

load_dotenv()  # local dev only

app = FastAPI(title="Fraud Scoring API", version="1.0.0")

# env flags
ENV = os.getenv("ENV", "prod").lower()     # set ENV=dev locally to see detailed errors
DEMO = os.getenv("DEMO_MODE", "0") == "1"  # optional demo switch

# singletons (lazy)
ENGINE = None
BUNDLE = None
MODEL = None
THRESHOLD = None
FEAT_COLS = None


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = sql_engine()
    return ENGINE


def _ensure_model():
    """
    Optional: if MODEL_URL is set and MODEL_PATH missing, download it.
    """
    mp = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
    mu = os.getenv("MODEL_URL")
    if not os.path.exists(mp):
        if mu:
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(mu, mp)
        else:
            # don't crash here; let load_bundle raise a clear error
            pass
    return mp


def _ensure_bundle():
    global BUNDLE, MODEL, THRESHOLD, FEAT_COLS
    if BUNDLE is None:
        _ensure_model()
        BUNDLE = load_bundle()  # expects MODEL_PATH env
        MODEL = BUNDLE["model"]
        THRESHOLD = float(BUNDLE.get("threshold", 0.5))
        FEAT_COLS = BUNDLE["features"]
    return BUNDLE


def _fail(e: Exception):
    # Return detailed error only in dev
    msg = f"{type(e).__name__}: {e}"
    if ENV == "dev":
        raise HTTPException(status_code=500, detail=f"Scoring failed: {msg}")
    raise HTTPException(status_code=500, detail="Scoring failed")


@app.get("/health")
def health():
    try:
        _ensure_bundle()
        return {"status": "ok", "model_loaded": bool(MODEL), "threshold": THRESHOLD}
    except Exception as e:
        if ENV == "dev":
            return {"status": "degraded", "detail": f"{type(e).__name__}: {e}"}
        return {"status": "degraded"}


@app.get("/diag")
def diag():
    # Quick visibility into model path and DB connectivity
    mp = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
    model_exists = os.path.exists(mp)
    

    try:
        eng = _ensure_engine()
        with eng.connect() as con:
            ver = con.execute(text("SELECT @@VERSION")).scalar()
        db_ok = True
        db_info = str(ver)
    except Exception as e:
        db_ok = False
        db_info = f"{type(e).__name__}: {e}"

    try:
        _ensure_bundle()
        model_ok = True
    except Exception as e:
        model_ok = False
        db_info = db_info  # unchanged
    
    engine_url = str(ENGINE.url) if ENGINE else "not-created"


    return {
        "ENV": ENV,
        "model_path": mp,
        "model_exists": model_exists,
        "model_loaded": model_ok,
        "db_ok": db_ok,
        "db_info": db_info,
        "engin_url":engine_url
    }


@app.post("/score")
def score(payload: dict):
    # Validate input
    try:
        tx_id = int(payload.get("tx_id", 0))
        if tx_id <= 0:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tx_id")

    # Demo short-circuit (optional)
    if DEMO:
        return JSONResponse({
            "tx_id": tx_id, "score": 0.42, "label_pred": (tx_id % 2 == 0),
            "threshold": 0.5, "model": "demo"
        })

    try:
        # Ensure dependencies
        _ensure_engine()
        _ensure_bundle()

        # Fast existence check → 404 if not found
        with ENGINE.connect() as con:
            exists = con.execute(text("SELECT 1 FROM ops.Transactions WHERE tx_id=:id"),
                                 {"id": tx_id}).first()
        if not exists:
            raise HTTPException(status_code=404, detail=f"tx_id {tx_id} not found")

        # 1) fetch context for this tx
        df = fetch_context(ENGINE, tx_id)

        # 2) engineer features for ALL rows, pick the target later
        X, _, meta, _ = engineer(df, is_training=False)

        # 3) align to training feature set
        for col in FEAT_COLS:
            if col not in X.columns:
                X[col] = 0.0
        X = X[FEAT_COLS].fillna(0.0)

        # 4) locate row for tx_id
        idx = meta.index[meta["tx_id"] == tx_id]
        if len(idx) == 0:
            idx = [X.index[-1]]

        # 5) score
        if hasattr(MODEL, "predict_proba"):
            y_all = MODEL.predict_proba(X)[:, 1]
        else:
            y_all = MODEL.decision_function(X)

        score_val = float(y_all[idx[-1]])
        label_pred = bool(score_val >= THRESHOLD)

        # 6) upsert output
        upsert_score(ENGINE, tx_id, score_val, label_pred)

        return JSONResponse({
            "tx_id": tx_id,
            "score": score_val,
            "label_pred": label_pred,
            "threshold": THRESHOLD,
            "model": BUNDLE.get("which")
        })
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
