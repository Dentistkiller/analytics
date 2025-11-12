# analytics/src/score_one.py
import os
import json
import joblib
import pandas as pd
from sqlalchemy import text
from datetime import timedelta
from .common import sql_engine
from .features import engineer

# -------- Configuration --------
MODEL_PATH    = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
SCORE_SCHEMA  = os.getenv("SCORE_SCHEMA", "ml")
SCORE_TABLE   = os.getenv("SCORE_TABLE",  "TxScores")
TS_FMT        = "%Y-%m-%d %H:%M:%S"

def load_bundle():
    """Load the trained bundle (model, threshold, features...)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def fetch_context(eng, tx_id: int) -> pd.DataFrame:
    """
    Fetch the target transaction plus a small recent context window.
    This version raises exceptions (no sys.exit), so callers can return JSON errors.
    """
    # Target row
    tx = pd.read_sql(
        text("SELECT * FROM [ops].[Transactions] WHERE [tx_id] = :id"),
        eng,
        params={"id": tx_id},
    )
    if tx.empty:
        raise LookupError(f"tx_id {tx_id} not found")

    # Parse timestamp for a lookback window
    tx_utc = pd.to_datetime(tx.iloc[0]["tx_utc"], format=TS_FMT, utc=True, errors="coerce")
    if pd.isna(tx_utc):
        # If parsing fails, just use now so we still get a narrow query
        tx_utc = pd.Timestamp.utcnow()

    card_id = int(tx.iloc[0]["card_id"])
    cust_id = int(tx.iloc[0]["customer_id"])
    since   = (tx_utc - pd.Timedelta(days=30)).strftime(TS_FMT)

    ctx = pd.read_sql(
        text("""
            SELECT *
            FROM [ops].[Transactions]
            WHERE (card_id = :card OR customer_id = :cust)
              AND tx_utc >= :since
              AND tx_id <= :id
            ORDER BY tx_utc, tx_id
        """),
        eng,
        params={"card": card_id, "cust": cust_id, "since": since, "id": tx_id},
    )

    # Ensure the target row is present
    if ctx[ctx["tx_id"] == tx_id].empty:
        ctx = pd.concat([ctx, tx], ignore_index=True)

    # Add dummy label column if missing (some feature pipelines expect it)
    if "label" not in ctx.columns:
        ctx["label"] = 0

    return ctx


Query 1
35363738394041424344454647
EXEC sp_executesql @sql;

-- Outgoing FKs (ml.TxScores referencing others)
SET @sql = N'';
SELECT @sql = @sql + N'ALTER TABLE ' + QUOTENAME(OBJECT_SCHEMA_NAME(parent_object_id)) + N'.' +
                      QUOTENAME(OBJECT_NAME(parent_object_id)) + N' DROP CONSTRAINT ' + QUOTENAME(name) + N';'
FROM sys.foreign_keys
WHERE parent_object_id = @obj_id;
EXEC sp_executesql @sql;


Results
Messages
Failed to execute query. Error: Incorrect syntax near 'QUOTENAME'.
Incorrect syntax near 'QUOTENAME'.
Incorrect syntax near 'QUOTENAME'.
Incorrect syntax near 'QUOTENAME'.
def main():
    """
    CLI usage (optional): python -m src.score_one <tx_id>
    Prints JSON {tx_id, score, label_pred}.
    """
    import sys
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m src.score_one <tx_id>")
    tx_id = int(sys.argv[1])

    eng    = sql_engine()
    bundle = load_bundle()
    model  = bundle["model"]
    thr    = float(bundle.get("threshold", 0.5))
    feats  = bundle.get("features") or []

    df = fetch_context(eng, tx_id)
    X, _, meta, _ = engineer(df, is_training=False)

    # Align features
    for col in feats:
        if col not in X.columns:
            X[col] = 0.0
    if feats:
        X = X[feats].fillna(0.0)

    # Pick the target row (fallback to last)
    row_index = None
    if hasattr(meta, "columns") and "tx_id" in meta.columns:
        hits = meta.index[meta["tx_id"] == tx_id]
        if len(hits) > 0:
            row_index = hits[-1]
    if row_index is None:
        row_index = X.index[-1]

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[row_index, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[row_index])
    else:
        score = float(model.predict(X)[row_index])

    label_pred = bool(score >= thr)
    upsert_score(eng, tx_id, score, label_pred)

    print(json.dumps({"tx_id": tx_id, "score": score, "label_pred": label_pred}))

if __name__ == "__main__":
    main()
