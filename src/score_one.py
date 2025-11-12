# analytics/src/score_one.py
import os, sys, json, joblib
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta
from .common import sql_engine
from .features import engineer



MODEL_PATH = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
TS_FMT = "%Y-%m-%d %H:%M:%S"

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        print(f"ERR: model bundle not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    return joblib.load(MODEL_PATH)

def fetch_context(eng, tx_id: int):
    # Pull the target row
    tx = pd.read_sql(text("""
        SELECT * FROM ops.Transactions WHERE tx_id = :id
    """), eng, params={"id": tx_id})
    if tx.empty:
        print(f"ERR: tx_id {tx_id} not found", file=sys.stderr)
        sys.exit(3)

    # Parse timestamp (string) for a lookback
    tx_utc = pd.to_datetime(tx.iloc[0]["tx_utc"], format=TS_FMT, utc=True, errors="coerce")
    if pd.isna(tx_utc):
        # if parsing fails, just use a wide window
        tx_utc = pd.Timestamp.utcnow()

    card_id = int(tx.iloc[0]["card_id"])
    cust_id = int(tx.iloc[0]["customer_id"])

    # Pull recent history for same card + same customer (30 days window)
    since = (tx_utc - pd.Timedelta(days=30)).strftime(TS_FMT)
    ctx = pd.read_sql(text(f"""
        SELECT *
        FROM ops.Transactions
        WHERE (card_id = :card OR customer_id = :cust)
          AND tx_utc >= :since
          AND tx_id <= :id
        ORDER BY tx_utc, tx_id
    """), eng, params={"card": card_id, "cust": cust_id, "since": since, "id": tx_id})

    # Ensure target row is present
    if ctx[ctx["tx_id"] == tx_id].empty:
        ctx = pd.concat([ctx, tx], ignore_index=True)

    # Add a dummy label column (engineer expects it when is_training=True sometimes)
    if "label" not in ctx.columns:
        ctx["label"] = 0

    return ctx

def upsert_score(eng, tx_id: int, score: float, label_pred: bool):
    with eng.begin() as con:
        # Try update, else insert
        rows = con.execute(text("""
            UPDATE ml.TxScores
            SET score = :score, label_pred = :label_pred,
                reason_json = COALESCE(reason_json,'{"source":"model"}')
            WHERE tx_id = :id;
            SELECT @@ROWCOUNT AS rc;
        """), {"id": tx_id, "score": float(score), "label_pred": bool(label_pred)}).fetchone()
        if rows and rows.rc == 0:
            con.execute(text("""
                INSERT INTO ml.TxScores (tx_id, score, label_pred, reason_json)
                VALUES (:id, :score, :label_pred, '{"source":"model"}');
            """), {"id": tx_id, "score": float(score), "label_pred": bool(label_pred)})

def main():
    if len(sys.argv) < 2:
        raise LookupError(f"tx_id {tx_id} not found")

    tx_id = int(sys.argv[1])
    eng = sql_engine()
    bundle = load_bundle()

    # bundle: { which, model, threshold, features, ... }
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))
    feat_cols = bundle["features"]

    df = fetch_context(eng, tx_id)
    # Build features for ALL rows, then pick target row at the end
    X, _, meta, _ = engineer(df, is_training=False)

    # Align features to training columns (add missing, drop extras)
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feat_cols].fillna(0.0)

    # Score all; take the last row that corresponds to tx_id
    # (context ordered by time, but we ensure meta has tx_id)
    idx = meta.index[meta["tx_id"] == tx_id]
    if len(idx) == 0:
        # fallback: last row
        idx = [X.index[-1]]

    y_score_all = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    score = float(y_score_all[idx[-1]])
    label_pred = score >= thr

    upsert_score(eng, tx_id, score, label_pred)
    print(json.dumps({"tx_id": tx_id, "score": score, "label_pred": bool(label_pred)}))

if __name__ == "__main__":
    main()
