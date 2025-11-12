# analytics/src/score.py
import os, json, joblib
import pandas as pd
from sqlalchemy import text
from common import sql_engine
from extract import get_scoring_candidates
from features import engineer

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v2.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")

def _reason_stub(X_row: pd.Series) -> dict:
    """
    Lightweight reason extraction—good enough for UI until SHAP is added.
    """
    keys = ["amount_log","card_tx_count_24h","cust_tx_count_24h","keyed_nan","entry_mode_keyed","channel_ecom"]
    reasons = {}
    for k in keys:
        if k in X_row.index:
            v = float(X_row[k])
            if abs(v) > 0:
                reasons[k] = v
    return reasons

def score_and_write():
    eng = sql_engine()
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    thr = float(bundle["threshold"])

    # create a run for this batch scoring
    with eng.begin() as con:
        run_id = con.execute(text(
          "INSERT INTO ml.Runs(model_version, notes) OUTPUT INSERTED.run_id VALUES(:mv, :notes)"
        ), {"mv": bundle.get("model_version","v2"), "notes":"batch score"}).scalar_one()

    df = get_scoring_candidates()
    if df.empty:
        print("No new transactions to score.")
        # still mark run finished
        with eng.begin() as con:
            con.execute(text("UPDATE ml.Runs SET finished_at=SYSUTCDATETIME() WHERE run_id=:r"), {"r": run_id})
        return

    X, _, meta, feat_cols = engineer(df, is_training=False)
    # align features with training set
    missing = [c for c in bundle["features"] if c not in X.columns]
    for c in missing: X[c] = 0.0
    X = X[bundle["features"]]

    proba = model.predict_proba(X)[:,1]
    pred  = (proba >= thr).astype(int)

    rows = []
    for i, tx_id in enumerate(meta["tx_id"].values):
        reasons = _reason_stub(X.iloc[i])
        rows.append({
            "tx_id": int(tx_id), "run_id": int(run_id), "score": float(proba[i]),
            "label_pred": int(pred[i]), "threshold": thr,
            "reason_json": json.dumps(reasons)
        })

    pd.DataFrame(rows).to_sql("TxScores", eng, schema="ml", if_exists="append", index=False)

    with eng.begin() as con:
        con.execute(text("UPDATE ml.Runs SET finished_at=SYSUTCDATETIME() WHERE run_id=:r"), {"r": run_id})

    print(f"Scored {len(rows)} transactions | run_id={run_id}")

if __name__ == "__main__":
    score_and_write()
