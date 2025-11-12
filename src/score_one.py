# analytics/src/score_one.py
import os
import pickle
from urllib.parse import quote_plus

import pandas as pd

# --- Loading the trained bundle ---
def load_bundle():
    model_path = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found at {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

# --- Simple context fetch (service.py already has a safe fallback) ---
def fetch_context(engine, tx_id: int) -> pd.DataFrame:
    # This SELECT is simple; service.py will fall back to _fetch_context_safe() if anything fails.
    with engine.connect() as c:
        # Detect the PK/ID column at runtime from service.py; here we assume TxId or tx_id exists,
        # but service.py will catch and fallback if not.
        # Pull the exact tx row plus a small history window, if desired.
        df = pd.read_sql_query(
            f"SELECT TOP 1 * FROM [ops].[Transactions] WHERE [tx_id] = {int(tx_id)}",
            engine,
        )
    if df.empty:
        raise LookupError(f"tx_id {tx_id} not found")
    return df

# --- Upsert the score into ml.TxScores explicitly (SQL Server, SA 2.x-safe) ---
def upsert_score(engine, tx_id: int, score: float, flagged: bool):
    """
    Writes/updates the score for a transaction to [ml].[TxScores].
    Uses MERGE with parameter binding (pyodbc '?').
    """
    schema = os.getenv("SCORE_SCHEMA", "ml")
    table  = os.getenv("SCORE_TABLE",  "TxScores")
    threshold = float(os.getenv("THRESHOLD", "0.5"))
    run_id    = os.getenv("MODEL_VERSION", "unknown")

    def q(name: str) -> str:
        return f"[{name}]"

    merge_sql = f"""
MERGE {q(schema)}.{q(table)} AS tgt
USING (
    SELECT
        CAST(? AS BIGINT)       AS tx_id,
        CAST(? AS FLOAT)        AS score,
        CAST(? AS BIT)          AS label_pred,
        CAST(? AS FLOAT)        AS threshold,
        CAST(? AS NVARCHAR(64)) AS run_id
) AS src
ON (tgt.tx_id = src.tx_id)
WHEN MATCHED THEN
    UPDATE SET
        score       = src.score,
        label_pred  = src.label_pred,
        threshold   = src.threshold,
        run_id      = src.run_id,
        explained_at = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
    INSERT (tx_id, score, label_pred, threshold, run_id, explained_at)
    VALUES (src.tx_id, src.score, src.label_pred, src.threshold, src.run_id, SYSUTCDATETIME());
"""

    params = (int(tx_id), float(score), 1 if flagged else 0, threshold, run_id)
    # Use a transaction so the write is committed even if the connection pool recycles.
    with engine.begin() as conn:
        conn.exec_driver_sql(merge_sql, params)
