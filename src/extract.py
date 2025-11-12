import pandas as pd
from .common import sql_engine

def get_training_data():
    eng = sql_engine()
    q = """
    SELECT t.tx_id, t.customer_id, t.card_id, t.merchant_id,
           t.amount, t.currency, t.tx_utc, t.entry_mode, t.channel,
           ISNULL(l.label, 0) AS label
    FROM ops.Transactions t
    LEFT JOIN ml.Labels l ON l.tx_id = t.tx_id
    WHERE t.tx_utc < DATEADD(day, -1, SYSUTCDATETIME()) -- temporal guard to avoid leakage
    """
    return pd.read_sql(q, eng)

def get_scoring_candidates():
    eng = sql_engine()
    q = """
    SELECT t.tx_id, t.customer_id, t.card_id, t.merchant_id,
           t.amount, t.currency, t.tx_utc, t.entry_mode, t.channel
    FROM ops.Transactions t
    LEFT JOIN ml.TxScores s ON s.tx_id = t.tx_id
    WHERE s.tx_id IS NULL
    """
    return pd.read_sql(q, eng)
