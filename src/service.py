import os
import traceback
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # MUST run before importing modules that read env vars

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel

import pandas as pd
from sqlalchemy.exc import ResourceClosedError, ProgrammingError

# ---- Local package imports (relative to src/) ----
from .common import sql_engine
from .features import engineer
from .score_one import load_bundle, fetch_context, upsert_score

app = FastAPI(title="VeriPay Scoring API", version=os.getenv("MODEL_VERSION", "unknown"))

# ---------------- Config for your transactions table ----------------
TX_SCHEMA = os.getenv("TX_SCHEMA", "ops")
TX_TABLE  = os.getenv("TX_TABLE",  "Transactions")

def _qn(name: str) -> str:
    """Quote a SQL Server identifier with brackets."""
    return f"[{name}]"

def _table_2part(schema: str, table: str) -> str:
    return f"{_qn(schema)}.{_qn(table)}"

# ---------------- Models ----------------
class ScoreResponse(BaseModel):
    tx_id: int
    score: float
    flagged: bool
    explain: Optional[dict] = None

class TxRequest(BaseModel):
    tx_id: int

class LatestTx(BaseModel):
    tx_id: int

# ---------------- Global Bundle Warmup ----------------
BUNDLE = load_bundle()
MODEL = BUNDLE.get("model")
THR = float(BUNDLE.get("threshold", os.getenv("THRESHOLD", 0.5)))
FEATS = BUNDLE.get("features") or []

# ---------------- Error Handling ----------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    # Always return structured JSON (no blank 500s)
    print("UNHANDLED EXCEPTION:", repr(exc))
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
        },
    )

# ---------------- Helper: detect PK / ID column ----------------
def _detect_pk_column(conn) -> str | None:
    """
    Return the first PK column name for TX_SCHEMA.TX_TABLE, or None.
    """
    sql = f"""
    SELECT TOP 1 c.name
    FROM sys.indexes i
    JOIN sys.index_columns ic
        ON ic.object_id = i.object_id AND ic.index_id = i.index_id AND ic.key_ordinal = 1
    JOIN sys.columns c
        ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID('{TX_SCHEMA}.{TX_TABLE}')
      AND i.is_primary_key = 1
    ORDER BY ic.key_ordinal;
    """
    row = conn.exec_driver_sql(sql).first()
    return row[0] if row else None

def _fallback_id_column(conn) -> str | None:
    """
    If no PK, try common names, identity, or the first int/decimal column.
    """
    # 1) Common names
    cols = [r[0] for r in conn.exec_driver_sql(f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{TX_SCHEMA}' AND TABLE_NAME = '{TX_TABLE}'
    """).all()]
    for c in ["TxId", "TransactionId", "TransactionID", "TransId", "Id", "ID"]:
        if c in cols:
            return c

    # 2) Identity column
    ident = conn.exec_driver_sql(f"""
        SELECT c.name
        FROM sys.columns c
        WHERE c.object_id = OBJECT_ID('{TX_SCHEMA}.{TX_TABLE}')
          AND c.is_identity = 1
        ORDER BY c.column_id
    """).first()
    if ident:
        return ident[0]

    # 3) Any numeric column
    any_int = conn.exec_driver_sql(f"""
        SELECT TOP 1 COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{TX_SCHEMA}' AND TABLE_NAME = '{TX_TABLE}'
          AND DATA_TYPE IN ('bigint','int','numeric','decimal')
        ORDER BY ORDINAL_POSITION
    """).first()
    return any_int[0] if any_int else None

def _get_id_column(conn) -> str:
    pk = _detect_pk_column(conn)
    if pk:
        return pk
    fb = _fallback_id_column(conn)
    if fb:
        return fb
    raise HTTPException(status_code=500, detail=f"Could not determine an ID column for {TX_SCHEMA}.{TX_TABLE}")

# ---------------- Safe context fetch (fallback for SA 2.x cursor issues) ----------------
def _fetch_context_safe(eng, tx_id: int) -> pd.DataFrame:
    """
    Safe read of the target tx and some history rows using SQLAlchemy 2.x + pandas.
    Avoids cursor/Result misuse that triggers ResourceClosedError.
    """
    with eng.connect() as c:
        id_col = _get_id_column(c)
    # Pull the target row + up to 1000 previous rows by ID (tune as needed)
    sql = (
        f"SELECT TOP 1000 * "
        f"FROM {_table_2part(TX_SCHEMA, TX_TABLE)} "
        f"WHERE {_qn(id_col)} <= {int(tx_id)} "
        f"ORDER BY {_qn(id_col)} ASC"
    )
    df = pd.read_sql_query(sql, eng)
    if df.empty:
        raise LookupError(f"tx_id {tx_id} not found")
    return df

# ---------------- Health & Diagnostics ----------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Service healthy"}

@app.get("/drivers")
def drivers():
    import pyodbc
    return {"pyodbc_drivers": pyodbc.drivers()}

@app.get("/db/ping")
def db_ping():
    eng = sql_engine()
    with eng.connect() as c:
        db = c.exec_driver_sql("SELECT DB_NAME()").scalar_one()
        version = c.exec_driver_sql("SELECT @@VERSION").scalar_one()
    return {"status": "ok", "db": db, "version": version}

@app.get("/db/schema")
def db_schema():
    eng = sql_engine()
    with eng.connect() as c:
        cols = c.exec_driver_sql(f"""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{TX_SCHEMA}' AND TABLE_NAME = '{TX_TABLE}'
            ORDER BY ORDINAL_POSITION
        """).all()
    return {
        "table": f"{TX_SCHEMA}.{TX_TABLE}",
        "columns": [
            {"name": r[0], "type": r[1], "nullable": r[2], "position": int(r[3])}
            for r in cols
        ],
    }

@app.get("/diag")
def diag():
    eng = sql_engine()
    driver = getattr(eng.dialect, "driver", "?")
    with eng.connect() as c:
        id_col = _get_id_column(c)
        cnt = c.exec_driver_sql(f"SELECT COUNT(*) FROM {_table_2part(TX_SCHEMA, TX_TABLE)}").scalar_one()
    return {
        "status": "ok",
        "server": os.getenv("SQL_SERVER"),
        "db": os.getenv("SQL_DB"),
        "driver": driver,
        "table": f"{TX_SCHEMA}.{TX_TABLE}",
        "id_column": id_col,
        "tx_count": int(cnt),
        "model_path": os.getenv("MODEL_PATH"),
        "model_version": os.getenv("MODEL_VERSION", "unknown"),
    }

@app.get("/db/sample", response_model=LatestTx)
def db_sample():
    eng = sql_engine()
    with eng.connect() as c:
        id_col = _get_id_column(c)
        row = c.exec_driver_sql(
            f"SELECT TOP 1 {_qn(id_col)} FROM {_table_2part(TX_SCHEMA, TX_TABLE)} ORDER BY {_qn(id_col)} DESC"
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail=f"No rows in {TX_SCHEMA}.{TX_TABLE}")
        return LatestTx(tx_id=int(row[0]))

@app.get("/tx/latest", response_model=LatestTx)
def latest_tx():
    return db_sample()

# ---------------- Core Scoring ----------------
def _score_single_tx(tx_id: int) -> ScoreResponse:
    eng = sql_engine()

    # 1) Try the project's original fetch (good if already SA 2.x-safe).
    # 2) If it raises due to result handling (ResourceClosedError/ProgrammingError),
    #    or anything else, fallback to the safe SELECT approach.
    try:
        df = fetch_context(eng, tx_id)
    except LookupError:
        # propagate 'not found' cleanly
        raise
    except (ResourceClosedError, ProgrammingError):
        df = _fetch_context_safe(eng, tx_id)
    except Exception:
        df = _fetch_context_safe(eng, tx_id)

    # Feature engineering (non-training path)
    X, _, meta, _ = engineer(df, is_training=False)

    # Align to training feature set
    for c in FEATS:
        if c not in X.columns:
            X[c] = 0.0
    if FEATS:
        X = X[FEATS].fillna(0.0)

    # Select the row corresponding to this tx_id (fallback to last row)
    row_index = None
    if hasattr(meta, "columns") and "tx_id" in meta.columns:
        hits = meta.index[meta["tx_id"] == tx_id]
        if len(hits) > 0:
            row_index = hits[-1]
    if row_index is None:
        row_index = X.index[-1]

    # Predict
    if hasattr(MODEL, "predict_proba"):
        score = float(MODEL.predict_proba(X)[row_index, 1])
    elif hasattr(MODEL, "decision_function"):
        score = float(MODEL.decision_function(X)[row_index])
    else:
        score = float(MODEL.predict(X)[row_index])

    flagged = bool(score >= THR)

    # Persist score for UI consumption
    try:
        upsert_score(eng, tx_id, score, flagged)
    except (ResourceClosedError, ProgrammingError):
        # Common in SA 2.x when executing non-SELECT statements; ignore result reading
        pass

    # Optional explanations (if available)
    explanation = None
    try:
        from . import explain
        if hasattr(explain, "explain_single"):
            explanation = explain.explain_single(MODEL, X.loc[[row_index]])
    except Exception:
        explanation = None

    return ScoreResponse(tx_id=tx_id, score=score, flagged=flagged, explain=explanation)

@app.post("/score/{tx_id}", response_model=ScoreResponse)
def score_path(tx_id: int):
    try:
        return _score_single_tx(tx_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/score", response_model=ScoreResponse)
def score_body(req: TxRequest):
    try:
        return _score_single_tx(req.tx_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/")
def root():
    return {"message": "VeriPay Scoring API", "version": app.version}
