import os
import traceback
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # load .env before using env vars

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel

import pandas as pd
from sqlalchemy import text, BigInteger, Float, Boolean, NVARCHAR, bindparam
from sqlalchemy.exc import ResourceClosedError, ProgrammingError

from .common import sql_engine
from .features import engineer
from .score_one import load_bundle, fetch_context, upsert_score

app = FastAPI(title="VeriPay Scoring API", version=os.getenv("MODEL_VERSION", "unknown"))

# -------- Config: source transactions table --------
TX_SCHEMA = os.getenv("TX_SCHEMA", "ops")
TX_TABLE  = os.getenv("TX_TABLE",  "Transactions")
SCORE_SCHEMA = os.getenv("SCORE_SCHEMA", "ml")
SCORE_TABLE  = os.getenv("SCORE_TABLE",  "TxScores")

def _qn(name: str) -> str:
    return f"[{name}]"

def _table_2part(schema: str, table: str) -> str:
    return f"{_qn(schema)}.{_qn(table)}"

def _score_tbl() -> str:
    return _table_2part(SCORE_SCHEMA, SCORE_TABLE)

# -------- Request/Response models --------
class ScoreResponse(BaseModel):
    tx_id: int
    score: float
    flagged: bool
    explain: Optional[dict] = None

class TxRequest(BaseModel):
    tx_id: int

class LatestTx(BaseModel):
    tx_id: int

# -------- Model bundle warmup --------
BUNDLE = load_bundle()
MODEL = BUNDLE.get("model")
THR   = float(BUNDLE.get("threshold", os.getenv("THRESHOLD", 0.5)))
FEATS = BUNDLE.get("features") or []

# -------- Error handler (always JSON) --------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    print("UNHANDLED EXCEPTION:", repr(exc))
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )

# -------- Helpers: find an ID column if not named tx_id --------
def _detect_pk_column(conn) -> str | None:
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
    cols = [r[0] for r in conn.exec_driver_sql(f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{TX_SCHEMA}' AND TABLE_NAME = '{TX_TABLE}'
    """).all()]
    for c in ["tx_id", "TxId", "TransactionId", "TransactionID", "TransId", "Id", "ID"]:
        if c in cols:
            return c
    ident = conn.exec_driver_sql(f"""
        SELECT c.name
        FROM sys.columns c
        WHERE c.object_id = OBJECT_ID('{TX_SCHEMA}.{TX_TABLE}')
          AND c.is_identity = 1
        ORDER BY c.column_id
    """).first()
    if ident:
        return ident[0]
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

# -------- Safe context fetch fallback --------
def _fetch_context_safe(eng, tx_id: int) -> pd.DataFrame:
    with eng.connect() as c:
        id_col = _get_id_column(c)
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

# -------- Health & diagnostics --------
@app.get("/")
def root():
    return {"message": "VeriPay Scoring API", "version": app.version}

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

# --- New: prove DB identity the API is using
@app.get("/whoami")
def whoami():
    eng = sql_engine()
    with eng.connect() as c:
        db = c.exec_driver_sql("SELECT DB_NAME()").scalar_one()
        user = c.exec_driver_sql("SELECT SUSER_SNAME()").scalar_one()
    return {"db": db, "user": user, "server": os.getenv("SQL_SERVER")}

# --- New: fetch a score row to verify insert/upsert landed
@app.get("/scores/get/{tx_id}")
def get_score(tx_id: int):
    eng = sql_engine()
    with eng.connect() as c:
        row = c.exec_driver_sql(
            f"SELECT tx_id, score, label_pred, threshold, run_id, explained_at "
            f"FROM {_score_tbl()} WHERE tx_id = ?", (tx_id,)
        ).first()
        if not row:
            return {"found": False}
        cols = ["tx_id", "score", "label_pred", "threshold", "run_id", "explained_at"]
        return {"found": True, **{k: row[i] for i, k in enumerate(cols)}}

# -------- Core Scoring --------
def _score_single_tx(tx_id: int) -> ScoreResponse:
    eng = sql_engine()

    # Prefer project fetch; fall back to safe fetch on SA 2.x result/cursor quirks.
    try:
        df = fetch_context(eng, tx_id)
    except LookupError:
        raise
    except (ResourceClosedError, ProgrammingError):
        df = _fetch_context_safe(eng, tx_id)
    except Exception:
        df = _fetch_context_safe(eng, tx_id)

    # Feature engineering
    X, _, meta, _ = engineer(df, is_training=False)

    # Align to training features
    for c in FEATS:
        if c not in X.columns:
            X[c] = 0.0
    if FEATS:
        X = X[FEATS].fillna(0.0)

    # Pick row for this tx
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

    # Persist score (do NOT swallow errors; let 500 surface if perms/firewall wrong)
    upsert_score(eng, tx_id, score, flagged)

    # Optional explanation
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

@app.post("/diag/write-test")
def diag_write_test(tx_id: int = Query(...)):
    eng = sql_engine()
    score  = 0.123
    flagged = False
    thr    = float(os.getenv("THRESHOLD", "0.5"))
    tbl    = _score_tbl()

    # UPDATE without run_id
    upd_sql = text(f"""
        UPDATE {tbl}
           SET score        = :score,
               label_pred   = :label,
               threshold    = :thr,
               explained_at = SYSUTCDATETIME(),
               reason_json  = COALESCE(reason_json, '{{"source":"model"}}')
         WHERE tx_id = :tx_id;
    """).bindparams(
        bindparam("tx_id",  type_=BigInteger()),
        bindparam("score",  type_=Float()),
        bindparam("label",  type_=Boolean()),
        bindparam("thr",    type_=Float()),
    )

    # INSERT without run_id
    ins_sql = text(f"""
        INSERT INTO {tbl}
            (tx_id, score, label_pred, threshold, explained_at, reason_json)
        VALUES
            (:tx_id, :score, :label, :thr, SYSUTCDATETIME(), '{{"source":"model"}}');
    """).bindparams(
        bindparam("tx_id",  type_=BigInteger()),
        bindparam("score",  type_=Float()),
        bindparam("label",  type_=Boolean()),
        bindparam("thr",    type_=Float()),
    )

    sel_sql = text("""
        SELECT tx_id, score, label_pred
        FROM {tbl}
        WHERE tx_id = :tx_id;
    """.format(tbl=tbl)).bindparams(bindparam("tx_id", type_=BigInteger()))

    params = {
        "tx_id":  int(tx_id),
        "score":  float(score),
        "label":  bool(flagged),
        "thr":    thr,
    }

    with eng.begin() as con:
        upd_res = con.execute(upd_sql, params)
        if upd_res.rowcount == 0:
            con.execute(ins_sql, params)
        row = con.execute(sel_sql, {"tx_id": int(tx_id)}).first()

    if not row:
        return {
            "wrote": False,
            "reason": "Row not visible after write (DB mismatch/permissions).",
            "db_hint": {"api_db": os.getenv("SQL_DB"), "table": f"{SCORE_SCHEMA}.{SCORE_TABLE}", "tx_id": tx_id}
        }

    return {"wrote": True, "row": {"tx_id": int(row[0]), "score": float(row[1]), "flagged": bool(row[2])}}


@app.post("/score", response_model=ScoreResponse)
def score_body(req: TxRequest):
    try:
        return _score_single_tx(req.tx_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
