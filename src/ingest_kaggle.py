import os, os.path, zipfile, io
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from kaggle.api.kaggle_api_extended import KaggleApi
from common import sql_engine
import pyodbc  # ensure driver is installed (ODBC Driver 17/18)

# Kaggle dataset
DATASET = "mlg-ulb/creditcardfraud"        # provides creditcard.csv
MAX_ROWS = int(os.getenv("KAGGLE_MAX_ROWS", "100000"))

# We will store *all* timestamps as strings, not datetime dtypes.
# Use a SQL-friendly ISO-like format (no 'T', no 'Z').
TS_FMT = "%Y-%m-%d %H:%M:%S"               # 19 chars, e.g., "2025-01-01 11:31:45"


# ----------------------------- helpers -------------------------------------- #

def insert_with_identity(sa_conn, df, schema, table, cols):
    """
    Bulk insert rows where the PK is an IDENTITY and we want to set it explicitly.
    Uses the same DBAPI connection as the current SQLAlchemy transaction.
    """
    if df.empty:
        return

    full = f"{schema}.{table}"
    collist = ",".join(cols)
    placeholders = ",".join(["?"] * len(cols))
    data = list(map(tuple, df[cols].itertuples(index=False, name=None)))

    # Access the live DBAPI connection from SQLAlchemy Connection (2.x)
    dbapi_conn = sa_conn.connection.driver_connection
    cur = dbapi_conn.cursor()
    cur.fast_executemany = True
    try:
        cur.execute(f"SET IDENTITY_INSERT {full} ON;")
        cur.executemany(f"INSERT INTO {full} ({collist}) VALUES ({placeholders})", data)
    finally:
        try:
            cur.execute(f"SET IDENTITY_INSERT {full} OFF;")
        finally:
            cur.close()


def _download_csv() -> pd.DataFrame:
    """
    Download creditcard.csv via Kaggle API. Falls back to full zip if single-file fails.
    """
    api = KaggleApi(); api.authenticate()
    try:
        content = api.dataset_download_file(DATASET, "creditcard.csv")
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open("creditcard.csv") as f:
                return pd.read_csv(f)
    except Exception:
        tmp_dir = "kaggle_dl"
        os.makedirs(tmp_dir, exist_ok=True)
        api.dataset_download_files(DATASET, path=tmp_dir, quiet=False, unzip=True)
        return pd.read_csv(os.path.join(tmp_dir, "creditcard.csv"))


def _synthesize_ops(df: pd.DataFrame):
    """
    Build small dimension tables and a transactions frame using *string* timestamps.
    Returns dataframes: customers, cards, merchants, tx, labels
    """
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

    # Build tx_utc as *string* (no datetime dtype in pandas)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df["tx_utc"] = [
        (base + timedelta(seconds=float(s))).strftime(TS_FMT)
        for s in df["Time"].tolist()
    ]

    channels = ["POS", "ecom", "in-app"]
    entries  = ["chip", "swipe", "keyed"]

    num_customers = 5000
    num_cards     = 6000
    num_merchants = 120
    np.random.seed(42)

    df["customer_id"] = np.random.randint(1, num_customers+1, size=len(df))
    df["card_id"]     = np.random.randint(1, num_cards+1, size=len(df))
    df["merchant_id"] = np.random.randint(1, num_merchants+1, size=len(df))
    df["amount"]      = df["Amount"].round(2)
    df["currency"]    = "ZAR"
    df["channel"]     = np.random.choice(channels, size=len(df), p=[0.55, 0.35, 0.10])
    df["entry_mode"]  = np.random.choice(entries,  size=len(df), p=[0.60, 0.25, 0.15])
    df["status"]      = "Pending"

    customers = pd.DataFrame({
        "customer_id": np.arange(1, num_customers+1, dtype=int),
        "name": [f"Customer {i}" for i in range(1, num_customers+1)],
        "email_hash": None,
        "phone_hash": None
    })

    cards = pd.DataFrame({
        "card_id": np.arange(1, num_cards+1, dtype=int),
        "customer_id": np.random.randint(1, num_customers+1, size=num_cards),
        "network": np.random.choice(["Visa", "MC"], size=num_cards),
        "last4": [f"{np.random.randint(0, 9999):04d}" for _ in range(num_cards)],
        "issue_country": "ZA"
    })

    merchants = pd.DataFrame({
        "merchant_id": np.arange(1, num_merchants+1, dtype=int),
        "name": [f"Merchant {i}" for i in range(1, num_merchants+1)],
        "category": np.random.choice(["Retail", "Electronics", "Grocery", "Fuel", "Online"], size=num_merchants),
        "country": "ZA",
        "risk_level": np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1], size=num_merchants)
    })

    # Transactions (tx_utc stays as *string*)
    tx = df[[
        "customer_id", "card_id", "merchant_id", "amount",
        "currency", "tx_utc", "entry_mode", "channel", "status"
    ]].copy()

    # Labels skeleton (we map to tx_id after inserting transactions)
    labels = pd.DataFrame({
        "row_idx": np.arange(len(df), dtype=int),
        "label": df["Class"].astype(int),
    })

    return customers, cards, merchants, tx, labels


def _truncate(sa_conn):
    sa_conn.execute(text("DELETE FROM ml.TxScores;"))
    sa_conn.execute(text("DELETE FROM ml.Labels;"))
    sa_conn.execute(text("DELETE FROM ops.Transactions;"))
    sa_conn.execute(text("DELETE FROM ops.Cards;"))
    sa_conn.execute(text("DELETE FROM ops.Customers;"))
    sa_conn.execute(text("DELETE FROM ops.Merchants;"))


# ------------------------------- main --------------------------------------- #

def main(reset=True):
    eng = sql_engine()
    print("Downloading Kaggle creditcard.csv ...")
    df = _download_csv()
    print(f"Kaggle rows: {len(df):,}")

    customers, cards, merchants, tx, labels = _synthesize_ops(df)

    # 1) Insert dimensions and transactions
    with eng.begin() as con:
        if reset:
            _truncate(con)

        # Dimensions: explicit identity values
        insert_with_identity(
            con, customers, "ops", "Customers",
            ["customer_id", "name", "email_hash", "phone_hash"]
        )
        insert_with_identity(
            con, merchants, "ops", "Merchants",
            ["merchant_id", "name", "category", "country", "risk_level"]
        )
        insert_with_identity(
            con, cards, "ops", "Cards",
            ["card_id", "customer_id", "network", "last4", "issue_country"]
        )

        # Transactions: let SQL generate tx_id (IDENTITY). tx_utc is a *string*.
        tx.to_sql("Transactions", con, schema="ops", if_exists="append", index=False)

    # 2) Fetch tx_id by chronological order and align with Kaggle rows
    with eng.connect() as con:
        tx_db = pd.read_sql(
            "SELECT tx_id, tx_utc FROM ops.Transactions ORDER BY tx_utc, tx_id",
            con
        )

    labels = labels.sort_values("row_idx").reset_index(drop=True)
    tx_db  = tx_db.reset_index(drop=True)
    n = min(len(labels), len(tx_db))
    labels = labels.iloc[:n].copy()
    tx_db  = tx_db.iloc[:n].copy()

    # Map labels -> tx_id
    labels["tx_id"] = tx_db["tx_id"].values

    # 3) Build final Labels frame (all strings)
    labels = labels[["tx_id", "label"]].copy()
    labels["labeled_at"] = datetime.utcnow().strftime(TS_FMT)  # keep as string
    labels["source"] = "kaggle"
    labels = labels[["tx_id", "label", "labeled_at", "source"]]

    # 4) Insert Labels once
    labels.to_sql("Labels", eng, schema="ml", if_exists="append", index=False)

    print(
        "Loaded -> "
        f"Customers={len(customers):,}  Cards={len(cards):,}  Merchants={len(merchants):,}  "
        f"Transactions={len(tx):,}  Labels={len(labels):,}"
    )


if __name__ == "__main__":
    main()
