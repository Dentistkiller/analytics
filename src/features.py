import pandas as pd
import numpy as np

CATEGORICAL_COLS = ["channel", "entry_mode", "currency"]

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # tx_utc is stored as string in SQL; parse locally for features
    # Drop rows that fail to parse to keep rolling ops stable
    df["tx_utc"] = pd.to_datetime(df["tx_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["tx_utc"])

    # numeric amount
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)

    # normalize categoricals
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    return df

def _encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            dummies = pd.get_dummies(df[c], prefix=c, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
    return df

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tx_hour"] = df["tx_utc"].dt.hour
    df["tx_dow"]  = df["tx_utc"].dt.dayofweek
    df["is_weekend"] = df["tx_dow"].isin([5, 6]).astype(int)
    return df

def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    # NumPy 2.x: use positional clip, not lower=
    df["amount_log"] = np.log1p(df["amount"].clip(0))
    return df

def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Per-card rolling windows (lowercase 'h'/'d' to avoid FutureWarning)
    df = df.sort_values(["card_id", "tx_utc"])
    for window, name in [("24h", "24h"), ("7d", "7d")]:
        g = df.groupby("card_id").rolling(window=window, on="tx_utc")
        # counts exclude current row
        df[f"card_tx_count_{name}"] = g["tx_id"].count().values - 1
        # sum excluding current amount, clipped at 0
        df[f"card_amt_sum_{name}"] = (g["amount"].sum().values - df["amount"].values).clip(0)

    # Per-customer rolling
    df = df.sort_values(["customer_id", "tx_utc"])
    for window, name in [("24h", "24h"), ("7d", "7d")]:
        g = df.groupby("customer_id").rolling(window=window, on="tx_utc")
        df[f"cust_tx_count_{name}"] = g["tx_id"].count().values - 1
        df[f"cust_amt_sum_{name}"]  = (g["amount"].sum().values - df["amount"].values).clip(0)

    # Novelty flags
    df["is_new_card_merchant"] = (
        df.groupby(["card_id", "merchant_id"]).cumcount().eq(0).astype(int)
    )
    df["is_new_customer_merchant"] = (
        df.groupby(["customer_id", "merchant_id"]).cumcount().eq(0).astype(int)
    )
    return df

def engineer(df: pd.DataFrame, is_training: bool = True):
    df = _prep(df)
    df = _time_features(df)
    df = _amount_features(df)
    df = _velocity_features(df)
    df = _encode_cats(df)

    # keep meta for writing scores later
    meta = df[["tx_id", "tx_utc", "customer_id", "card_id", "merchant_id"]].copy()

    num_cols = [
        "amount", "amount_log", "tx_hour", "tx_dow", "is_weekend",
        "card_tx_count_24h", "card_amt_sum_24h", "card_tx_count_7d", "card_amt_sum_7d",
        "cust_tx_count_24h", "cust_amt_sum_24h", "cust_tx_count_7d", "cust_amt_sum_7d",
        "is_new_card_merchant", "is_new_customer_merchant"
    ]
    cat_cols = [c for c in df.columns if c.startswith(tuple(f"{x}_" for x in CATEGORICAL_COLS))]
    feat_cols = [c for c in (num_cols + cat_cols) if c in df.columns]

    X = df[feat_cols].fillna(0.0)
    y = df["label"].astype(int) if ("label" in df.columns and is_training) else None
    return X, y, meta, feat_cols
