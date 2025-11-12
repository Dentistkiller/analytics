import os, joblib
import pandas as pd
from sqlalchemy import text
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .common import sql_engine
from .extract import get_training_data
from .features import engineer
from .metrics import pr_auc, roc_auc, pick_threshold_by_flag_rate, confusion_at_threshold, recall_at_top_k

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "kaggle_v1")

def _temporal_split(df: pd.DataFrame, cutoff_days=14, fallback_test_size=0.2):
    """
    Parse tx_utc strings -> datetimes, try a time-based split; otherwise fall back
    to stratified 80/20 so we never end up with empty train/test.
    """
    df = df.copy()
    df["tx_utc"] = pd.to_datetime(df["tx_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["tx_utc"]).sort_values("tx_utc")

    if df.empty:
        return df.iloc[0:0].copy(), df.copy()

    tmin, tmax = df["tx_utc"].min(), df["tx_utc"].max()
    span = tmax - tmin

    if span >= pd.Timedelta(days=cutoff_days):
        cutoff = tmax - pd.Timedelta(days=cutoff_days)
        train_df = df[df["tx_utc"] <= cutoff].copy()
        test_df  = df[df["tx_utc"]  > cutoff].copy()
    else:
        train_df = pd.DataFrame(); test_df = pd.DataFrame()

    if train_df.empty or test_df.empty:
        if "label" in df.columns and df["label"].nunique() > 1:
            train_df, test_df = train_test_split(
                df, test_size=fallback_test_size, random_state=42, stratify=df["label"]
            )
        else:
            n_test = max(1, int(len(df) * fallback_test_size))
            test_df = df.iloc[-n_test:].copy()
            train_df = df.iloc[:-n_test].copy()

    if train_df.empty and not test_df.empty:
        train_df = test_df.iloc[:1].copy()
        test_df = test_df.iloc[1:].copy()

    return train_df, test_df

def _write_metrics(run_id: int, metrics: dict):
    eng = sql_engine()
    rows = [{"run_id": run_id, "metric": k, "value": float(v)} for k, v in metrics.items()]
    pd.DataFrame(rows).to_sql("Metrics", eng, schema="ml", if_exists="append", index=False)

def main():
    eng = sql_engine()
    df = get_training_data()
    if df.empty or df["label"].sum() == 0:
        print("Not enough labeled data; ensure Kaggle ingest ran and Labels exist.")
        return

    train_df, test_df = _temporal_split(df, cutoff_days=14, fallback_test_size=0.2)

    # Engineer features
    Xtr, ytr, _, feat_cols = engineer(train_df, is_training=True)
    Xte, yte, _, _         = engineer(test_df,  is_training=True)

    # Guard: if train is empty after engineering, redo split on raw df
    if Xtr.shape[0] == 0 or ytr is None or len(ytr) == 0:
        if df["label"].nunique() > 1:
            tr_df, te_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        else:
            tr_df, te_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
        Xtr, ytr, _, feat_cols = engineer(tr_df, is_training=True)
        Xte, yte, _, _         = engineer(te_df, is_training=True)

    # Baseline: Calibrated Logistic
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")  # ↑ max_iter to curb warnings
    logit_cal = CalibratedClassifierCV(logit, cv=3, method="isotonic")
    logit_cal.fit(Xtr, ytr)
    p_logit = logit_cal.predict_proba(Xte)[:, 1]

    # XGBoost
    pos = max(1, int((ytr == 1).sum()))
    neg = max(1, int((ytr == 0).sum()))
    spw = neg / pos
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric="logloss",
        scale_pos_weight=spw
    )
    xgb.fit(Xtr, ytr)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    # Pick best by PR-AUC  (FIXED unpack)
    score_logit = pr_auc(yte, p_logit)
    score_xgb   = pr_auc(yte, p_xgb)
    if score_xgb >= score_logit:
        chosen_name, chosen_model = "xgb", xgb
        y_score = p_xgb
    else:
        chosen_name, chosen_model = "logit", logit_cal
        y_score = p_logit

    thr = pick_threshold_by_flag_rate(y_score, flag_rate=0.03)
    metrics = {
        "pr_auc": pr_auc(yte, y_score),
        "roc_auc": roc_auc(yte, y_score),
        "recall_at_3pct": recall_at_top_k(yte.values, y_score, 0.03),
        **{f"conf_{k}": v for k, v in confusion_at_threshold(yte.values, y_score, thr).items()}
    }

    # Register a training run & metrics
    with eng.begin() as con:
        run_id = con.execute(text(
            "INSERT INTO ml.Runs(model_version, notes, label_policy) OUTPUT INSERTED.run_id "
            "VALUES(:mv, :notes, :lp)"
        ), {"mv": MODEL_VERSION, "notes": f"pretrain on Kaggle using {chosen_name}", "lp": "kaggle"}).scalar_one()

    _write_metrics(run_id, metrics)

    bundle = {
        "which": chosen_name,
        "model": chosen_model,
        "threshold": float(thr),
        "features": feat_cols,
        "metrics": metrics,
        "model_version": MODEL_VERSION,
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"[OK] Saved {MODEL_PATH} | run_id={run_id} | PR-AUC={metrics['pr_auc']:.3f} thr={thr:.3f}")

if __name__ == "__main__":
    main()
