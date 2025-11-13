import os
import joblib
import shap
import pandas as pd

from .features import engineer
from .extract import get_training_data

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_kaggle.pkl")

# Cache SHAP explainer so we don't rebuild it on every request
_explainer = None
_background_cols = None


def _init_explainer(model):
    """
    Lazy initialisation of a SHAP explainer using training data as background.
    """
    global _explainer, _background_cols
    if _explainer is not None:
        return

    # Load training data and engineer features
    df = get_training_data()
    X, y, _, _ = engineer(df, is_training=True)

    # Use a sample as background to keep it fast-ish
    background = X.sample(min(2000, len(X)), random_state=42)
    _background_cols = background.columns

    # Let SHAP pick the right explainer for the model (Tree, Kernel, etc.)
    _explainer = shap.Explainer(model, background)


def explain_single(model, X_row: pd.DataFrame) -> dict:
    """
    Explain a single transaction.

    Parameters
    ----------
    model : trained model (same object used for scoring)
    X_row : pandas DataFrame with a SINGLE row of model-ready features
            (this is what the API passes: X.loc[[row_index]])

    Returns
    -------
    dict
        JSON-serializable explanation structure.
    """
    # Make sure explainer is initialised
    _init_explainer(model)

    # Align columns of X_row to the explainer's background columns
    X_use = X_row.copy()
    for col in _background_cols:
        if col not in X_use.columns:
            X_use[col] = 0.0
    X_use = X_use[_background_cols]

    # Get SHAP values
    sv = _explainer(X_use)

    # Single row → take index 0
    values = sv.values[0]
    base_value = float(getattr(sv, "base_values", [0.0])[0])

    # Build a nice ranked list of features
    feats = []
    for f, v in zip(_background_cols, values):
        val = float(v)
        feats.append({
            "feature": f,
            "shap_value": val,
            "abs_shap": abs(val),
        })

    # Sort by absolute impact and keep top 10
    feats_sorted = sorted(feats, key=lambda d: d["abs_shap"], reverse=True)
    top10 = feats_sorted[:10]

    return {
        "source": "shap",
        "base_value": base_value,
        "top_features": top10,
    }


# --- Keep your original script behaviour as CLI for beeswarm summary ---
def main(model_path: str = MODEL_PATH, out_png: str = "reports/shap_summary.png"):
    """
    Offline script: generate a SHAP beeswarm plot for training data.
    """
    bundle = joblib.load(model_path)
    model = bundle["model"]
    df = get_training_data()
    X, y, _, _ = engineer(df, is_training=True)

    explainer = shap.Explainer(model, X)
    sv = explainer(X.sample(min(2000, len(X)), random_state=42))
    shap.plots.beeswarm(sv, show=False)

    import matplotlib.pyplot as plt
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved {out_png}")


if __name__ == "__main__":
    main()
