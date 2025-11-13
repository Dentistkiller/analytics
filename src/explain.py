def explain_single(model, X_row: pd.DataFrame) -> dict:
    """
    Very simple, safe explainer for a single row.
    No SHAP, no extra dependencies.
    Just returns the top numeric features by absolute value.
    """
    # Ensure we have exactly one row
    if X_row.shape[0] != 1:
        X_row = X_row.iloc[[0]]

    row = X_row.iloc[0]

    features = []
    for name, value in row.items():
        # Only try numeric-like values
        try:
            v = float(value)
        except Exception:
            continue

        features.append({
            "feature": name,
            "value": v,
            "abs_value": abs(v),
        })

    # Sort by absolute magnitude and take top 10
    features_sorted = sorted(features, key=lambda f: f["abs_value"], reverse=True)
    top10 = features_sorted[:10]

    return {
        "source": "simple_debug_explainer",
        "top_features": top10
    }
