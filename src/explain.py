import os, joblib, shap
from features import engineer
from extract import get_training_data

def main(model_path=os.getenv("MODEL_PATH","models/model_kaggle.pkl"), out_png="reports/shap_summary.png"):
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
