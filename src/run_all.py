import subprocess, sys

steps = [
    ["python", "src/ingest_kaggle.py"],   # 1) load Kaggle into SQL (ops + ml.Labels)
    ["python", "src/train_kaggle.py"],    # 2) train and save models/model_kaggle.pkl
    ["python", "src/score.py"],           # 3) score any unscored tx (including freshly loaded)
]

for cmd in steps:
    print("\n==> Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)

print("\nAll done. Open your app: /Dashboard and /Transactions.")
