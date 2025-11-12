import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def pr_auc(y_true, y_score):  return float(average_precision_score(y_true, y_score))
def roc_auc(y_true, y_score): return float(roc_auc_score(y_true, y_score))

def pick_threshold_by_flag_rate(y_score, flag_rate=0.03):
    if len(y_score) == 0: return 0.5
    s = np.sort(y_score); idx = max(0, int(np.floor((1-flag_rate)*len(s))) - 1)
    return float(s[idx])

def confusion_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true==1) & (y_pred==1)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    prec = float(tp / max(1, tp+fp))
    rec  = float(tp / max(1, tp+fn))
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec}

def recall_at_top_k(y_true, y_score, k_rate=0.03):
    n = len(y_score);  k = max(1, int(np.ceil(k_rate*n)))
    idx = np.argsort(-y_score)[:k]
    return float((y_true[idx]==1).sum() / max(1, (y_true==1).sum()))
