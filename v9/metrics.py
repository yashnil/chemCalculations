import numpy as np
from scipy.spatial.distance import jensenshannon

def _topk_idx(arr, k):          # helper
    return np.argsort(arr, axis=1)[:, -k:]

def precision_at_k(y_true, y_pred, k=10):
    t_idx = _topk_idx(y_true, k); p_idx = _topk_idx(y_pred, k)
    hits  = [(len(set(t)&set(p))/k) for t, p in zip(t_idx, p_idx)]
    return float(np.mean(hits))

def mae_topk(y_true, y_pred, k=10):
    idx = _topk_idx(y_true, k)
    err = np.take_along_axis(np.abs(y_true - y_pred), idx, 1)
    return float(err.mean())

def log_mae_topk(y_true, y_pred, k=10, eps=1e-12):
    idx = _topk_idx(y_true, k)
    lt  = np.take_along_axis(np.log10(y_true+eps),  idx, 1)
    lp  = np.take_along_axis(np.log10(y_pred+eps),  idx, 1)
    return float(np.abs(lt - lp).mean())

def js_topk(y_true, y_pred, k=10, eps=1e-12):
    idx  = _topk_idx(y_true, k)
    jt   = np.take_along_axis(y_true, idx, 1)
    jp   = np.take_along_axis(y_pred, idx, 1)
    jt  /= jt.sum(axis=1, keepdims=True);  jp /= jp.sum(axis=1, keepdims=True)
    return float(np.mean([jensenshannon(a, b, 2.) for a, b in zip(jt, jp)]))
