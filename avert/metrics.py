"""Detection, latency and trust metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y, p):
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def detection_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    both = len(np.unique(y)) == 2
    return dict(
        precision=prec, recall=rec, f1=2 * prec * rec / max(1e-12, prec + rec),
        pr_auc=float(average_precision_score(y, p)) if both else np.nan,
        roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
        fpr=fp / max(1, fp + tn), threshold=thr, n_pos=int(y.sum()), n=len(y),
    )


def episode_latency(y_grid, pred_grid):
    """Per attack episode (contiguous anomalous days of one asset): days from onset to first alarm.

    Returns (episode detection rate, median latency in days over detected episodes, n_episodes).
    """
    lat, n_ep = [], 0
    for y, p in zip(y_grid, pred_grid):
        t, D = 0, len(y)
        while t < D:
            if y[t]:
                e = t
                while e < D and y[e]:
                    e += 1
                n_ep += 1
                hits = np.flatnonzero(p[t:e])
                if hits.size:
                    lat.append(int(hits[0]))
                t = e
            else:
                t += 1
    rate = len(lat) / n_ep if n_ep else np.nan
    return rate, float(np.median(lat)) if lat else np.nan, n_ep


def trust_metrics(T, fraud, claim, compromised, low):
    """T, fraud, claim: [n, d] over the evaluation window; compromised: [n]."""
    honest = ~compromised
    honest_low = float((T[honest] < low).mean()) if honest.any() else np.nan
    ttl, n_att, n_reach = [], 0, 0
    for i in np.flatnonzero(compromised):
        f = np.flatnonzero(fraud[i] & claim[i])
        if f.size == 0:
            continue
        n_att += 1
        lowd = np.flatnonzero(T[i, f[0]:] < low)
        if lowd.size:
            n_reach += 1
            ttl.append(int(lowd[0]))
    fr = fraud & claim
    lg = claim & ~fraud
    return dict(
        honest_low_trust_rate=honest_low,
        attacked_assets=n_att,
        low_trust_reach_rate=n_reach / n_att if n_att else np.nan,
        median_days_to_low=float(np.median(ttl)) if ttl else np.nan,
        mean_trust_on_fraud=float(T[fr].mean()) if fr.any() else np.nan,
        mean_trust_on_legit=float(T[lg].mean()) if lg.any() else np.nan,
    )
