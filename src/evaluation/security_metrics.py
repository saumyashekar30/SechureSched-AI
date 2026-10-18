"""Security metrics. Ground-truth labels are used here for *evaluation only*."""
import numpy as np


def detection_metrics(df, risk_col='risk_score', threshold=0.5):
    """Precision / recall / F1 of the Isolation-Forest flag (risk > threshold) against the
    injected labels.  This is a property of the detector, identical for every scheduler.
    Undefined values (e.g. recall when no attacker exists) are NaN."""
    pred = df[risk_col].values > threshold
    true = df['is_malicious'].values.astype(bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else (0.0 if tp + fn else np.nan)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'flagged_benign_rate': fp / max(int(np.sum(~true)), 1)}


def security_violation_metrics(res):
    """Scheduler-level security outcome from a finished schedule (needs arrival_time, burst_time,
    first_cpu_time, completion_time, wt, is_malicious).

    A malicious process m commits a *harmful priority inversion* ("security violation") if some
    benign process b had already arrived when m first got the CPU, has a strictly shorter actual
    burst, and still completes after m.  Merely running an attacker is not a violation (an attacker
    that is run last, or only when nothing shorter waits, harms nobody); the definition follows
    the threat model: a falsified short declaration that lets a long job jump ahead of shorter benign jobs.
    """
    mal = res['is_malicious'].values.astype(bool)
    out = {'benign_avg_wt': float(res.loc[~mal, 'wt'].mean()) if (~mal).any() else np.nan,
           'malicious_avg_wt': float(res.loc[mal, 'wt'].mean()) if mal.any() else np.nan}
    if not mal.any() or mal.all():
        out.update({'violation_rate': np.nan, 'violations': 0})
        return out
    m, b = res[mal], res[~mal]
    harmed = (
        (b['arrival_time'].values[None, :] <= m['first_cpu_time'].values[:, None])
        & (b['burst_time'].values[None, :] < m['burst_time'].values[:, None])
        & (b['completion_time'].values[None, :] > m['completion_time'].values[:, None])
    ).any(axis=1)
    out.update({'violation_rate': float(harmed.mean()), 'violations': int(harmed.sum())})
    return out
