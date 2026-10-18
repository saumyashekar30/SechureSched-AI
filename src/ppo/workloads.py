"""Workload construction for PPO training/evaluation.

The committed ``data/processed/processes.csv`` is left untouched.  For training
we cut *additional* workloads out of the same Alibaba ``batch_instance.csv``
prefix the existing preprocessor reads, using the same cleaning and the same
synthetic syscall-rate distribution (N(30, 5) clipped to [10, 50]).  The
trace is cut into disjoint blocks, one window is sampled per block, and windows are split assigned to train / validation / test.
"""
import numpy as np
import pandas as pd

from ..utils.config import BATCH_INSTANCE_PATH
from ..anomaly.generate_anomalies import inject_anomalies
from ..security.risk_model import RiskScoreModel

WORKLOAD_COLUMNS = ['pid', 'arrival_time', 'burst_time', 'declared_burst', 'syscall_rate', 'is_malicious']


def load_trace_windows(cfg):
    """Return a list of clean workload DataFrames (one per trace window)."""
    inst = pd.read_csv(BATCH_INSTANCE_PATH, header=None, nrows=cfg.trace_rows, usecols=[0, 1, 2, 3, 4, 5])
    inst.columns = ['start_time', 'end_time', 'job_id', 'task_id', 'instance_id', 'status']
    inst = inst.dropna(subset=['start_time', 'end_time']).copy()
    inst['burst_time'] = inst['end_time'] - inst['start_time']
    inst = inst[inst['burst_time'] > 0].sort_values('start_time', kind='stable').reset_index(drop=True)

    rs = np.random.RandomState(42)
    syscalls = rs.normal(30.0, 5.0, size=len(inst)).clip(10.0, 50.0)

    windows = []
    w, seg = cfg.window_size, cfg.segment_size
    rng = np.random.RandomState(42)
    for lo in range(0, len(inst) - seg + 1, seg):
        # sample `w` instances from a block of `seg` consecutive arrivals (as the existing
        # preprocessor samples from the trace), keep them in arrival order
        pick = np.sort(rng.choice(seg, size=w, replace=False)) + lo
        chunk = inst.iloc[pick]
        windows.append(pd.DataFrame({
            'pid': np.arange(1, w + 1),
            'arrival_time': np.round((chunk['start_time'].values - chunk['start_time'].min()).astype(float), 2),
            'burst_time': np.round(chunk['burst_time'].values.astype(float), 2),
            'declared_burst': np.round(chunk['burst_time'].values.astype(float), 2),
            'syscall_rate': np.round(syscalls[pick], 2),
            'is_malicious': 0,
        }))
    return windows


def split_windows(windows, cfg, seed=42):
    """Random, disjoint assignment of windows to train / val / test."""
    order = np.random.RandomState(seed).permutation(len(windows))
    a = cfg.n_train_windows
    b = a + cfg.n_val_windows
    c = b + cfg.n_test_windows
    pick = lambda idx: [windows[i] for i in idx]
    return pick(order[:a]), pick(order[a:b]), pick(order[b:c])


def fit_risk_model(train_windows, cfg):
    """Isolation Forest (existing class, existing settings) fitted on clean training windows only."""
    model = RiskScoreModel(contamination=cfg.iforest_contamination, random_state=cfg.iforest_random_state)
    model.fit(pd.concat(train_windows, ignore_index=True))
    return model


def score_risk(risk_model, df):
    """Vectorised version of ``RiskScoreModel.predict_risk`` (identical formula)."""
    feats = df[['declared_burst', 'syscall_rate']].values
    raw = risk_model.model.decision_function(feats)
    return np.clip(1.0 / (1.0 + np.exp(raw * 12.0)), 0.0, 1.0)


def build_episodes(windows, risk_model, rates, cfg, seed=42, vary_length=True):
    """Inject anomalies at every rate into every window and attach risk scores.

    Returns a list of dicts: {'df', 'rate', 'window'}.  Deterministic given ``seed``.
    """
    rng = np.random.RandomState(seed)
    episodes = []
    for wi, win in enumerate(windows):
        n = int(rng.randint(cfg.min_episode_len, len(win) + 1)) if vary_length else len(win)
        base = win.iloc[:n].reset_index(drop=True)
        for rate in rates:
            df = inject_anomalies(base, rate, seed=seed + wi)
            df['is_malicious'] = df['is_malicious'].astype(int)
            df['risk_score'] = score_risk(risk_model, df)
            episodes.append({'df': df, 'rate': rate, 'window': wi})
    return episodes
