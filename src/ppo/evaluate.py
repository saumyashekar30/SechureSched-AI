"""Evaluate PPO against FCFS / SJF / RR / SecureSched.

    python -m src.ppo.evaluate --models-dir results/ppo/models --seeds 42 43 44

Experiment A  seed-42 comparison on the committed ``data/processed/processes.csv``
              (identical pipeline to ``main.py``; baselines are checked against results/raw_results.csv).
Experiment B  held-out trace windows (never used for PPO training/validation) x attack rates x
              injection seeds, with mean / std.
Outputs go to results/ppo/ ; ``results/raw_results.csv`` is never written.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import ATTACK_RATES, RESULTS_DIR, RISK_LAMBDA, RR_QUANTUM, PROCESSED_PROCESSES_PATH
from src.anomaly.generate_anomalies import inject_anomalies
from src.security.risk_model import RiskScoreModel
from src.simulation.cpu_simulator import run_simulation
from src.evaluation.metrics import calculate_metrics
from src.evaluation.security_metrics import detection_metrics, security_violation_metrics
from src.ppo.config import TrainConfig
from src.ppo import workloads as W
from src.ppo.inference import PPOScheduler

BASELINES = ["FCFS", "SJF", "RR", "SecureSched"]
OUT_DIR = os.path.join(RESULTS_DIR, "ppo")


def score_schedule(res):
    row = calculate_metrics(res)
    row.update(security_violation_metrics(res))
    return row


def run_all_policies(df_scored, risk_model, ppo_models):
    """Run every policy on the *same* workload. ``ppo_models``: {seed: PPOScheduler}.
    Yields (policy, ppo_seed, results_df)."""
    plain = df_scored.drop(columns=['risk_score'])   # SecureSched computes its own score from the same model
    for pol in BASELINES:
        yield pol, None, run_simulation(plain, policy=pol, risk_model=risk_model,
                                        rr_quantum=RR_QUANTUM, penalty_lambda=RISK_LAMBDA)
    for seed, sched in ppo_models.items():
        yield "PPO", seed, sched.schedule(df_scored)


def load_models(models_dir, seeds, which):
    models = {}
    for s in seeds:
        d = os.path.join(models_dir, f"seed_{s}")
        path = os.path.join(d, "best_model.zip" if which == "best" and os.path.exists(os.path.join(d, "best_model.zip"))
                            else "final_model.zip")
        print(f"loading {path}")
        models[s] = PPOScheduler.load(path, os.path.join(d, "config.json"))
    return models


def experiment_a(ppo_models):
    """Committed workload, original seed-42 protocol (IF fitted on the clean workload itself)."""
    df_clean = pd.read_csv(PROCESSED_PROCESSES_PATH)
    risk_model = RiskScoreModel()                 # contamination 0.1, random_state 42 (class defaults)
    risk_model.fit(df_clean)
    rows = []
    for rate in ATTACK_RATES:
        df = inject_anomalies(df_clean, rate)
        df['risk_score'] = W.score_risk(risk_model, df)
        det = detection_metrics(df)
        for pol, seed, res in run_all_policies(df, risk_model, ppo_models):
            rows.append({'attack_rate': int(rate * 100), 'policy': pol, 'ppo_seed': seed,
                         **score_schedule(res), **det})
    out = pd.DataFrame(rows)
    # Consistency check against the preserved baseline results (read-only).
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))
    chk = out[out.policy.isin(BASELINES)].merge(ref, on=['attack_rate', 'policy'], suffixes=('', '_ref'))
    ok = bool(np.allclose(chk.avg_wt, chk.avg_wt_ref) and np.allclose(chk.avg_tat, chk.avg_tat_ref)
              and np.allclose(chk.jain_fairness, chk.jain_fairness_ref))
    print(f"baselines reproduce results/raw_results.csv exactly: {ok}")
    return out, ok


def experiment_b(ppo_models, cfg, inj_seeds):
    windows = W.load_trace_windows(cfg)
    train_w, _, test_w = W.split_windows(windows, cfg, seed=42)
    risk_model = W.fit_risk_model(train_w, cfg)
    rows = []
    for iseed in inj_seeds:
        eps = W.build_episodes(test_w, risk_model, ATTACK_RATES, cfg, seed=iseed, vary_length=False)
        for ep in eps:
            det = detection_metrics(ep['df'])
            for pol, seed, res in run_all_policies(ep['df'], risk_model, ppo_models):
                rows.append({'attack_rate': int(round(ep['rate'] * 100)), 'policy': pol, 'ppo_seed': seed,
                             'window': ep['window'], 'inj_seed': iseed, **score_schedule(res), **det})
    return pd.DataFrame(rows)


METRICS = ['avg_wt', 'avg_tat', 'jain_fairness', 'benign_avg_wt', 'violation_rate']


def summarize(df):
    g = df.groupby(['attack_rate', 'policy'])
    cols = METRICS + ['precision', 'recall', 'f1']
    agg = g[cols].agg(['mean', 'std'])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def paired_vs(df, a, b, metric='avg_wt'):
    """Paired per-episode comparison of policy ``a`` against ``b`` (PPO averaged over training seeds)."""
    keys = ['attack_rate', 'window', 'inj_seed']
    x = df[df.policy == a].groupby(keys)[metric].mean()
    y = df[df.policy == b].groupby(keys)[metric].mean()
    d = (x - y).dropna()
    return pd.DataFrame({'attack_rate': d.groupby(level=0).mean().index,
                         f'mean_diff_{a}_minus_{b}': d.groupby(level=0).mean().values,
                         'sem': d.groupby(level=0).sem().values,
                         f'frac_episodes_{a}_better': d.groupby(level=0).apply(lambda s: float((s < 0).mean())).values})


def plot_comparison(summary, path, title, metrics=METRICS, with_err=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {'avg_wt': 'Average waiting time (time units)', 'avg_tat': 'Average turnaround time (time units)',
              'jain_fairness': "Jain's fairness index (waiting time)", 'benign_avg_wt': 'Benign-process avg. waiting time',
              'violation_rate': 'Security-violation rate (malicious)'}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for pol, g in summary.groupby('policy'):
            g = g.sort_values('attack_rate')
            if m == 'violation_rate' and g[f'{m}_mean'].isna().all():
                continue
            err = g[f'{m}_std'].fillna(0).values if with_err and f'{m}_std' in g else None
            ax.errorbar(g['attack_rate'], g[f'{m}_mean'], yerr=err, marker='o', capsize=3, label=pol)
        ax.set_xlabel('Injected malicious tasks (%)')
        ax.set_ylabel(labels[m])
        ax.grid(True, linestyle='--', alpha=0.5)
    np.atleast_1d(axes)[0].legend(title='Policy')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(models_dir, seeds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in seeds:
        p = os.path.join(models_dir, f"seed_{s}", "training_log.csv")
        if not os.path.exists(p):
            continue
        log = pd.read_csv(p)
        ax.plot(log['timesteps'], log['episode_reward'].rolling(100, min_periods=10).mean(), label=f"seed {s}")
    ax.set_xlabel('Training timesteps (scheduling decisions)')
    ax.set_ylabel('Episode return (rolling mean, 100 episodes)')
    ax.set_title('PPO training reward curve')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Training seed')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.join(OUT_DIR, "models"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--inj-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--which", choices=["best", "final"], default="best")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ppo_models = load_models(args.models_dir, args.seeds, args.which)
    cfg = TrainConfig.from_json(os.path.join(args.models_dir, f"seed_{args.seeds[0]}", "config.json"))

    a, ok = experiment_a(ppo_models)
    a.to_csv(os.path.join(args.out_dir, "expA_seed42_processes_raw.csv"), index=False)
    a_sum = summarize(a)
    a_sum.to_csv(os.path.join(args.out_dir, "expA_seed42_processes_summary.csv"), index=False)
    plot_comparison(a_sum, os.path.join(args.out_dir, "expA_comparison.png"),
                    "Exp. A: committed 20-process workload, seed 42 (PPO: mean of training seeds)",
                    metrics=['avg_wt', 'avg_tat', 'jain_fairness', 'violation_rate'], with_err=False)

    b = experiment_b(ppo_models, cfg, args.inj_seeds)
    b.to_csv(os.path.join(args.out_dir, "expB_heldout_raw.csv"), index=False)
    b_sum = summarize(b)
    b_sum.to_csv(os.path.join(args.out_dir, "expB_heldout_summary.csv"), index=False)
    for other in ["SecureSched", "SJF"]:
        paired_vs(b, "PPO", other).to_csv(os.path.join(args.out_dir, f"expB_paired_PPO_vs_{other}.csv"), index=False)
    # variability across PPO training seeds
    (b[b.policy == "PPO"].groupby(['attack_rate', 'ppo_seed'])[METRICS].mean().reset_index()
        .to_csv(os.path.join(args.out_dir, "expB_ppo_by_training_seed.csv"), index=False))
    plot_comparison(b_sum, os.path.join(args.out_dir, "expB_comparison.png"),
                    "Exp. B: held-out windows (mean ± std over windows, injection seeds, PPO training seeds)")
    plot_training(args.models_dir, args.seeds, os.path.join(args.out_dir, "training_reward_curve.png"))

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n=== Exp. A (seed 42, committed workload; PPO = mean over training seeds) ===")
    print(a_sum[['attack_rate', 'policy', 'avg_wt_mean', 'avg_tat_mean', 'jain_fairness_mean', 'violation_rate_mean',
                 'precision_mean', 'recall_mean', 'f1_mean']].round(3).to_string(index=False))
    print("\n=== Exp. B (held-out windows) ===")
    print(b_sum[['attack_rate', 'policy', 'avg_wt_mean', 'avg_wt_std', 'avg_tat_mean', 'jain_fairness_mean',
                 'benign_avg_wt_mean', 'violation_rate_mean', 'precision_mean', 'recall_mean', 'f1_mean']].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
