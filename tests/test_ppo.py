import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.anomaly.generate_anomalies import inject_anomalies
from src.evaluation.metrics import calculate_metrics
from src.evaluation.security_metrics import detection_metrics, security_violation_metrics
from src.ppo import workloads as W
from src.ppo.config import EnvConfig, TrainConfig
from src.ppo.inference import run_chooser
from src.ppo.sched_env import SchedulingEnv
from src.security.risk_model import RiskScoreModel
from src.simulation.cpu_simulator import run_simulation
from src.utils.config import PROCESSED_PROCESSES_PATH, RESULTS_DIR


def synthetic(n=40, seed=0, rate=0.2):
    rs = np.random.RandomState(seed)
    df = pd.DataFrame({
        'pid': np.arange(1, n + 1),
        'arrival_time': np.sort(rs.randint(0, 60, n)).astype(float),
        'burst_time': np.round(rs.exponential(8, n) + 1, 2),
        'syscall_rate': np.round(rs.normal(30, 5, n).clip(10, 50), 2),
        'is_malicious': 0,
    })
    df['declared_burst'] = df['burst_time']
    return df, rate


@pytest.fixture(scope="module")
def pool():
    base, _ = synthetic(60, 1, 0)
    rm = RiskScoreModel(contamination=0.1, random_state=42)
    rm.fit(base)
    eps = []
    for s in range(6):
        for rate in (0.0, 0.1, 0.2):
            df, _ = synthetic(40, s)
            df = inject_anomalies(df, rate, seed=s)
            df['risk_score'] = W.score_risk(rm, df)
            eps.append({'df': df, 'rate': rate})
    return eps, rm


def rollout_random(env, rng):
    obs, _ = env.reset()
    steps, ret = 0, 0.0
    while True:
        assert np.isfinite(obs).all() and obs.min() >= 0 and obs.max() <= 1
        mask = env.action_masks()
        assert mask.any()
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert np.isfinite(r)
        steps += 1
        ret += r
        if term or trunc:
            return steps, ret


def test_gymnasium_api(pool):
    from gymnasium.utils.env_checker import check_env
    check_env(SchedulingEnv(pool[0]), skip_render_check=True)


def test_random_rollouts_are_valid_schedules(pool):
    eps, _ = pool
    env = SchedulingEnv(eps)
    rng = np.random.RandomState(0)
    for _ in range(30):
        steps, _ = rollout_random(env, rng)
        assert steps == env.n                              # terminates, one decision per process
        res = env.results_df()
        assert len(res) == env.n and sorted(res['pid']) == sorted(env.df['pid'])    # nothing lost/duplicated
        assert np.allclose(res['completion_time'] - res['first_cpu_time'], res['burst_time'], atol=0.01)  # exact burst
        assert (res['first_cpu_time'] >= res['arrival_time'] - 1e-9).all()
        assert np.allclose(res['wt'], res['tat'] - res['burst_time'], atol=0.011)
        assert np.allclose(res['tat'], res['completion_time'] - res['arrival_time'], atol=0.011)
        assert (res['wt'] >= -0.011).all()
        s = res.sort_values('first_cpu_time')               # non-preemptive: no overlap on the single CPU
        assert (s['first_cpu_time'].values[1:] >= s['completion_time'].values[:-1] - 0.011).all()
        assert env.invalid_actions == 0


def test_invalid_action_is_safe(pool):
    env = SchedulingEnv(pool[0])
    env.reset(options={'episode_index': 0})
    _, r, term, _, info = env.step(env.K - 1 if not env.action_masks()[-1] else 99)
    assert info['invalid_action'] and np.isfinite(r) and env.invalid_actions == 1


def test_candidate_window_bounded_and_masked(pool):
    cfg = EnvConfig(max_candidates=4)
    env = SchedulingEnv(pool[0], config=cfg)
    env.reset(options={'episode_index': 3})
    assert env.action_masks().sum() == min(4, len(env.ready))
    assert env.observation_space.contains(env._obs())


def test_wait_reward_accounts_exact_waiting(pool):
    env = SchedulingEnv(pool[0])
    env.reset(options={'episode_index': 2})
    total, done = 0.0, False
    while not done:
        _, _, done, _, info = env.step(0)
        total += info['delta_wait']
    assert total == pytest.approx(env.results_df()['wt'].sum(), rel=1e-6, abs=0.1)


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_matches_existing_fcfs_and_sjf(rate):
    df = pd.read_csv(PROCESSED_PROCESSES_PATH)
    rm = RiskScoreModel()
    rm.fit(df)
    d = inject_anomalies(df, rate)
    d['risk_score'] = W.score_risk(rm, d)
    big = EnvConfig(max_candidates=64)
    choosers = {'FCFS': lambda e: 0,
                'SJF': lambda e: int(np.argmin([e.declared[i] for i in e._cands]))}
    for pol, ch in choosers.items():
        ref = calculate_metrics(run_simulation(d.drop(columns='risk_score'), policy=pol, risk_model=rm))
        got = calculate_metrics(run_chooser(d, ch, big))
        assert ref == got


def test_labels_are_not_observed(pool):
    eps, _ = pool
    ep = eps[-1]
    flipped = ep['df'].copy()
    flipped['is_malicious'] = 1 - flipped['is_malicious']
    a, b = SchedulingEnv([ep]), SchedulingEnv([{'df': flipped}])
    oa, _ = a.reset(options={'episode_index': 0})
    ob, _ = b.reset(options={'episode_index': 0})
    assert np.array_equal(oa, ob)
    for _ in range(10):
        oa, ra, *_ = a.step(0)
        ob, rb, *_ = b.step(0)
        assert np.array_equal(oa, ob) and ra == rb


def test_vectorised_risk_matches_predict_risk(pool):
    eps, rm = pool
    df = eps[-1]['df'].head(8)
    expected = [rm.predict_risk(r.declared_burst, r.syscall_rate) for r in df.itertuples()]
    assert np.allclose(W.score_risk(rm, df), expected)


def test_security_metrics_definition():
    res = pd.DataFrame({
        'arrival_time': [0, 0, 0], 'burst_time': [10.0, 2.0, 3.0], 'is_malicious': [1, 0, 0],
        'first_cpu_time': [0.0, 10.0, 12.0], 'completion_time': [10.0, 12.0, 15.0], 'wt': [0.0, 10.0, 12.0]})
    m = security_violation_metrics(res)
    assert m['violation_rate'] == 1.0 and m['violations'] == 1
    res2 = res.copy()   # attacker last: harmless
    res2['first_cpu_time'], res2['completion_time'] = [5.0, 0.0, 2.0], [15.0, 2.0, 5.0]
    assert security_violation_metrics(res2)['violation_rate'] == 0.0
    d = detection_metrics(pd.DataFrame({'risk_score': [.9, .9, .1, .1], 'is_malicious': [1, 0, 1, 0]}))
    assert d['precision'] == 0.5 and d['recall'] == 0.5 and d['f1'] == 0.5


def test_existing_baselines_unchanged():
    ref = pd.read_csv(os.path.join(RESULTS_DIR, 'raw_results.csv'))
    df = pd.read_csv(PROCESSED_PROCESSES_PATH)
    for rate in (0.0, 0.2):
        d = inject_anomalies(df, rate)
        rm = RiskScoreModel()
        rm.fit(df)
        for pol in ('FCFS', 'SJF', 'RR', 'SecureSched'):
            m = calculate_metrics(run_simulation(d, policy=pol, risk_model=rm, rr_quantum=5.0, penalty_lambda=500.0))
            row = ref[(ref.attack_rate == int(rate * 100)) & (ref.policy == pol)].iloc[0]
            assert (m['avg_wt'], m['avg_tat'], m['jain_fairness']) == (row.avg_wt, row.avg_tat, row.jain_fairness)


def test_ppo_save_load_and_heldout_evaluation(pool, tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from src.ppo.inference import PPOScheduler
    eps, _ = pool
    cfg = TrainConfig()
    env = ActionMasker(SchedulingEnv(eps[:12]), lambda e: e.unwrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=128, batch_size=64, seed=0, device="cpu")
    model.learn(256)
    path = str(tmp_path / "m")
    model.save(path)
    cfg.to_json(str(tmp_path / "config.json"))
    assert TrainConfig.from_json(str(tmp_path / "config.json")) == cfg
    sched = PPOScheduler.load(path + ".zip", str(tmp_path / "config.json"))
    held_out = eps[12][ 'df']                                   # not in the 12 training episodes
    res = sched.schedule(held_out)
    assert len(res) == len(held_out) and np.isfinite(res[['wt', 'tat']].values).all()
    obs, _ = SchedulingEnv(held_out).reset()
    mask = np.ones(cfg.env.max_candidates, dtype=bool)
    a1, _ = model.predict(obs, action_masks=mask, deterministic=True)
    a2, _ = sched.model.predict(obs, action_masks=mask, deterministic=True)
    assert int(a1) == int(a2)
