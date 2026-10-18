"""Gymnasium environment for security-aware, non-preemptive CPU scheduling.

Scheduling model (mirrors ``src/simulation/cpu_simulator.py`` for FCFS/SJF/SecureSched):
  * single CPU, **non-preemptive**: a decision is taken whenever the CPU becomes
    free and at least one process is ready; the chosen process runs for its full
    *actual* burst.  If the ready queue is empty the clock jumps to the next arrival.
  * the scheduler never sees the actual burst or the ground-truth label; it sees the
    *declared* burst, the observed waiting time and the Isolation-Forest risk score.

MDP (details in docs/PPO.md):
  observation : K candidate slots x [valid, declared_burst, waiting_time, risk] + 5 global features
  action      : Discrete(K) -- index of a candidate slot (invalid slots are masked)
  reward      : negative weighted sum of normalised waiting, aging-weighted delay,
                risk inversion and idle-time costs.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig

SLOT_FEATURES = 4
GLOBAL_FEATURES = 5
_LOG_CAP = np.log1p(100.0)


def _squash(x):
    """Map a non-negative ratio to [0, 1] on a log scale (heavy-tailed trace)."""
    return float(min(np.log1p(max(x, 0.0)) / _LOG_CAP, 1.0))


class SchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episodes, config=None, seed=None):
        """``episodes``: list of dicts with a DataFrame under 'df' (columns of processes.csv
        plus 'risk_score'), or a single such DataFrame."""
        super().__init__()
        self.cfg = config or EnvConfig()
        if hasattr(episodes, "columns"):
            episodes = [{"df": episodes}]
        self.episodes = episodes
        self.K = self.cfg.max_candidates
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.K * SLOT_FEATURES + GLOBAL_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)
        self._seed_hint = seed
        self._loaded = False

    # ------------------------------------------------------------------ setup
    def _load(self, df):
        d = df.reset_index(drop=True)
        order = np.argsort(d['arrival_time'].values, kind='stable')
        d = d.iloc[order].reset_index(drop=True)
        self.df = d
        self.n = len(d)
        self.arrival = d['arrival_time'].values.astype(float)
        self.burst = d['burst_time'].values.astype(float)          # actual; reward/sim only
        self.declared = d['declared_burst'].values.astype(float)
        self.risk = d['risk_score'].values.astype(float)
        # Ground truth is kept for evaluation only and never enters the observation.
        self.is_mal = d['is_malicious'].values.astype(int) if 'is_malicious' in d else np.zeros(self.n, int)
        self._loaded = True

        # Reward normaliser: mean FCFS waiting per process for this workload. It only rescales
        # rewards across episodes of very different scale; it is not part of the observation.
        t, tot = 0.0, 0.0
        for i in range(self.n):
            t = max(t, self.arrival[i])
            tot += t - self.arrival[i]
            t += self.burst[i]
        self.unit = max(tot / self.n, float(np.median(self.burst)), 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "episode" in options:
            ep = options["episode"]
        else:
            idx = options.get("episode_index")
            if idx is None:
                idx = int(self.np_random.integers(len(self.episodes)))
            ep = self.episodes[idx]
        self._load(ep["df"])

        self.ready, self.next_idx = [], 0
        self.time = 0.0
        self.busy, self.idle = 0.0, 0.0
        self.start = np.full(self.n, np.nan)
        self.finish = np.full(self.n, np.nan)
        self.done_count = 0
        self.invalid_actions = 0
        self.order = []
        self._admit()
        if not self.ready:                       # jump to first arrival (idle at t=0 is not counted)
            self.time = self.arrival[self.next_idx]
            self._admit()
        self._cands = self._select_candidates()
        return self._obs(), {"n_processes": self.n}

    # ------------------------------------------------------------------ helpers
    def _admit(self):
        while self.next_idx < self.n and self.arrival[self.next_idx] <= self.time:
            self.ready.append(self.next_idx)
            self.next_idx += 1

    def _ref(self):
        """Running median of declared bursts of processes that have already arrived."""
        if self.next_idx == 0:
            return 1.0
        return max(float(np.median(self.declared[:self.next_idx])), 0.5)

    def _select_candidates(self):
        """Bounded candidate window. If the queue exceeds K: K/4 oldest, K/2 shortest declared
        among unflagged (risk <= threshold), remainder shortest declared overall."""
        r, K = self.ready, self.K
        if len(r) <= K:
            return list(r)
        chosen = list(r[:K // 4])
        taken = set(chosen)
        safe = sorted((i for i in r if i not in taken and self.risk[i] <= self.cfg.risk_threshold),
                      key=lambda i: self.declared[i])
        for i in safe[:K // 2]:
            chosen.append(i)
            taken.add(i)
        rest = sorted((i for i in r if i not in taken), key=lambda i: self.declared[i])
        for i in rest[:K - len(chosen)]:
            chosen.append(i)
        return sorted(chosen)

    def _obs(self):
        ref = self._ref()
        obs = np.zeros(self.K * SLOT_FEATURES + GLOBAL_FEATURES, dtype=np.float32)
        for s, i in enumerate(self._cands):
            o = s * SLOT_FEATURES
            obs[o] = 1.0
            obs[o + 1] = _squash(self.declared[i] / ref)
            obs[o + 2] = _squash((self.time - self.arrival[i]) / ref)
            obs[o + 3] = self.risk[i]
        g = self.K * SLOT_FEATURES
        q = len(self.ready)
        obs[g] = _squash(q / 4.0)                                           # queue length
        obs[g + 1] = 0.0 if q == 0 else max(q - len(self._cands), 0) / q    # share of queue not shown
        obs[g + 2] = float(np.mean(self.risk[self.ready])) if q else 0.0    # mean queue risk
        obs[g + 3] = float(np.mean(self.risk[self.ready] > self.cfg.risk_threshold)) if q else 0.0
        obs[g + 4] = self.busy / self.time if self.time > 0 else 0.0        # CPU utilisation so far
        return np.clip(obs, 0.0, 1.0)

    def action_masks(self):
        m = np.zeros(self.K, dtype=bool)
        m[:len(self._cands)] = True
        return m

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        cfg = self.cfg
        action = int(action)
        invalid = not (0 <= action < len(self._cands))
        if invalid:                              # safe fallback: oldest candidate (FCFS) + penalty
            self.invalid_actions += 1
            action = 0
        p = self._cands[action]
        ref = self._ref()
        start = self.time
        b = self.burst[p]
        fin = start + b

        others = [i for i in self.ready if i != p]
        # exact incremental waiting time caused by this dispatch
        d_wait = b * len(others)
        fair_delay = sum(min(1.0, (start - self.arrival[i]) / (cfg.starvation_horizon * ref)) * b for i in others)
        j = self.next_idx
        while j < self.n and self.arrival[j] < fin:
            d_wait += fin - self.arrival[j]
            j += 1

        n_w = len(others)
        n_safe = sum(1 for i in others if self.risk[i] <= cfg.risk_threshold)
        c_sec = self.risk[p] * n_safe / n_w if (self.risk[p] > cfg.risk_threshold and n_w) else 0.0

        self.ready.remove(p)
        self.start[p], self.finish[p] = start, fin
        self.order.append(int(p))
        self.done_count += 1
        self.busy += b
        self.time = fin
        self._admit()
        idle = 0.0
        if not self.ready and self.next_idx < self.n:
            idle = self.arrival[self.next_idx] - self.time
            self.idle += idle
            self.time = self.arrival[self.next_idx]
            self._admit()

        cap = cfg.reward_clip
        c_wait = min(d_wait / self.unit, cap)
        c_fair = min(fair_delay / self.unit, cap)
        c_idle = min(idle / ref, 10.0) / 10.0
        reward = -(cfg.w_wait * c_wait + cfg.w_fair * c_fair + cfg.w_sec * c_sec + cfg.w_idle * c_idle)
        if invalid:
            reward -= cfg.invalid_action_penalty

        terminated = self.done_count == self.n
        self._cands = self._select_candidates()
        info = {"pid": int(self.df['pid'].iloc[p]), "invalid_action": invalid, "delta_wait": float(d_wait),
                "components": {"wait": c_wait, "fair": c_fair, "sec": c_sec, "idle": c_idle}}
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ output
    def results_df(self):
        """Per-process results in the same schema as ``run_simulation`` (plus risk)."""
        import pandas as pd
        d = self.df.copy()
        d['first_cpu_time'] = self.start
        d['completion_time'] = np.round(self.finish, 2)
        d['tat'] = np.round(d['completion_time'] - d['arrival_time'], 2)
        d['wt'] = np.round(d['tat'] - d['burst_time'], 2)
        return d.iloc[np.argsort(self.finish, kind='stable')].reset_index(drop=True)
