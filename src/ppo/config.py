"""Configuration for the PPO scheduling component (environment, reward, training)."""
import json
from dataclasses import dataclass, asdict, field

from ..utils.config import RISK_LAMBDA, RR_QUANTUM


@dataclass
class EnvConfig:
    # Observation
    max_candidates: int = 16          # K: slots shown to the agent
    risk_threshold: float = 0.5       # same "flagged" threshold as the heuristic scheduler
    # Reward weights (see docs/PPO.md for the formulation)
    w_wait: float = 1.0               # normalised incremental waiting time (also drives turnaround)
    w_fair: float = 0.25              # aging-weighted delay (starvation / fairness)
    w_sec: float = 1.0                # risk-inversion penalty
    w_idle: float = 0.1               # CPU idle time
    starvation_horizon: float = 50.0  # tau, in units of the running median declared burst
    reward_clip: float = 20.0         # per-component clip (magnitude)
    invalid_action_penalty: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 150_000
    learning_rate: float = 3e-4
    n_steps: int = 1024               # rollout length
    batch_size: int = 128
    gamma: float = 0.99
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    n_epochs: int = 10
    net_arch: list = field(default_factory=lambda: [128, 128])
    eval_freq: int = 10_000
    n_eval_episodes: int = 30
    # Workloads
    window_size: int = 50             # instances per workload window
    segment_size: int = 250           # consecutive trace instances each window is sampled from
    min_episode_len: int = 20
    n_train_windows: int = 120
    n_val_windows: int = 15
    n_test_windows: int = 30
    trace_rows: int = 50_000          # same prefix the existing preprocessor reads
    # Existing, preserved experiment settings
    iforest_contamination: float = 0.1
    iforest_random_state: int = 42
    risk_lambda: float = RISK_LAMBDA
    rr_quantum: float = RR_QUANTUM
    env: EnvConfig = field(default_factory=EnvConfig)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        d["env"] = EnvConfig(**d["env"])
        return cls(**d)
