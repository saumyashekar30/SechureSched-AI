"""Train a Maskable PPO scheduling policy.

    python -m src.ppo.train --timesteps 150000 --seed 42 --out-dir results/ppo/models/seed_42
"""
import argparse
import csv
import os
import random
import sys

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import ATTACK_RATES, RESULTS_DIR
from src.ppo.config import TrainConfig
from src.ppo import workloads as W
from src.ppo.sched_env import SchedulingEnv


def set_global_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(episodes, cfg, seed):
    from gymnasium.wrappers import TimeLimit  # noqa: F401  (episodes terminate on their own)
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    env = SchedulingEnv(episodes, config=cfg.env)
    env = Monitor(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    env.reset(seed=seed)
    return env


def prepare_data(cfg):
    """Windows -> split -> Isolation Forest fitted on clean *training* windows -> episode pools."""
    windows = W.load_trace_windows(cfg)
    train_w, val_w, test_w = W.split_windows(windows, cfg, seed=42)
    risk_model = W.fit_risk_model(train_w, cfg)
    return risk_model, train_w, val_w, test_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--n-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--clip-range", type=float)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    cfg = TrainConfig()
    for k, v in [("total_timesteps", args.timesteps), ("seed", args.seed), ("learning_rate", args.learning_rate),
                 ("n_steps", args.n_steps), ("batch_size", args.batch_size), ("gamma", args.gamma),
                 ("clip_range", args.clip_range)]:
        if v is not None:
            setattr(cfg, k, v)
    out = args.out_dir or os.path.join(RESULTS_DIR, "ppo", "models", f"seed_{cfg.seed}")
    os.makedirs(out, exist_ok=True)
    set_global_seed(cfg.seed)

    risk_model, train_w, val_w, _ = prepare_data(cfg)
    train_eps = W.build_episodes(train_w, risk_model, ATTACK_RATES, cfg, seed=cfg.seed)
    val_eps = W.build_episodes(val_w, risk_model, ATTACK_RATES, cfg, seed=1000)
    print(f"train episodes: {len(train_eps)}  val episodes: {len(val_eps)}")

    class RewardLogger(BaseCallback):
        def __init__(self, path):
            super().__init__()
            self.path = path
            self.f = None

        def _on_training_start(self):
            self.f = open(self.path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["timesteps", "episode_reward", "episode_length"])

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.w.writerow([self.num_timesteps, info["episode"]["r"], info["episode"]["l"]])
            return True

        def _on_training_end(self):
            self.f.close()

    train_env = make_env(train_eps, cfg, cfg.seed)
    val_env = make_env(val_eps, cfg, cfg.seed + 1)
    model = MaskablePPO(
        "MlpPolicy", train_env, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, gamma=cfg.gamma, clip_range=cfg.clip_range, gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef, n_epochs=cfg.n_epochs, seed=cfg.seed, device="cpu", verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch))),
    )
    callbacks = [
        RewardLogger(os.path.join(out, "training_log.csv")),
        MaskableEvalCallback(val_env, n_eval_episodes=min(cfg.n_eval_episodes, len(val_eps)),
                             eval_freq=cfg.eval_freq, best_model_save_path=out,
                             log_path=os.path.join(out, "eval"), deterministic=True, verbose=0),
        CheckpointCallback(save_freq=max(cfg.total_timesteps // 3, 1), save_path=os.path.join(out, "checkpoints"),
                           name_prefix="ppo_sched"),
    ]
    print(f"training {cfg.total_timesteps} timesteps (seed {cfg.seed}) -> {out}")
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)
    model.save(os.path.join(out, "final_model"))
    cfg.to_json(os.path.join(out, "config.json"))
    joblib.dump(risk_model, os.path.join(out, "risk_model.joblib"))
    print("saved:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
