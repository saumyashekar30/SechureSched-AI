"""PPO inference: run a trained policy (or any chooser) through the scheduling environment."""
import json
import os

import numpy as np

from .config import TrainConfig
from .sched_env import SchedulingEnv


def run_chooser(df_with_risk, chooser, env_config=None):
    """Run ``chooser(env) -> action`` to completion on one workload; return the results DataFrame
    (same schema as ``run_simulation``)."""
    env = SchedulingEnv(df_with_risk, config=env_config)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(chooser(env))
    return env.results_df()


class PPOScheduler:
    """Loads a trained MaskablePPO policy and schedules workloads deterministically."""

    def __init__(self, model, env_config):
        self.model = model
        self.env_config = env_config

    @classmethod
    def load(cls, model_path, config_path=None):
        from sb3_contrib import MaskablePPO
        config_path = config_path or os.path.join(os.path.dirname(model_path), "config.json")
        cfg = TrainConfig.from_json(config_path)
        return cls(MaskablePPO.load(model_path, device="cpu"), cfg.env)

    def schedule(self, df_with_risk):
        env = SchedulingEnv(df_with_risk, config=self.env_config)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
        return env.results_df()
