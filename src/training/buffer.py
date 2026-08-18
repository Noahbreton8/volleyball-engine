"""
Rollout buffer for MAPPO. Stores trajectories for all agents across N episodes.
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import NamedTuple


class Batch(NamedTuple):
    obs: torch.Tensor          # (B, obs_dim)
    global_obs: torch.Tensor   # (B, global_obs_dim)
    actions: torch.Tensor      # (B, act_dim)
    log_probs: torch.Tensor    # (B,)
    returns: torch.Tensor      # (B,)
    advantages: torch.Tensor   # (B,)
    agent_ids: torch.Tensor    # (B,) agent index
    roles: torch.Tensor        # (B,) role index


class RolloutBuffer:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self._obs: list[np.ndarray] = []
        self._global_obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._log_probs: list[float] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._dones: list[bool] = []
        self._agent_ids: list[int] = []
        self._roles: list[int] = []
        self._env_ids: list[int] = []

    def add(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        agent_id: int,
        role: int,
        env_id: int = 0,
    ):
        self._obs.append(obs.copy())
        self._global_obs.append(global_obs.copy())
        self._actions.append(action.copy())
        self._log_probs.append(float(log_prob))
        self._rewards.append(float(reward))
        self._values.append(float(value))
        self._dones.append(bool(done))
        self._agent_ids.append(agent_id)
        self._roles.append(role)
        self._env_ids.append(int(env_id))

    def compute_returns(self, last_value: float = 0.0) -> Batch:
        """GAE-lambda return computation, computed per (env_id, agent_id) for correctness."""
        n = len(self._rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        env_ids = np.array(self._env_ids, dtype=np.int32)
        agent_ids = np.array(self._agent_ids, dtype=np.int32)

        for env_id in np.unique(env_ids):
            for agent_id in np.unique(agent_ids[env_ids == env_id]):
                idx = np.where((env_ids == env_id) & (agent_ids == agent_id))[0]
                # idx is in step-insertion order; reverse for backward GAE sweep
                gae = 0.0
                next_val = last_value
                for pos in reversed(idx):
                    done = self._dones[pos]
                    val = self._values[pos]
                    delta = self._rewards[pos] + self.gamma * next_val * (1 - done) - val
                    gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
                    advantages[pos] = gae
                    returns[pos] = gae + val
                    next_val = val

        # Normalize advantages globally
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return Batch(
            obs=torch.tensor(np.array(self._obs), dtype=torch.float32),
            global_obs=torch.tensor(np.array(self._global_obs), dtype=torch.float32),
            actions=torch.tensor(np.array(self._actions), dtype=torch.float32),
            log_probs=torch.tensor(self._log_probs, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            agent_ids=torch.tensor(self._agent_ids, dtype=torch.long),
            roles=torch.tensor(self._roles, dtype=torch.long),
        )

    def __len__(self):
        return len(self._rewards)
