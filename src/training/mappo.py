"""
MAPPO trainer. Centralized training with decentralized execution.
One update call processes all agents' rollout data with the shared critic.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Iterator

from .networks import AgentPolicyBank, CentralizedCritic
from .buffer import Batch, RolloutBuffer
from ..env.constants import (
    TOTAL_PLAYERS, LOCAL_OBS_DIM, GLOBAL_OBS_DIM, PLAYER_ACT_DIM, Role
)


class MAPPOTrainer:
    def __init__(
        self,
        roles: list[Role],
        device: torch.device | None = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 512,
        frozen_update_interval: int = 200,  # episodes between self-play freezes
    ):
        self.device = device or torch.device("cpu")
        self.roles = roles
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.frozen_update_interval = frozen_update_interval
        self._episode_count = 0

        self.policy_bank = AgentPolicyBank(
            LOCAL_OBS_DIM, PLAYER_ACT_DIM, roles, self.device
        )
        self.critic = CentralizedCritic(GLOBAL_OBS_DIM).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.policy_bank.parameters(), lr=lr_actor, eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=1e-5
        )

    def update(self, batch: Batch) -> dict[str, float]:
        """Run PPO_EPOCHS of minibatch updates. Returns loss dict for logging."""
        batch = Batch(*(t.to(self.device) for t in batch))
        n = len(batch.returns)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            for mb_idx in self._minibatch_indices(n):
                mb_obs = batch.obs[mb_idx]
                mb_gobs = batch.global_obs[mb_idx]
                mb_act = batch.actions[mb_idx]
                mb_old_lp = batch.log_probs[mb_idx]
                mb_ret = batch.returns[mb_idx]
                mb_adv = batch.advantages[mb_idx]
                mb_aids = batch.agent_ids[mb_idx]

                # Critic loss
                values = self.critic(mb_gobs)
                critic_loss = nn.functional.mse_loss(values, mb_ret)

                # Actor loss: must handle per-agent roles
                # Group by role for efficient forward pass
                total_actor_loss = torch.tensor(0.0, device=self.device)
                total_entropy = torch.tensor(0.0, device=self.device)
                role_counts = 0

                for role in Role:
                    mask = torch.tensor(
                        [self.roles[int(aid)] == role for aid in mb_aids],
                        dtype=torch.bool, device=self.device
                    )
                    if mask.sum() == 0:
                        continue

                    actor = self.policy_bank.actors[role]
                    log_probs, entropy = actor.evaluate(mb_obs[mask], mb_act[mask])

                    ratio = torch.exp(log_probs - mb_old_lp[mask])
                    adv = mb_adv[mask]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    total_actor_loss = total_actor_loss + actor_loss
                    total_entropy = total_entropy + entropy.mean()
                    role_counts += 1

                if role_counts > 0:
                    total_actor_loss = total_actor_loss / role_counts
                    total_entropy = total_entropy / role_counts

                loss = (
                    total_actor_loss
                    + self.value_loss_coeff * critic_loss
                    - self.entropy_coeff * total_entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy_bank.parameters()), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(total_actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(total_entropy.item())

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def on_episode_end(self):
        self._episode_count += 1
        if self._episode_count % self.frozen_update_interval == 0:
            self.policy_bank.update_frozen()

    def get_value(self, global_obs: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(t).item()

    def save(self, path: str):
        torch.save({
            "actors": self.policy_bank.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "episodes": self._episode_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_bank.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self._episode_count = ckpt.get("episodes", 0)
        self.policy_bank.update_frozen()

    def _minibatch_indices(self, n: int) -> Iterator[torch.Tensor]:
        indices = torch.randperm(n)
        for start in range(0, n, self.minibatch_size):
            yield indices[start : start + self.minibatch_size]
