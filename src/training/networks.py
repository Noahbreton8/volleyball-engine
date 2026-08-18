"""
Actor and Centralized Critic networks for MAPPO.
Actor: local obs → action distribution (Gaussian).
Critic: global obs → value scalar.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: list[int], out_dim: int, activation=nn.Tanh,
         output_gain: float = 1.0) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        linear = nn.Linear(prev, h)
        nn.init.orthogonal_(linear.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(linear.bias)
        layers += [linear, activation()]
        prev = h
    out_linear = nn.Linear(prev, out_dim)
    nn.init.orthogonal_(out_linear.weight, gain=output_gain)
    nn.init.zeros_(out_linear.bias)
    layers.append(out_linear)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Gaussian policy: outputs mean and log_std for each action dimension.
    log_std is a learned parameter (not obs-dependent) for stability.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] = [512, 256, 128]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim, output_gain=0.1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 2.0)  # std≈0.135, allows active balance exploration

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.net(obs))   # bound actions to (-1, 1)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self(obs)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Value function that observes global state (all agents + ball).
    Shared across all agents to reduce parameter count.
    """
    def __init__(self, global_obs_dim: int, hidden: list[int] = [1024, 512, 256]):
        super().__init__()
        self.net = _mlp(global_obs_dim, hidden, 1)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


class AgentPolicyBank:
    """
    Holds one Actor per role. Maps agent_idx → correct Actor at runtime.
    Also holds frozen opponent copies for self-play.
    """
    def __init__(self, obs_dim: int, act_dim: int, roles, device: torch.device):
        from ..env.constants import Role
        self.device = device
        self.roles = roles

        # One actor per role
        self.actors: dict[Role, Actor] = {
            role: Actor(obs_dim, act_dim).to(device)
            for role in Role
        }

        # Frozen opponent actors (updated periodically in self-play)
        self.frozen_actors: dict[Role, Actor] = {
            role: Actor(obs_dim, act_dim).to(device)
            for role in Role
        }
        self.update_frozen()

    def get_actor(self, agent_idx: int, frozen: bool = False) -> Actor:
        role = self.roles[agent_idx]
        return self.frozen_actors[role] if frozen else self.actors[role]

    def update_frozen(self):
        """Copy current policy weights to frozen opponents."""
        for role in self.actors:
            self.frozen_actors[role].load_state_dict(
                self.actors[role].state_dict()
            )

    def parameters(self):
        for actor in self.actors.values():
            yield from actor.parameters()

    def state_dict(self) -> dict:
        return {str(int(role)): actor.state_dict() for role, actor in self.actors.items()}

    def load_state_dict(self, sd: dict):
        from ..env.constants import Role
        for role in Role:
            key = str(int(role))
            if key in sd:
                self.actors[role].load_state_dict(sd[key])
