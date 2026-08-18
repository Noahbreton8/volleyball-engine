"""
Parallel environment runner using Python multiprocessing.
Each worker process owns one VolleyballEnv and runs episodes independently.
The main process collects trajectories from all workers and runs PPO updates.
"""
import multiprocessing as mp
import numpy as np
import torch
from typing import Any

from ..env.constants import (
    TOTAL_PLAYERS, PLAYERS_PER_TEAM, LOCAL_OBS_DIM, GLOBAL_OBS_DIM,
    PLAYER_ACT_DIM, Role
)


# Commands sent to worker processes
_CMD_RESET  = 0
_CMD_STEP   = 1
_CMD_CLOSE  = 2
_CMD_ACT    = 3   # send actions, get (obs, reward, done, info, global_obs)


def _worker_fn(worker_id: int, pipe: mp.connection.Connection, seed_offset: int):
    """
    Worker process: owns one VolleyballEnv. Receives commands from main process.
    Uses pipe for bidirectional communication.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.env.volleyball_env import VolleyballEnv

    env = VolleyballEnv(render_mode="none", record=False)
    episode = seed_offset + worker_id * 10000

    while True:
        cmd, payload = pipe.recv()

        if cmd == _CMD_RESET:
            episode += 1
            obs, _ = env.reset(seed=episode)
            global_obs = env.get_global_obs()
            pipe.send((obs, global_obs))

        elif cmd == _CMD_STEP:
            actions = payload  # dict {int: np.ndarray}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())
            if done:
                # Auto-reset so next step starts a fresh episode (standard PPO practice).
                # Keeps all 64 rollout steps as valid training data instead of letting
                # the env sit in a post-terminal state for the remaining steps.
                episode += 1
                obs, _ = env.reset(seed=episode)
            global_obs = env.get_global_obs()
            pipe.send((obs, rewards, done, infos, global_obs))

        elif cmd == _CMD_CLOSE:
            env.close()
            pipe.send(None)
            break


class ParallelEnvs:
    """
    Pool of N volleyball environments running in separate processes.
    Synchronous step: sends actions to all workers, collects results.
    """
    def __init__(self, n_envs: int, seed_offset: int = 0):
        self.n_envs = n_envs
        self._pipes: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_fn,
                args=(i, child_conn, seed_offset),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._pipes.append(parent_conn)
            self._procs.append(proc)

    def reset_all(self) -> tuple[list[dict], list[np.ndarray]]:
        for pipe in self._pipes:
            pipe.send((_CMD_RESET, None))
        results = [pipe.recv() for pipe in self._pipes]
        obs_list = [r[0] for r in results]
        gobs_list = [r[1] for r in results]
        return obs_list, gobs_list

    def step_all(
        self,
        actions_list: list[dict[int, np.ndarray]],
    ) -> list[tuple]:
        """Send actions to all envs, return list of (obs, rewards, done, infos, global_obs)."""
        for pipe, actions in zip(self._pipes, actions_list):
            pipe.send((_CMD_STEP, actions))
        return [pipe.recv() for pipe in self._pipes]

    def reset_one(self, env_i: int) -> tuple[dict, np.ndarray]:
        """Reset a single environment by index. Returns (obs, global_obs)."""
        self._pipes[env_i].send((_CMD_RESET, None))
        obs, gobs = self._pipes[env_i].recv()
        return obs, gobs

    def close(self):
        for pipe in self._pipes:
            pipe.send((_CMD_CLOSE, None))
        for pipe in self._pipes:
            pipe.recv()
        for proc in self._procs:
            proc.join(timeout=5)


def collect_parallel_rollout(
    envs: ParallelEnvs,
    trainer,
    steps_per_env: int = 64,
) -> tuple[list[dict], dict]:
    """
    Collect `steps_per_env` control steps from each parallel env.
    Returns (buffer_entries, stats).
    Each buffer entry: (obs, global_obs, action, log_prob, reward, value, done, agent_id, role)
    """
    obs_list, gobs_list = envs.reset_all()
    entries = []
    stats = {"wins": 0, "contacts": 0, "episodes": envs.n_envs}

    for _ in range(steps_per_env):
        # Compute actions for all envs × all agents
        actions_for_envs = []
        log_probs_for_envs = []
        values_for_envs = []

        for env_i in range(envs.n_envs):
            obs = obs_list[env_i]
            gobs = gobs_list[env_i]
            value = trainer.get_value(gobs)
            values_for_envs.append(value)

            actions = {}
            lps = {}
            for i in range(TOTAL_PLAYERS):
                obs_t = torch.tensor(obs[i], dtype=torch.float32, device=trainer.device).unsqueeze(0)
                frozen = (i >= PLAYERS_PER_TEAM)
                actor = trainer.policy_bank.get_actor(i, frozen=frozen)
                with torch.no_grad():
                    action, lp = actor.act(obs_t)
                actions[i] = action.squeeze(0).cpu().numpy()
                lps[i] = lp.item()
            actions_for_envs.append(actions)
            log_probs_for_envs.append(lps)

        results = envs.step_all(actions_for_envs)

        for env_i, (next_obs, rewards, done, infos, next_gobs) in enumerate(results):
            gobs = gobs_list[env_i]
            value = values_for_envs[env_i]
            actions = actions_for_envs[env_i]
            lps = log_probs_for_envs[env_i]

            info0 = infos[0]
            if done and info0.get("winning_team", -1) == 0:
                stats["wins"] += 1
            if info0.get("contact_player", -1) >= 0:
                stats["contacts"] += 1

            for i in range(PLAYERS_PER_TEAM):
                entries.append((
                    obs_list[env_i][i],
                    gobs,
                    actions[i],
                    lps[i],
                    rewards[i],
                    value,
                    done,
                    i,
                    int(trainer.roles[i]),
                    env_i,  # env_id for per-env GAE computation
                ))

            if done:
                # Reset only this env — resetting all would corrupt other envs' obs
                new_obs, new_gobs = envs.reset_one(env_i)
                obs_list[env_i] = new_obs
                gobs_list[env_i] = new_gobs
                stats["episodes"] += 1
            else:
                obs_list[env_i] = next_obs
                gobs_list[env_i] = next_gobs

    return entries, stats
