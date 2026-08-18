"""
Main MAPPO training loop for volleyball agents.

Usage:
    python -m src.training.train
    python -m src.training.train --episodes 50000 --record-every 500 --n-envs 8
    python -m src.training.train --resume checkpoints/latest.pt
"""
import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from ..env.volleyball_env import VolleyballEnv
from ..env.constants import (
    TOTAL_PLAYERS, PLAYERS_PER_TEAM, LOCAL_OBS_DIM, GLOBAL_OBS_DIM,
    PLAYER_ACT_DIM, Role
)
from .mappo import MAPPOTrainer
from .buffer import RolloutBuffer
from .parallel_env import ParallelEnvs, collect_parallel_rollout
from ..visualization.recorder import save_recording


def collect_rollout(
    env: VolleyballEnv,
    trainer: MAPPOTrainer,
    buffer: RolloutBuffer,
    episode: int,
    record: bool = False,
) -> dict:
    """Run one rally episode, fill buffer, return episode stats."""
    env.record = record
    obs_dict, _ = env.reset(seed=episode)

    done = False
    ep_rewards = np.zeros(TOTAL_PLAYERS, dtype=np.float32)
    ep_contacts = 0
    winning_team = -1

    while not done:
        global_obs = env.get_global_obs()
        value = trainer.get_value(global_obs)

        actions = {}
        log_probs = {}

        for i in range(TOTAL_PLAYERS):
            obs_t = torch.tensor(obs_dict[i], dtype=torch.float32, device=trainer.device)
            # Team B (agents 6-11) uses frozen opponent policy in self-play
            use_frozen = (i >= PLAYERS_PER_TEAM)
            actor = trainer.policy_bank.get_actor(i, frozen=use_frozen)
            with torch.no_grad():
                action, lp = actor.act(obs_t.unsqueeze(0))
            actions[i] = action.squeeze(0).cpu().numpy()
            log_probs[i] = lp.item()

        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        done_flag = any(terminated.values()) or any(truncated.values())
        info0 = infos[0]
        if done_flag:
            winning_team = info0.get("winning_team", -1)
        if info0.get("contact_player", -1) >= 0:
            ep_contacts += 1

        # Store transitions for team A agents only (team B is frozen opponent)
        for i in range(PLAYERS_PER_TEAM):
            buffer.add(
                obs=obs_dict[i],
                global_obs=global_obs,
                action=actions[i],
                log_prob=log_probs[i],
                reward=rewards[i],
                value=value,
                done=done_flag,
                agent_id=i,
                role=int(trainer.roles[i]),
                env_id=0,
            )
            ep_rewards[i] += rewards[i]

        obs_dict = next_obs
        done = done_flag

    return {
        "ep_rewards": ep_rewards,
        "winning_team": winning_team,
        "ep_contacts": ep_contacts,
        "recording": env.get_recording() if record else None,
    }


def train(
    total_episodes: int = 100_000,
    rollouts_per_update: int = 16,
    record_every: int = 500,
    save_every: int = 1000,
    resume: str | None = None,
    checkpoint_dir: str = "checkpoints",
    recordings_dir: str = "recordings",
    log_dir: str = "runs",
    n_envs: int = 1,
    steps_per_env: int = 64,
    frozen_update_interval: int = 200,
    entropy_coeff: float = 0.01,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Parallel envs: {n_envs}")

    # Single env for recording + role extraction
    single_env = VolleyballEnv(render_mode="none")
    roles = single_env.roles

    trainer = MAPPOTrainer(roles=roles, device=device, frozen_update_interval=frozen_update_interval, entropy_coeff=entropy_coeff)

    if resume:
        trainer.load(resume)
        print(f"[train] Resumed from {resume} (episode {trainer._episode_count})")

    writer = SummaryWriter(log_dir=log_dir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(recordings_dir).mkdir(parents=True, exist_ok=True)

    buffer = RolloutBuffer(
        n_agents=PLAYERS_PER_TEAM,
        obs_dim=LOCAL_OBS_DIM,
        global_obs_dim=GLOBAL_OBS_DIM,
        act_dim=PLAYER_ACT_DIM,
    )

    # Spin up parallel envs (or just use single env if n_envs=1)
    if n_envs > 1:
        parallel = ParallelEnvs(n_envs)
    else:
        parallel = None

    ep = trainer._episode_count
    team_a_wins = 0
    start_time = time.time()
    losses = {}

    print(f"[train] Starting from episode {ep}, target {total_episodes}")

    while ep < total_episodes:
        buffer.clear()

        if parallel is not None:
            # Parallel collection: n_envs × steps_per_env transitions
            entries, stats = collect_parallel_rollout(parallel, trainer, steps_per_env)
            for entry in entries:
                buffer.add(*entry)
            episodes_this_iter = stats["episodes"]
            contacts_this_iter = stats["contacts"]
            wins_this_iter = stats["wins"]
        else:
            # Single-env collection
            episodes_this_iter = rollouts_per_update
            contacts_this_iter = 0
            wins_this_iter = 0
            for _ in range(rollouts_per_update):
                stats = collect_rollout(single_env, trainer, buffer, ep)
                if stats["winning_team"] == 0:
                    wins_this_iter += 1
                contacts_this_iter += stats["ep_contacts"]
                trainer.on_episode_end()
                ep += 1

        # Record one episode periodically (always single-env for simplicity)
        if ep % record_every < episodes_this_iter:
            record_stats = collect_rollout(single_env, trainer, buffer, ep, record=True)
            if record_stats["recording"]:
                rec_path = Path(recordings_dir) / f"episode_{ep:06d}.npz"
                save_recording(record_stats["recording"], rec_path,
                               metadata={"episode": ep})
                print(f"[train] Recorded episode → {rec_path}")

        if parallel is not None:
            ep += episodes_this_iter
            team_a_wins += wins_this_iter

        # PPO update
        if len(buffer) > 0:
            batch = buffer.compute_returns()
            losses = trainer.update(batch)

        # Logging
        win_rate = team_a_wins / max(1, ep)
        writer.add_scalar("train/win_rate_team_a", win_rate, ep)
        writer.add_scalar("train/contacts_per_episode",
                          contacts_this_iter / max(1, episodes_this_iter), ep)
        if losses:
            writer.add_scalar("train/actor_loss", losses["actor_loss"], ep)
            writer.add_scalar("train/critic_loss", losses["critic_loss"], ep)
            writer.add_scalar("train/entropy", losses["entropy"], ep)

        elapsed = time.time() - start_time
        eps_per_sec = ep / max(elapsed, 1)
        eta = (total_episodes - ep) / max(eps_per_sec, 1)

        print(
            f"ep {ep:6d}/{total_episodes}  "
            f"win%={win_rate*100:.1f}  "
            f"speed={eps_per_sec:.1f}ep/s  "
            f"ETA={eta/60:.0f}m"
            + (f"  actor={losses.get('actor_loss',0):.3f}" if losses else "")
        )

        # Checkpoint
        if ep % save_every < episodes_this_iter:
            ckpt = Path(checkpoint_dir) / f"checkpoint_{ep:06d}.pt"
            trainer.save(str(ckpt))
            trainer.save(str(Path(checkpoint_dir) / "latest.pt"))
            print(f"[train] Saved checkpoint → {ckpt}")

    if parallel:
        parallel.close()
    single_env.close()
    writer.close()
    print("[train] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int,   default=100_000)
    parser.add_argument("--rollouts",      type=int,   default=16)
    parser.add_argument("--record-every",  type=int,   default=500)
    parser.add_argument("--save-every",    type=int,   default=1000)
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--checkpoint-dir", type=str,  default="checkpoints")
    parser.add_argument("--recordings-dir", type=str,  default="recordings")
    parser.add_argument("--log-dir",       type=str,   default="runs")
    parser.add_argument("--n-envs",        type=int,   default=1,
                        help="Parallel envs for physics. Use CPU core count (e.g. 8).")
    parser.add_argument("--steps-per-env", type=int,   default=64,
                        help="Steps collected per env per update when using --n-envs.")
    parser.add_argument("--frozen-update-interval", type=int, default=200,
                        help="Episodes between frozen opponent (Team B) weight updates.")
    parser.add_argument("--entropy-coeff", type=float, default=0.01,
                        help="PPO entropy bonus coefficient. Higher = more exploration.")
    args = parser.parse_args()

    train(
        total_episodes=args.episodes,
        rollouts_per_update=args.rollouts,
        record_every=args.record_every,
        save_every=args.save_every,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        recordings_dir=args.recordings_dir,
        log_dir=args.log_dir,
        n_envs=args.n_envs,
        steps_per_env=args.steps_per_env,
        frozen_update_interval=args.frozen_update_interval,
        entropy_coeff=args.entropy_coeff,
    )


if __name__ == "__main__":
    main()
