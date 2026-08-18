"""
Phase 1 pre-training: teach agents to stand upright and walk toward a target.
No ball, no volleyball rules. Just locomotion.

Run this before full volleyball training:
    python -m src.training.pretrain_locomotion --steps 300000
    python -m src.training.train --resume checkpoints/locomotion_pretrain.pt --episodes 500000
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import mujoco
from ..env.scene_builder import build_scene_xml
from ..env.constants import (
    TOTAL_PLAYERS, PLAYERS_PER_TEAM, PLAYER_ACT_DIM,
    LOCAL_OBS_DIM, GLOBAL_OBS_DIM, PLAYER_QPOS_DIM, PLAYER_QVEL_DIM,
    SIM_DT, PHYSICS_STEPS_PER_CONTROL, TEAM_A_START_POSITIONS,
    TEAM_B_START_POSITIONS, START_HEIGHT, Role
)
from .mappo import MAPPOTrainer


TARGET_HEIGHT = 1.0    # torso z must stay above this to be considered upright
UPRIGHT_REWARD = 0.5   # per step for staying upright
FALL_PENALTY = -5.0    # when torso drops below TARGET_HEIGHT
ALIVE_BONUS = 0.05     # small per-step bonus just for existing


def locomotion_step_reward(torso_z: float, torso_z_prev: float, vel_z: float) -> float:
    """Dense reward for staying upright at standing height. Penalizes jumping AND falling."""
    if torso_z < 0.8:
        return FALL_PENALTY           # -5.0: fell over
    if torso_z > START_HEIGHT + 0.5:  # jumped above ~1.85m
        return -3.0                   # strong penalty to prevent the jump exploit
    # Gaussian around ideal standing height after settling (~1.25-1.35m)
    deviation = abs(torso_z - START_HEIGHT)
    upright = max(0.0, 1.0 - deviation / 0.4)
    return ALIVE_BONUS + UPRIGHT_REWARD * upright


class LocoEnv:
    """Minimal environment: 12 humanoids on the court, no ball physics, just stay up."""

    def __init__(self):
        xml = build_scene_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache player qpos/qvel/act addresses
        self.qpos_adr = []
        self.qvel_adr = []
        for i in range(TOTAL_PLAYERS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_root")
            self.qpos_adr.append(self.model.jnt_qposadr[jid])
            self.qvel_adr.append(self.model.jnt_dofadr[jid])
        self.act_adr = [i * PLAYER_ACT_DIM for i in range(TOTAL_PLAYERS)]

        self._step = 0
        self.max_steps = 300

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        all_pos = list(TEAM_A_START_POSITIONS) + list(TEAM_B_START_POSITIONS)
        rng = np.random.default_rng()
        for i, (x, y) in enumerate(all_pos):
            adr = self.qpos_adr[i]
            self.data.qpos[adr:adr+3] = [x, y, START_HEIGHT]
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            # Small noise on joint angles so agents aren't perfectly identical
            noise = rng.uniform(-0.05, 0.05, PLAYER_QPOS_DIM - 7).astype(np.float64)
            self.data.qpos[adr+7:adr+PLAYER_QPOS_DIM] = noise
        # Settle for 30 physics steps with zero control so feet land and joints damp out
        self.data.ctrl[:] = 0.0
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)
        self._step = 0
        self._prev_z = [self._torso_z(i) for i in range(TOTAL_PLAYERS)]
        return self._get_obs()

    def step(self, actions: dict):
        for i, act in actions.items():
            adr = self.act_adr[i]
            self.data.ctrl[adr:adr+PLAYER_ACT_DIM] = np.clip(act, -1, 1) * 0.4

        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        rewards = {}
        for i in range(TOTAL_PLAYERS):
            z = self._torso_z(i)
            adr_v = self.qvel_adr[i]
            # Angular velocity of torso root (indices 3-5 in qvel)
            ang_vel = self.data.qvel[adr_v+3:adr_v+6]
            ang_vel_penalty = -0.005 * float(np.dot(ang_vel, ang_vel))
            rewards[i] = locomotion_step_reward(z, self._prev_z[i], 0.0) + ang_vel_penalty
            self._prev_z[i] = z

        # Only terminate at time limit or extreme jumping — do NOT terminate on fall.
        # Agents lying on the ground (z<0.7) get FALL_PENALTY=-5.0 every step,
        # creating a massive advantage signal at the fall transition (~500 nats stronger
        # than the lying state). This gives PPO a strong gradient to prevent falls.
        all_fallen = all(self._torso_z(i) < 0.8 for i in range(TOTAL_PLAYERS))
        done = self._step >= self.max_steps or all_fallen or any(
            self._torso_z(i) > 2.0
            for i in range(TOTAL_PLAYERS)
        )
        return self._get_obs(), rewards, done

    def _torso_z(self, i: int) -> float:
        return float(self.data.qpos[self.qpos_adr[i] + 2])

    def _get_obs(self) -> dict:
        obs = {}
        for i in range(TOTAL_PLAYERS):
            adr_q = self.qpos_adr[i]
            adr_v = self.qvel_adr[i]
            qpos = self.data.qpos[adr_q:adr_q+PLAYER_QPOS_DIM].copy()
            qvel = self.data.qvel[adr_v:adr_v+PLAYER_QVEL_DIM].copy()
            flat = np.concatenate([qpos, qvel], dtype=np.float32)
            # Pad to LOCAL_OBS_DIM
            padded = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
            n = min(len(flat), LOCAL_OBS_DIM)
            padded[:n] = flat[:n]
            obs[i] = padded
        return obs

    def get_global_obs(self) -> np.ndarray:
        parts = [np.zeros(9, dtype=np.float32)]  # ball placeholder
        for i in range(TOTAL_PLAYERS):
            adr_q = self.qpos_adr[i]
            adr_v = self.qvel_adr[i]
            parts.append(self.data.qpos[adr_q:adr_q+PLAYER_QPOS_DIM].copy())
            parts.append(self.data.qvel[adr_v:adr_v+PLAYER_QVEL_DIM].copy())
        parts.append(np.zeros(GLOBAL_OBS_DIM - 9 - TOTAL_PLAYERS*(PLAYER_QPOS_DIM+PLAYER_QVEL_DIM), dtype=np.float32))
        result = np.concatenate(parts, dtype=np.float32)
        return result[:GLOBAL_OBS_DIM]


def pretrain(steps: int = 300_000, save_path: str = "checkpoints/locomotion_pretrain.pt",
             log_dir: str = "runs/locomotion"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] Locomotion pre-training for {steps} steps on {device}")

    from ..env.volleyball_env import VolleyballEnv
    tmp = VolleyballEnv(render_mode="none")
    roles = tmp.roles
    tmp.close()

    trainer = MAPPOTrainer(roles=roles, device=device, ppo_epochs=5)
    writer = SummaryWriter(log_dir=log_dir)
    Path("checkpoints").mkdir(exist_ok=True)

    env = LocoEnv()
    obs = env.reset()

    total_reward = 0.0
    ep_count = 0
    ep_steps = 0
    t0 = time.time()

    from .buffer import RolloutBuffer
    buffer = RolloutBuffer(TOTAL_PLAYERS, LOCAL_OBS_DIM, GLOBAL_OBS_DIM, PLAYER_ACT_DIM)

    for step in range(steps):
        global_obs = env.get_global_obs()
        value = trainer.get_value(global_obs)

        actions = {}
        log_probs = {}
        for i in range(TOTAL_PLAYERS):
            obs_t = torch.tensor(obs[i], dtype=torch.float32, device=trainer.device).unsqueeze(0)
            actor = trainer.policy_bank.get_actor(i, frozen=False)
            with torch.no_grad():
                action, lp = actor.act(obs_t)
            actions[i] = action.squeeze(0).cpu().numpy()
            log_probs[i] = lp.item()

        next_obs, rewards, done = env.step(actions)

        for i in range(TOTAL_PLAYERS):
            buffer.add(obs[i], global_obs, actions[i], log_probs[i],
                       rewards[i], value, done, i, int(roles[i]))
            total_reward += rewards[i]

        obs = next_obs
        ep_steps += 1

        if done:
            obs = env.reset()
            ep_count += 1

            if ep_count % 50 == 0:
                avg_r = total_reward / max(1, ep_steps * TOTAL_PLAYERS)
                elapsed = time.time() - t0
                print(f"  step {step:7d}  ep {ep_count:5d}  avg_reward={avg_r:.3f}  "
                      f"speed={step/elapsed:.0f}steps/s")
                writer.add_scalar("pretrain/avg_reward", avg_r, step)
                total_reward = 0.0
                ep_steps = 0

            # PPO update every 8 episodes (more frequent updates needed for balance learning)
            if ep_count % 8 == 0 and len(buffer) > 0:
                batch = buffer.compute_returns()
                losses = trainer.update(batch)
                buffer.clear()
                writer.add_scalar("pretrain/actor_loss", losses["actor_loss"], step)

    trainer.save(save_path)
    print(f"[pretrain] Saved → {save_path}")
    print(f"[pretrain] Now run:")
    print(f"  python -m src.training.train --resume {save_path} --episodes 500000 --n-envs 8")
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int, default=300_000)
    parser.add_argument("--save",     type=str, default="checkpoints/locomotion_pretrain.pt")
    parser.add_argument("--log-dir",  type=str, default="runs/locomotion")
    args = parser.parse_args()
    pretrain(args.steps, args.save, args.log_dir)


if __name__ == "__main__":
    main()
