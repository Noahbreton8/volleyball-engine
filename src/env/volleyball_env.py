"""
Multi-agent Gymnasium environment for 6v6 indoor volleyball.
One episode = one rally. Observation dicts keyed by agent index (0-11).
"""
import numpy as np
import mujoco
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from .scene_builder import build_scene_xml
from .constants import (
    Phase, Role, PLAYERS_PER_TEAM, TOTAL_PLAYERS, PLAYER_ACT_DIM,
    LOCAL_OBS_DIM, GLOBAL_OBS_DIM, PHYSICS_STEPS_PER_CONTROL,
    MAX_RALLY_STEPS, MAGNUS_COEFF, TEAM_ROLE_ASSIGNMENT,
    TEAM_A_START_POSITIONS, TEAM_B_START_POSITIONS, START_HEIGHT, NET_HEIGHT,
    PLAYER_QPOS_DIM, PLAYER_QVEL_DIM, SIM_DT, OBS_PHASE, OBS_ROLE,
)
from .phase import PhaseTracker
from .rewards import compute_agent_reward


class VolleyballEnv(gym.Env):
    """
    Multi-agent volleyball environment.

    Agents: 0-5 = team A, 6-11 = team B.
    Each agent receives a local observation and a shared role-specific reward.
    The centralized critic receives global_obs from get_global_obs().
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, render_mode: str = "none", record: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.record = record

        # Build and compile MuJoCo model
        xml = build_scene_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache body/joint ids
        self._cache_ids()

        # Roles per agent (0-5 team A, 6-11 team B)
        self.roles: list[Role] = list(TEAM_ROLE_ASSIGNMENT) + list(TEAM_ROLE_ASSIGNMENT)

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(LOCAL_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(PLAYER_ACT_DIM,), dtype=np.float32
        )

        # State
        self.phase_tracker = PhaseTracker()
        self._step_count = 0
        self._episode_contacts: list[dict] = []  # for recording
        self._player_reset_xy = np.zeros((TOTAL_PLAYERS, 2))

        # Viewer (lazy init)
        self._viewer = None

        # Recording buffer
        self._recording: list[dict] | None = None

    def _cache_ids(self):
        """Cache MuJoCo body/joint/actuator IDs for fast lookup."""
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.player_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"p{i}_torso")
            for i in range(TOTAL_PLAYERS)
        ]
        # qpos/qvel start index per player (freejoint = first DOF)
        # MuJoCo lays out qpos in body-tree order; we find via joint ids
        self.player_qpos_adr = []
        self.player_qvel_adr = []
        for i in range(TOTAL_PLAYERS):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_root"
            )
            self.player_qpos_adr.append(self.model.jnt_qposadr[jid])
            self.player_qvel_adr.append(self.model.jnt_dofadr[jid])

        # Ball freejoint addresses
        ball_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.ball_qpos_adr = self.model.jnt_qposadr[ball_jid]
        self.ball_qvel_adr = self.model.jnt_dofadr[ball_jid]

        # Actuator start per player (25 actuators each, sequential)
        self.player_act_adr = [i * PLAYER_ACT_DIM for i in range(TOTAL_PLAYERS)]

        # Touch sensor ids
        self.touch_sensor_ids = {}
        for i in range(TOTAL_PLAYERS):
            for side in ("l", "r"):
                name = f"touch_p{i}_{side}"
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
                )
                self.touch_sensor_ids[(i, side)] = sid

        # Hand site ids for reach reward (site_xpos updated by mj_step)
        self.player_lhand_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"p{i}_l_hand_site")
            for i in range(TOTAL_PLAYERS)
        ]
        self.player_rhand_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"p{i}_r_hand_site")
            for i in range(TOTAL_PLAYERS)
        ]
        # Upper arm body ids — contact happens here, not at hand site
        self.player_luarm_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"p{i}_l_uarm")
            for i in range(TOTAL_PLAYERS)
        ]
        self.player_ruarm_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"p{i}_r_uarm")
            for i in range(TOTAL_PLAYERS)
        ]

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[int, np.ndarray], dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize which team receives: 50/50 so team A learns both serve and receive
        _default_serving = int(self.np_random.integers(0, 2))
        serving_team = options.get("serving_team", _default_serving) if options else _default_serving
        rotation = options.get("rotation", 0) if options else 0

        self._place_players(rotation)
        # Save post-settle positions for torso drift penalty (iter18)
        self._player_reset_xy = np.array([
            self.data.qpos[self.player_qpos_adr[i] : self.player_qpos_adr[i] + 2].copy()
            for i in range(TOTAL_PLAYERS)
        ])
        self._place_ball(serving_team)
        self.phase_tracker.reset(serving_team)
        self._step_count = 0
        self._prev_ball_pos = self._get_ball_pos().copy()
        self._prev_phase = Phase.SERVE

        if self.record:
            self._recording = []

        mujoco.mj_forward(self.model, self.data)

        obs = {i: self._get_obs(i) for i in range(TOTAL_PLAYERS)}
        return obs, {}

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> tuple[dict, dict, dict, dict, dict]:
        """
        Apply actions for all agents, advance physics, compute rewards.
        actions: {agent_id: np.ndarray of shape (PLAYER_ACT_DIM,)}
        """
        # Apply actions
        for i, action in actions.items():
            adr = self.player_act_adr[i]
            self.data.ctrl[adr : adr + PLAYER_ACT_DIM] = np.clip(action, -1, 1)

        # Physics substeps + Magnus force + elastic collision override
        contact_player = -1
        _elastic_applied = False  # apply elastic collision once per contact event
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            self._apply_magnus_force()
            # Capture ball velocity BEFORE physics step (pre-contact velocity for elastic formula)
            _adr = self.ball_qvel_adr
            _ball_vel_pre = self.data.qvel[_adr : _adr + 3].copy()
            mujoco.mj_step(self.model, self.data)
            cp = self._detect_ball_contact()
            if cp >= 0:
                if not _elastic_applied:
                    contact_player = cp
                    self._apply_elastic_collision(cp, _ball_vel_pre)
                    _elastic_applied = True
            else:
                _elastic_applied = False  # reset when contact ends

        ball_pos = self._get_ball_pos()
        ball_vel = self._get_ball_vel()

        # Precompute nearest hand distance to ball for reach reward.
        # Encourages agents to raise/move arms toward ball even before contact.
        hand_ball_dists = []
        for i in range(TOTAL_PLAYERS):
            l_pos = self.data.site_xpos[self.player_lhand_site_id[i]]
            r_pos = self.data.site_xpos[self.player_rhand_site_id[i]]
            dist = float(min(np.linalg.norm(l_pos - ball_pos),
                             np.linalg.norm(r_pos - ball_pos)))
            hand_ball_dists.append(dist)

        # Detect ball crossing the net (x=0) and clearing it (z > NET_HEIGHT).
        # Gives bonus to team A when ball goes A→B, and to team B when B→A.
        net_cross_bonus = [0.0] * TOTAL_PLAYERS
        prev_x = self._prev_ball_pos[0]
        curr_x = ball_pos[0]
        if prev_x * curr_x < 0:  # sign changed → crossed net plane
            frac = abs(prev_x) / (abs(prev_x) + abs(curr_x))
            z_at_net = self._prev_ball_pos[2] + frac * (ball_pos[2] - self._prev_ball_pos[2])
            if z_at_net > NET_HEIGHT:  # ball cleared the net
                sending_team = 0 if prev_x > 0 else 1  # team that sent the ball
                for i in range(TOTAL_PLAYERS):
                    if (i < PLAYERS_PER_TEAM) == (sending_team == 0):
                        net_cross_bonus[i] = 5.0  # strong reward for clearing net

        # Phase update
        phase, terminal, winning_team = self.phase_tracker.update(
            ball_pos, ball_vel, contact_player
        )

        self._step_count += 1
        timeout = self._step_count >= MAX_RALLY_STEPS
        if timeout and not terminal:
            terminal = True
            winning_team = -1  # no winner on timeout

        # Compute rewards
        rewards = {}
        for i in range(TOTAL_PLAYERS):
            team = 0 if i < PLAYERS_PER_TEAM else 1
            won = (winning_team == team) if terminal else False
            agent_pos = self._get_player_pos(i)
            opp_team = 1 - team
            opp_ids = range(PLAYERS_PER_TEAM) if opp_team == 0 else range(PLAYERS_PER_TEAM, TOTAL_PLAYERS)
            opp_positions = np.array([self._get_player_pos(j) for j in opp_ids])

            rewards[i] = compute_agent_reward(
                agent_idx=i,
                role=self.roles[i],
                phase=phase,
                phase_prev=self._prev_phase,
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                ball_pos_prev=self._prev_ball_pos,
                agent_pos=agent_pos,
                opponent_positions=opp_positions,
                contact_happened=(contact_player == i),
                rally_terminal=terminal,
                won_rally=won,
                team=team,
            )
            rewards[i] += net_cross_bonus[i]

            # Reach reward: dense signal for moving arm toward ball.
            # Only active when ball is on agent's own side (they should be intercepting it).
            ball_on_own_side = (ball_pos[0] > 0) == (team == 0)
            if ball_on_own_side:
                hd = hand_ball_dists[i]
                # iter27: reach radius 2.5→3.5m, coeff 0.5→0.3.
                # At episode start: ball z=3.5m, hand z≈1.0m → dist=2.5m.
                # Old 2.5m cap: reach=0 at step 0. New 3.5m: 0.3*(1-2.5/3.5)=0.086/step
                # from step 0 — gradient fires immediately to pull arm toward ball.
                if hd < 3.5:
                    rewards[i] += 0.3 * max(0.0, 1.0 - hd / 3.5)

                # 3D anticipation reward: guide upper arm to (pred_x, pred_y, 1.5m).
                # Contact happens at z≈1.5m (arm contact height). Arm horizontal
                # = upper arm center at shoulder height ≈1.5m — matches naturally.
                # Previous iter18 bug: XY-only reward gave no gradient for arm height,
                # so agents positioned laterally but never raised the arm. Adding Z
                # unifies XY+Z into single signal. Falloff 0.35m in 3D space.
                if ball_pos[2] > 1.5:
                    dz = ball_pos[2] - 1.5
                    disc = ball_vel[2] ** 2 + 2.0 * 9.81 * dz
                    if disc >= 0:
                        t_to_arm = (ball_vel[2] + float(np.sqrt(disc))) / 9.81
                        if t_to_arm > 0:
                            pred_x = ball_pos[0] + ball_vel[0] * t_to_arm
                            pred_y = ball_pos[1] + ball_vel[1] * t_to_arm
                            agent_xy = self.data.qpos[self.player_qpos_adr[i] : self.player_qpos_adr[i] + 2]
                            use_right = (pred_x - agent_xy[0]) > 0
                            uarm_id = self.player_ruarm_id[i] if use_right else self.player_luarm_id[i]
                            uarm_xyz = self.data.xpos[uarm_id]  # full 3D
                            pred_xyz = np.array([pred_x, pred_y, 1.5])
                            uarm_dist_3d = float(np.linalg.norm(uarm_xyz - pred_xyz))
                            # iter27: anticipation coeff 1.0→0.2. Max anticipation
                            # 0.2*19=3.8pts < contact(1.5)+win(5.0)=6.5pts. Contact
                            # now dominant. Reach now provides gradient from step 0
                            # (radius 3.5m), anticipation provides precision gradient
                            # when arm near pred_xyz (radius 0.35m). Two-stage signal.
                            rewards[i] += 0.2 * max(0.0, 1.0 - uarm_dist_3d / 0.35)

                            # Overshoot penalty: measure UPPER ARM past pred_x, not torso.
                            # iter20 bug: torso threshold let arm reach pred_x+0.18 before
                            # triggering, penalty 0.052/step < loco 0.1/step → loco won.
                            # Fix: penalize when uarm_x > pred_x (arm past ball) with
                            # coeff 1.0 > loco 0.1, so arm stays AT pred_x not past it.
                            arm_sign = 1.0 if use_right else -1.0
                            arm_past_pred = arm_sign * (float(uarm_xyz[0]) - pred_x)
                            if arm_past_pred > 0.05:
                                rewards[i] -= 1.0 * min(1.0, arm_past_pred - 0.05)


        self._prev_ball_pos = ball_pos.copy()
        self._prev_phase = phase

        # Record frame
        if self.record and self._recording is not None:
            self._recording.append(self._snapshot())

        obs = {i: self._get_obs(i) for i in range(TOTAL_PLAYERS)}
        terminated = {i: terminal for i in range(TOTAL_PLAYERS)}
        truncated = {i: timeout and not terminal for i in range(TOTAL_PLAYERS)}
        infos = {
            i: {
                "phase": int(phase),
                "winning_team": winning_team,
                "contact_player": contact_player,
                "step": self._step_count,
            }
            for i in range(TOTAL_PLAYERS)
        }

        return obs, rewards, terminated, truncated, infos

    def get_global_obs(self) -> np.ndarray:
        """Centralized critic observation: full state of all agents + ball."""
        parts = [self._get_ball_obs()]
        for i in range(TOTAL_PLAYERS):
            adr_q = self.player_qpos_adr[i]
            adr_v = self.player_qvel_adr[i]
            parts.append(self.data.qpos[adr_q : adr_q + PLAYER_QPOS_DIM].copy())
            parts.append(self.data.qvel[adr_v : adr_v + PLAYER_QVEL_DIM].copy())
        phase_oh = np.zeros(OBS_PHASE, dtype=np.float32)
        phase_oh[int(self.phase_tracker.phase)] = 1.0
        parts.append(phase_oh)
        score = np.zeros(2, dtype=np.float32)  # score tracking is external
        parts.append(score)
        touches = np.array(self.phase_tracker.team_touches, dtype=np.float32) / 3.0
        parts.append(touches)
        return np.concatenate(parts, dtype=np.float32)

    def get_recording(self) -> list[dict] | None:
        return self._recording

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_players(self, rotation: int = 0):
        all_pos = list(TEAM_A_START_POSITIONS) + list(TEAM_B_START_POSITIONS)
        rng = np.random.default_rng()
        for i, (x, y) in enumerate(all_pos):
            adr = self.player_qpos_adr[i]
            self.data.qpos[adr : adr + 3] = [x, y, START_HEIGHT]
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            # Small joint angle noise to break agent symmetry
            noise = rng.uniform(-0.05, 0.05, PLAYER_QPOS_DIM - 7).astype(np.float64)
            self.data.qpos[adr + 7 : adr + PLAYER_QPOS_DIM] = noise
        # Settle physics so feet land gently on ground (START_HEIGHT=1.31 → feet at z≈0)
        self.data.ctrl[:] = 0.0
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def _place_ball(self, serving_team: int):
        rng = self.np_random if hasattr(self, "np_random") and self.np_random is not None else np.random
        adr = self.ball_qpos_adr

        # Drop-serve: ball falls from z=3.5 at player_x ± 0.27m.
        # Geometry: torso+ball threshold=0.223m, 0.27>0.223 → no torso contact.
        # Arm at ±0.18m from center → gap=0.09m < arm+ball threshold 0.163m → arm contact.
        # Impact vel at z≈1.5: ~6.26 m/s → elastic gives +5.5 m/s up → max_z≈3.1m > NET_HEIGHT.
        if serving_team == 0:
            recv_pids = [7, 8, 9]   # team B front row
        else:
            recv_pids = [1, 2, 3]   # team A front row

        target_pid = int(rng.choice(recv_pids))
        arm_dir = float(rng.choice([-1.0, 1.0]))  # -1=left arm, +1=right arm

        # Query player root position after settle
        adr_q = self.player_qpos_adr[target_pid]
        player_x = float(self.data.qpos[adr_q])
        player_y = float(self.data.qpos[adr_q + 1])

        x_noise = float(rng.uniform(-0.02, 0.02))
        y_noise = float(rng.uniform(-0.02, 0.02))
        target_x = player_x + arm_dir * 0.27 + x_noise
        target_y = player_y + y_noise

        self.data.qpos[adr : adr + 3] = [target_x, target_y, 3.5]
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3] = [0.0, 0.0, 0.0]
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_qvel_adr + 3 : self.ball_qvel_adr + 6] = 0.0

    def _apply_magnus_force(self):
        """F_magnus = k * (omega × v), applied to ball body each physics step."""
        adr_v = self.ball_qvel_adr
        lin_vel = self.data.qvel[adr_v : adr_v + 3]
        ang_vel = self.data.qvel[adr_v + 3 : adr_v + 6]
        magnus = MAGNUS_COEFF * np.cross(ang_vel, lin_vel)
        self.data.xfrc_applied[self.ball_body_id, :3] = magnus

    def _detect_ball_contact(self) -> int:
        """Return global player index of agent who just touched the ball, or -1."""
        ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_g"
        )
        for k in range(self.data.ncon):
            con = self.data.contact[k]
            g1, g2 = con.geom1, con.geom2
            # Check if one geom is the ball
            if g1 != ball_geom_id and g2 != ball_geom_id:
                continue
            other = g2 if g1 == ball_geom_id else g1
            # Find which player body owns this geom
            body_id = self.model.geom_bodyid[other]
            for pid in range(TOTAL_PLAYERS):
                # Check if body is part of player pid by checking body name prefix
                bname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if bname and bname.startswith(f"p{pid}_"):
                    return pid
        return -1

    def _apply_elastic_collision(self, contact_player: int, ball_vel_pre: np.ndarray):
        """Apply elastic bounce to ball's VERTICAL velocity (z-axis).
        MuJoCo's angled contact already gives ball horizontal (x) velocity toward net.
        We override z-velocity so ball gets proper upward bounce (COR=0.887).
        v_ball_z_new = -0.887 × v_ball_z_pre + 1.887 × v_arm_z"""
        ELASTIC_COEFF = 0.887
        ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_g")

        for k in range(self.data.ncon):
            con = self.data.contact[k]
            g1, g2 = con.geom1, con.geom2
            if g1 != ball_geom_id and g2 != ball_geom_id:
                continue
            other_geom = g2 if g1 == ball_geom_id else g1
            body_id = self.model.geom_bodyid[other_geom]
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not (bname and bname.startswith(f"p{contact_player}_")):
                continue

            # Arm z-velocity (world frame): cvel = [angular(3), linear(3)]
            arm_vel_z = float(self.data.cvel[body_id][5])

            # Apply elastic to z-component only using pre-contact ball z-velocity
            vel_z_pre = float(ball_vel_pre[2])
            vel_z_new = -ELASTIC_COEFF * vel_z_pre + (1.0 + ELASTIC_COEFF) * arm_vel_z

            # Override z-velocity; keep x,y from MuJoCo's angled contact (provides net-direction vel)
            adr = self.ball_qvel_adr
            self.data.qvel[adr + 2] = vel_z_new
            return

    def _get_ball_pos(self) -> np.ndarray:
        adr = self.ball_qpos_adr
        return self.data.qpos[adr : adr + 3].copy()

    def _get_ball_vel(self) -> np.ndarray:
        adr = self.ball_qvel_adr
        return self.data.qvel[adr : adr + 3].copy()

    def _get_ball_spin(self) -> np.ndarray:
        adr = self.ball_qvel_adr
        return self.data.qvel[adr + 3 : adr + 6].copy()

    def _get_ball_obs(self) -> np.ndarray:
        return np.concatenate([
            self._get_ball_pos(),
            self._get_ball_vel(),
            self._get_ball_spin(),
        ], dtype=np.float32)

    def _get_player_pos(self, pid: int) -> np.ndarray:
        adr = self.player_qpos_adr[pid]
        return self.data.qpos[adr : adr + 3].copy()

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        team = 0 if agent_idx < PLAYERS_PER_TEAM else 1
        opp_team = 1 - team
        teammate_ids = [j for j in range(PLAYERS_PER_TEAM * team,
                                         PLAYERS_PER_TEAM * (team + 1))
                        if j != agent_idx]
        opp_ids = list(range(PLAYERS_PER_TEAM * opp_team,
                             PLAYERS_PER_TEAM * (opp_team + 1)))

        # Flip x-axis for team B so both teams see the court identically:
        # own side = positive x, net = x→0, opponent side = negative x.
        # Without this flip the shared policy must learn two opposite sign conventions.
        sx = -1.0 if team == 1 else 1.0

        # Own joint state
        adr_q = self.player_qpos_adr[agent_idx]
        adr_v = self.player_qvel_adr[agent_idx]
        own_qpos = self.data.qpos[adr_q : adr_q + PLAYER_QPOS_DIM].copy()
        own_qvel = self.data.qvel[adr_v : adr_v + PLAYER_QVEL_DIM].copy()
        own_qpos[0] *= sx   # torso x position
        own_qvel[0] *= sx   # torso x velocity

        # Teammates: root pos + vel + facing dir (3+3+3=9 each)
        tm_obs = []
        for j in teammate_ids:
            tadr_q = self.player_qpos_adr[j]
            tadr_v = self.player_qvel_adr[j]
            tpos = self.data.qpos[tadr_q : tadr_q + 3].copy()
            tvel = self.data.qvel[tadr_v : tadr_v + 3].copy()
            tpos[0] *= sx; tvel[0] *= sx
            tm_obs.extend(tpos)
            tm_obs.extend(tvel)
            # Facing: forward vector from quaternion (simplified as quat[1:4])
            tm_obs.extend(self.data.qpos[tadr_q + 4 : tadr_q + 7])

        # Opponents: root pos + vel (3+3=6 each)
        opp_obs = []
        for j in opp_ids:
            oadr_q = self.player_qpos_adr[j]
            oadr_v = self.player_qvel_adr[j]
            opos = self.data.qpos[oadr_q : oadr_q + 3].copy()
            ovel = self.data.qvel[oadr_v : oadr_v + 3].copy()
            opos[0] *= sx; ovel[0] *= sx
            opp_obs.extend(opos)
            opp_obs.extend(ovel)

        # Phase one-hot
        phase_oh = np.zeros(OBS_PHASE, dtype=np.float32)
        phase_oh[int(self.phase_tracker.phase)] = 1.0

        # Role one-hot
        role_oh = np.zeros(OBS_ROLE, dtype=np.float32)
        role_oh[int(self.roles[agent_idx])] = 1.0

        # Score placeholder (managed externally per episode batch)
        score = np.zeros(2, dtype=np.float32)

        # Touch counts
        touches = np.array(self.phase_tracker.team_touches, dtype=np.float32) / 3.0

        # Ball obs with x-flip applied
        ball_obs = self._get_ball_obs().copy()
        ball_obs[0] *= sx   # ball x position
        ball_obs[3] *= sx   # ball x velocity

        # own_qpos+own_qvel come FIRST to match locomotion-pretraining index layout
        # (loco training: obs[0:32]=qpos, obs[32:62]=qvel, rest zeros)
        obs = np.concatenate([
            own_qpos,
            own_qvel,
            ball_obs,
            np.array(tm_obs, dtype=np.float32),
            np.array(opp_obs, dtype=np.float32),
            phase_oh,
            role_oh,
            score,
            touches,
        ], dtype=np.float32)

        # Ensure correct shape (pad or trim if counts differ due to rounding)
        if obs.shape[0] != LOCAL_OBS_DIM:
            padded = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
            n = min(obs.shape[0], LOCAL_OBS_DIM)
            padded[:n] = obs[:n]
            return padded

        return obs

    def _snapshot(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ball_pos": self._get_ball_pos(),
            "ball_vel": self._get_ball_vel(),
            "ball_spin": self._get_ball_spin(),
            "phase": int(self.phase_tracker.phase),
        }
