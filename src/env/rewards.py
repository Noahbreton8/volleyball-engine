"""
Role-specific reward functions. See ADR-004 for the full hierarchy.
All functions return a scalar float reward for one agent per timestep.
"""
import numpy as np
from .constants import (
    Role, Phase, SET_ZONE_CENTER, SET_ZONE_RADIUS,
    ATTACK_WINDOW_X_RANGE, ATTACK_WINDOW_Z_RANGE,
    COURT_HALF, COURT_HALF_W, PLAYERS_PER_TEAM
)


def _set_zone_for_team(team: int) -> np.ndarray:
    """Set zone center in world coords. Team 0 = x>0 side, Team 1 = x<0 side."""
    center = SET_ZONE_CENTER.copy()
    if team == 1:
        center[0] = -center[0]
    return center


def locomotion_reward(
    agent_pos: np.ndarray,
    ball_pos: np.ndarray,
    team: int,
    role: Role,
    phase: Phase,
) -> float:
    """Reward for moving toward the ball when it's in the agent's zone."""
    # Only reward if ball is on agent's side and phase is relevant
    ball_on_our_side = (ball_pos[0] > 0) == (team == 0)
    if not ball_on_our_side:
        return 0.0
    if phase not in (Phase.RECEIVE, Phase.SET, Phase.FREE_BALL, Phase.SERVE):
        return 0.0

    dist = np.linalg.norm(agent_pos[:2] - ball_pos[:2])
    # Restored to 0.1: iter19 experiments showed loco=0.02 is too weak — agents need
    # small locomotion signal to close the ~0.03m contact gap (torso must move ~0.05-0.10m
    # toward ball). Overshoot is now handled by targeted penalty in volleyball_env.py
    # instead of blanket locomotion reduction.
    return 0.1 * np.exp(-dist * 0.5)


def pass_reward(
    ball_vel_post: np.ndarray,
    ball_pos_post: np.ndarray,
    team: int,
) -> float:
    """
    Reward for a back-row player (libero/OH_BACK/OPP_BACK/MB_BACK) making a
    good pass. Measures cosine similarity between actual outgoing ball vector
    and ideal vector pointing to the set zone.
    """
    set_zone = _set_zone_for_team(team)
    ideal_dir = set_zone - ball_pos_post
    ideal_norm = np.linalg.norm(ideal_dir)
    if ideal_norm < 1e-6:
        return 0.0
    ideal_dir /= ideal_norm

    vel_norm = np.linalg.norm(ball_vel_post)
    if vel_norm < 0.5:
        return 0.0   # ball barely moving, no meaningful pass

    actual_dir = ball_vel_post / vel_norm
    cosine_sim = float(np.dot(actual_dir, ideal_dir))
    # Scale: max 1.0 for perfect direction, 0 for perpendicular, -0.5 for opposite
    return max(0.0, cosine_sim) * 1.0


def set_reward(
    ball_apex_pos: np.ndarray,
    team: int,
) -> float:
    """
    Reward for setter: ball apex must be within the attack window near the net.
    attack_window_x: distance from net (0 = at net), z: height range.
    Team 0 attack is toward x=0 from x>0; team 1 is toward x=0 from x<0.
    """
    # Net is at x=0; team 0 attacks from x>0, so attack window is x in [0.3, 1.5]
    dist_from_net = abs(ball_apex_pos[0])
    height = ball_apex_pos[2]

    x_min, x_max = ATTACK_WINDOW_X_RANGE
    z_min, z_max = ATTACK_WINDOW_Z_RANGE

    x_ok = x_min <= dist_from_net <= x_max
    z_ok = z_min <= height <= z_max

    if x_ok and z_ok:
        # Bonus for centering within the window
        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2
        x_err = abs(dist_from_net - x_center) / (x_max - x_min)
        z_err = abs(height - z_center) / (z_max - z_min)
        quality = 1.0 - 0.5 * (x_err + z_err)
        return max(0.3, quality) * 2.0

    # Partial credit based on how close to window
    x_dist = max(0.0, max(x_min - dist_from_net, dist_from_net - x_max))
    z_dist = max(0.0, max(z_min - height, height - z_max))
    miss = x_dist + z_dist
    return max(0.0, 0.5 - miss * 0.5)


def attack_reward(
    ball_landing_pos: np.ndarray,
    opponent_positions: np.ndarray,
    team: int,
) -> float:
    """
    Reward for attacker: ball lands in-bounds on opponent's side, away from defenders.
    """
    # Opponent side: team 0 attacks toward x<0, team 1 toward x>0
    opp_sign = -1 if team == 0 else 1
    ball_x = ball_landing_pos[0]

    # Must land on opponent's side
    if np.sign(ball_x) != opp_sign:
        return -0.5   # hit into own side / net

    # Must be in bounds
    if abs(ball_x) > COURT_HALF or abs(ball_landing_pos[1]) > COURT_HALF_W:
        return -0.3   # out of bounds

    base = 1.5  # base reward for in-bounds attack on opponent side

    # Bonus for landing away from defenders
    if opponent_positions is not None and len(opponent_positions) > 0:
        dists = np.linalg.norm(
            opponent_positions[:, :2] - ball_landing_pos[:2], axis=1
        )
        min_dist = np.min(dists)
        # Max bonus 1.0 for landing >3m from nearest defender
        bonus = min(1.0, min_dist / 3.0)
        base += bonus

    return base


def block_reward(
    ball_vel_post: np.ndarray,
    ball_pos_post: np.ndarray,
    team: int,
) -> float:
    """Reward for front-row MB/OPP blocking: redirect ball back to opponent side."""
    opp_sign = -1 if team == 0 else 1  # team 0 opponent is at x<0, so ball must go negative
    if ball_vel_post[0] * opp_sign > 1.0:
        return 1.0  # ball is moving toward opponent after block
    return 0.0


def upright_reward(agent_pos: np.ndarray) -> float:
    """Keep agents grounded at standing height (~1.29m). Penalizes falling AND jumping."""
    z = agent_pos[2]
    if z < 0.8:
        return -1.0   # hard falling penalty
    if z > 2.0:
        return -0.5   # jumping/flying penalty
    # Gaussian around ideal standing height ~1.29m (standard humanoid torso after settle)
    deviation = abs(z - 1.29)
    return 0.1 * max(0.0, 1.0 - deviation / 0.6)


def compute_agent_reward(
    agent_idx: int,
    role: Role,
    phase: Phase,
    phase_prev: Phase,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    ball_pos_prev: np.ndarray,
    agent_pos: np.ndarray,
    opponent_positions: np.ndarray,
    contact_happened: bool,
    rally_terminal: bool,
    won_rally: bool,
    team: int,
) -> float:
    """
    Master reward function. Called every control step for each agent.
    """
    reward = 0.0
    team = 0 if agent_idx < PLAYERS_PER_TEAM else 1

    # --- Upright survival bonus: always active (iter-2 improvement) ---
    reward += upright_reward(agent_pos)

    # --- Locomotion: always active ---
    reward += locomotion_reward(agent_pos, ball_pos, team, role, phase)

    # --- Dense ball-toward-net reward (iter-5 improvement) ---
    # When ball is on team's own side and moving toward the net (opponent), reward it.
    opp_dir = -1.0 if team == 0 else 1.0   # direction toward opponent's net
    ball_on_own_side = (ball_pos[0] > 0) == (team == 0)
    if ball_on_own_side:
        vel_toward_net = ball_vel[0] * opp_dir   # positive when heading toward opponent
        if vel_toward_net > 0:
            reward += 0.04 * vel_toward_net

        # Height-near-net reward: shaped trajectory signal for net clearance.
        # Ball needs z > NET_HEIGHT (2.43m) at x=0 to clear. Reward height when
        # ball is within 3m of net and above 1.2m — directly shapes the trajectory.
        dist_to_net = abs(ball_pos[0])
        if dist_to_net < 3.0 and ball_pos[2] > 1.2:
            height_above_floor = max(0.0, ball_pos[2] - 1.2)
            proximity = max(0.0, 1.0 - dist_to_net / 3.0)
            reward += 0.03 * height_above_floor * proximity

    # --- Contact rewards (only on the step a contact occurred) ---
    if contact_happened:
        # iter27: contact 1.0→1.5. With anticipation max 3.8pts, contact+win
        # (1.5+5.0=6.5) now clearly dominates — agents must contact to win.
        reward += 1.5

        # Vertical velocity shaping: penalize downward (ball hits ground), reward upward (clears net).
        if ball_vel[2] < 0:
            reward -= 0.3 * min(2.0, abs(ball_vel[2]))  # max -0.6
        else:
            reward += 0.15 * min(2.0, ball_vel[2])      # max +0.3 for upward contact

        back_row_roles = {Role.LIBERO, Role.OH_BACK, Role.OPP_BACK, Role.MB_BACK}

        opp_sign = -1 if team == 0 else 1  # positive = ball toward opponent

        if role in back_row_roles:
            # Pass: evaluate outgoing vector toward set zone
            reward += pass_reward(ball_vel, ball_pos, team) * 1.0

        elif role == Role.SETTER:
            # Set quality evaluated when ball reaches apex (approximated here
            # as height higher than contact point + 0.3m)
            if ball_pos[2] > ball_pos_prev[2] + 0.1:
                reward += set_reward(ball_pos, team) * 1.5

        elif role in {Role.OH_FRONT, Role.OPP_FRONT}:
            # Attack: reward fast hit toward opponent. DO NOT call attack_reward(ball_pos)
            # here — ball is still on own side at contact time, so position check always fails.
            vel_toward_opp = ball_vel[0] * opp_sign
            if vel_toward_opp > 1.0:
                reward += min(1.5, vel_toward_opp * 0.3)
                # Defender-avoidance bonus
                if opponent_positions is not None and len(opponent_positions) > 0:
                    dists = np.linalg.norm(
                        opponent_positions[:, :2] - ball_pos[:2], axis=1
                    )
                    reward += min(0.5, np.min(dists) / 3.0)

        elif role == Role.MB_FRONT:
            if phase == Phase.BLOCKED:
                reward += block_reward(ball_vel, ball_pos, team)
            else:
                vel_toward_opp = ball_vel[0] * opp_sign
                if vel_toward_opp > 1.0:
                    reward += min(1.2, vel_toward_opp * 0.3) * 0.8

    # --- Phase completion bonus ---
    if phase_prev == Phase.RECEIVE and phase == Phase.SET:
        reward += 0.2  # successful pass-to-set chain
    if phase_prev == Phase.SET and phase == Phase.ATTACK:
        reward += 0.3  # successful set-to-attack chain

    # --- Terminal reward ---
    if rally_terminal:
        reward += 5.0 if won_rally else -5.0

    return float(reward)
