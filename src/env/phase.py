"""
Volleyball phase state machine and rule enforcement.
Tracks ball possession, touch count, out-of-bounds, net faults.
"""
import numpy as np
from .constants import (
    Phase, PLAYERS_PER_TEAM, MAX_TEAM_TOUCHES,
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, ATTACK_LINE, BALL_RADIUS
)

COURT_HALF = COURT_LENGTH / 2   # 9.0
COURT_HALF_W = COURT_WIDTH / 2  # 4.5


class PhaseTracker:
    def __init__(self):
        self.reset()

    def reset(self, serving_team: int = 0):
        self.phase = Phase.SERVE
        self.serving_team = serving_team
        self.possession = serving_team       # which team last touched
        self.touch_count = [0, 0]            # touches this possession per team
        self.last_toucher = -1               # global player index
        self.consecutive_touches = 0         # same player touching consecutively
        self.serve_done = False
        self.fault_team = -1                 # team that committed a fault (-1 = none)
        self.fault_reason = ""

    def update(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        contact_player: int,           # global player index who just contacted (-1 = none)
    ) -> tuple[Phase, bool, int]:
        """
        Update phase state machine given current ball state and any contact.
        Returns (new_phase, terminal, winning_team).
        winning_team: 0 or 1 if terminal, else -1.
        """
        terminal = False
        winning_team = -1

        # Check ball out of bounds
        if self._is_out_of_bounds(ball_pos, ball_vel):
            terminal = True
            # Team that last touched loses the point
            winning_team = 1 - self.possession
            self.phase = Phase.DEAD
            return self.phase, terminal, winning_team

        # Check ball hit the floor
        if ball_pos[2] - BALL_RADIUS < 0.01 and np.linalg.norm(ball_vel) > 0.1:
            terminal = True
            # Team on whose side the ball lands loses
            ball_side = 0 if ball_pos[0] > 0 else 1
            winning_team = 1 - ball_side
            self.phase = Phase.DEAD
            return self.phase, terminal, winning_team

        # Process contact
        if contact_player >= 0:
            team = 0 if contact_player < PLAYERS_PER_TEAM else 1

            # Check for double touch (same player back-to-back)
            if contact_player == self.last_toucher and self.phase not in (Phase.SERVE, Phase.BLOCKED):
                terminal = True
                winning_team = 1 - team
                self.fault_team = team
                self.fault_reason = "double touch"
                self.phase = Phase.DEAD
                return self.phase, terminal, winning_team

            # Switch possession if new team touches
            if team != self.possession:
                self.possession = team
                self.touch_count[team] = 0

            self.touch_count[team] += 1
            self.last_toucher = contact_player

            # Too many touches
            if self.touch_count[team] > MAX_TEAM_TOUCHES:
                terminal = True
                winning_team = 1 - team
                self.fault_team = team
                self.fault_reason = "four touches"
                self.phase = Phase.DEAD
                return self.phase, terminal, winning_team

            # Advance phase
            self._advance_phase(team, ball_pos)

        return self.phase, terminal, winning_team

    def _advance_phase(self, touching_team: int, ball_pos: np.ndarray):
        touches = self.touch_count[touching_team]
        if self.phase == Phase.SERVE:
            self.phase = Phase.RECEIVE
            self.serve_done = True
        elif touches == 1:
            self.phase = Phase.RECEIVE
        elif touches == 2:
            self.phase = Phase.SET
        elif touches == 3:
            self.phase = Phase.ATTACK
        # After attack, next contact is on the other team -> RECEIVE again

    def _is_out_of_bounds(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        # Only flag out when ball is below net height and outside court
        # (avoid flagging mid-air trajectories that will land in)
        if pos[2] > NET_HEIGHT + 0.5:
            return False
        return (
            abs(pos[0]) > COURT_HALF + 0.15 or
            abs(pos[1]) > COURT_HALF_W + 0.15
        )

    @property
    def team_touches(self) -> list[int]:
        return self.touch_count.copy()
