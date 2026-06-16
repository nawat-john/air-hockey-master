"""Trajectory predictor — the "secret sauce" from plan §4.

Uses the *mirror unfolding* trick: reflecting the table across its side walls
turns a path that bounces many times into a straight line in the "unfolded"
world. We then fold the answer back into the real table. This gives an analytic
intercept point (where the puck crosses a defence line) and aim targets for
direct and bank shots — far cheaper and more accurate than step simulation.
"""

from __future__ import annotations

import numpy as np

from .config import TableConfig


def _fold(value: float, lo: float, hi: float) -> float:
    """Reflect ``value`` back and forth into the band ``[lo, hi]`` (sawtooth)."""
    span = hi - lo
    if span <= 0:
        return lo
    z = (value - lo) % (2 * span)
    return lo + z if z < span else lo + (2 * span - z)


class TrajectoryPredictor:
    def __init__(self, cfg: TableConfig | None = None):
        self.cfg = cfg or TableConfig()

    # ----------------------------------------------------------- positions
    def position_at(self, pos: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
        """Puck centre position after ``t`` seconds, including side-wall bounces.

        Ignores friction and mallets — a short-horizon kinematic estimate.
        """
        ylo, yhi = self.cfg.puck_y_band
        x = pos[0] + vel[0] * t
        x = float(np.clip(x, self.cfg.puck_radius, self.cfg.length - self.cfg.puck_radius))
        y = _fold(pos[1] + vel[1] * t, ylo, yhi)
        return np.array([x, y])

    def trajectory(self, pos: np.ndarray, vel: np.ndarray, horizon: float, n: int = 30) -> np.ndarray:
        ts = np.linspace(0.0, horizon, n)
        return np.stack([self.position_at(pos, vel, t) for t in ts])

    # ----------------------------------------------------------- intercept
    def intercept(self, pos: np.ndarray, vel: np.ndarray, x_line: float
                  ) -> tuple[float | None, float | None]:
        """Where/when the puck centre reaches the vertical line ``x = x_line``.

        Returns ``(y, t)`` or ``(None, None)`` if the puck is not heading there.
        """
        vx = float(vel[0])
        if abs(vx) < 1e-6:
            return None, None
        t = (x_line - pos[0]) / vx
        if t <= 0:
            return None, None
        ylo, yhi = self.cfg.puck_y_band
        y = _fold(pos[1] + vel[1] * t, ylo, yhi)
        return y, t

    # ----------------------------------------------------------- aiming
    def aim_point(self, puck_pos: np.ndarray, attack_left_goal: bool,
                  bank: bool = False) -> np.ndarray:
        """Target point the puck's outgoing velocity should pass through.

        ``attack_left_goal``: True to shoot at x=0 goal, False for x=L goal.
        ``bank``: aim at the goal's mirror image across the nearest side wall.
        """
        c = self.cfg
        goal_x = 0.0 if attack_left_goal else c.length
        goal_y = c.width / 2.0
        if not bank:
            return np.array([goal_x, goal_y])
        # Mirror the goal across whichever side wall is farther from the puck,
        # producing a one-bounce bank shot.
        mirror_y = 2 * c.width - goal_y if puck_pos[1] < c.width / 2 else -goal_y
        return np.array([goal_x, mirror_y])

    def strike_velocity(self, puck_pos: np.ndarray, attack_left_goal: bool,
                        speed: float, bank: bool = False) -> np.ndarray:
        """Desired puck velocity to send it toward the (possibly mirrored) goal."""
        target = self.aim_point(puck_pos, attack_left_goal, bank)
        direction = target - puck_pos
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-9:
            direction = np.array([-1.0 if attack_left_goal else 1.0, 0.0])
            nrm = 1.0
        return direction / nrm * speed
