"""2D air-hockey physics with fine substeps and continuous-ish collisions.

The simulation is deliberately *role agnostic*: it owns two mallets (0 = left,
1 = right) and a puck, and advances by one physics substep given each mallet's
target velocity. The :class:`~airhockey.env.AirHockeyEnv` decides who is the
learning agent and handles observations / mirroring.

Design notes (see plan §3.2 and §10):
- We integrate at ``physics_hz`` (default 200 Hz) and let the agent act at
  ``decision_hz`` (10 Hz), so each decision spans ``substeps`` integration steps.
- Mallets obey a max speed *and* a max acceleration — no teleporting.
- Puck–wall and puck–mallet collisions are resolved with positional correction
  so the puck never tunnels through a thin wall or a fast mallet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .config import TableConfig

# Goal event codes returned by :meth:`AirHockeyPhysics.step`.
GOAL_NONE = 0
GOAL_LEFT = 1   # puck entered the LEFT goal (x = 0)  -> right player scored
GOAL_RIGHT = 2  # puck entered the RIGHT goal (x = L) -> left player scored


@dataclass
class PhysicsState:
    puck_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    puck_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mallet_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))  # [mallet, xy]
    mallet_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    last_touch: int = -1  # which mallet last hit the puck (-1 = none)

    def copy(self) -> "PhysicsState":
        return PhysicsState(
            puck_pos=self.puck_pos.copy(),
            puck_vel=self.puck_vel.copy(),
            mallet_pos=self.mallet_pos.copy(),
            mallet_vel=self.mallet_vel.copy(),
            last_touch=self.last_touch,
        )


class AirHockeyPhysics:
    """Stateful 2D air-hockey simulator.

    Parameters can be overridden per-reset for domain randomization (plan §5.4).
    """

    def __init__(self, cfg: TableConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or TableConfig()
        self.rng = rng or np.random.default_rng()
        self.state = PhysicsState()
        # Per-episode (possibly randomized) parameters.
        self.friction = self.cfg.puck_friction
        self.wall_e = self.cfg.wall_restitution
        self.mallet_e = self.cfg.mallet_restitution
        self.reset()

    # ------------------------------------------------------------------ setup
    def mallet_x_bounds(self, mallet: int) -> tuple[float, float]:
        c = self.cfg
        r = c.mallet_radius
        if mallet == 0:  # left half
            return (r, c.half_x())
        return (c.half_x(), c.length - r)

    def home_pos(self, mallet: int) -> np.ndarray:
        c = self.cfg
        x = c.length * 0.15 if mallet == 0 else c.length * 0.85
        return np.array([x, c.width / 2.0])

    def reset(self, randomize: bool = False, serve_to: int | None = None) -> PhysicsState:
        c = self.cfg
        s = PhysicsState()
        s.mallet_pos = np.stack([self.home_pos(0), self.home_pos(1)])
        s.mallet_vel = np.zeros((2, 2))

        # Puck starts near the centre with a random serve velocity.
        s.puck_pos = np.array([c.length / 2.0, c.width / 2.0])
        speed = self.rng.uniform(1.5, 4.0)
        angle = self.rng.uniform(-np.pi, np.pi)
        if serve_to == 0:      # serve toward the left (agent) goal
            angle = np.pi + self.rng.uniform(-0.6, 0.6)
        elif serve_to == 1:    # serve toward the right (opponent) goal
            angle = self.rng.uniform(-0.6, 0.6)
        s.puck_vel = np.array([np.cos(angle), np.sin(angle)]) * speed
        s.last_touch = -1
        self.state = s

        if randomize:
            self.friction = self.cfg.puck_friction * self.rng.uniform(0.5, 1.8)
            self.wall_e = float(np.clip(self.cfg.wall_restitution * self.rng.uniform(0.9, 1.05), 0.5, 0.99))
            self.mallet_e = float(np.clip(self.cfg.mallet_restitution * self.rng.uniform(0.9, 1.05), 0.5, 0.99))
        else:
            self.friction = self.cfg.puck_friction
            self.wall_e = self.cfg.wall_restitution
            self.mallet_e = self.cfg.mallet_restitution
        return self.state

    # ------------------------------------------------------------------ step
    def step(self, target_vel0: np.ndarray, target_vel1: np.ndarray) -> int:
        """Advance one physics substep. Returns a GOAL_* event code."""
        c = self.cfg
        dt = c.physics_dt
        s = self.state

        targets = np.stack([np.asarray(target_vel0, float), np.asarray(target_vel1, float)])
        self._integrate_mallets(targets, dt)

        # Move puck.
        s.puck_pos = s.puck_pos + s.puck_vel * dt

        self._wall_collisions_y(dt)
        for m in range(2):
            self._mallet_collision(m)
        goal = self._x_walls_and_goals()

        # Friction + speed clamp.
        s.puck_vel *= np.exp(-self.friction * dt)
        spd = float(np.linalg.norm(s.puck_vel))
        if spd > c.puck_max_speed:
            s.puck_vel *= c.puck_max_speed / spd
        return goal

    def _integrate_mallets(self, targets: np.ndarray, dt: float) -> None:
        c = self.cfg
        s = self.state
        # Clamp target speed.
        tnorm = np.linalg.norm(targets, axis=1, keepdims=True)
        scale = np.where(tnorm > c.mallet_max_speed, c.mallet_max_speed / np.maximum(tnorm, 1e-9), 1.0)
        targets = targets * scale

        for m in range(2):
            dv = targets[m] - s.mallet_vel[m]
            dv_mag = float(np.linalg.norm(dv))
            max_dv = c.mallet_max_accel * dt
            if dv_mag > max_dv:
                dv *= max_dv / dv_mag
            s.mallet_vel[m] = s.mallet_vel[m] + dv
            new_pos = s.mallet_pos[m] + s.mallet_vel[m] * dt

            # Confine to own half + table (hard clamp, kill the offending vel comp).
            xlo, xhi = self.mallet_x_bounds(m)
            ylo, yhi = c.mallet_radius, c.width - c.mallet_radius
            if new_pos[0] < xlo:
                new_pos[0] = xlo; s.mallet_vel[m, 0] = 0.0
            elif new_pos[0] > xhi:
                new_pos[0] = xhi; s.mallet_vel[m, 0] = 0.0
            if new_pos[1] < ylo:
                new_pos[1] = ylo; s.mallet_vel[m, 1] = 0.0
            elif new_pos[1] > yhi:
                new_pos[1] = yhi; s.mallet_vel[m, 1] = 0.0
            s.mallet_pos[m] = new_pos

    def _wall_collisions_y(self, dt: float) -> None:
        c = self.cfg
        s = self.state
        r = c.puck_radius
        if s.puck_pos[1] < r:
            s.puck_pos[1] = r
            s.puck_vel[1] = abs(s.puck_vel[1]) * self.wall_e
        elif s.puck_pos[1] > c.width - r:
            s.puck_pos[1] = c.width - r
            s.puck_vel[1] = -abs(s.puck_vel[1]) * self.wall_e

    def _x_walls_and_goals(self) -> int:
        c = self.cfg
        s = self.state
        r = c.puck_radius
        x, y = s.puck_pos
        in_goal_y = c.goal_y_min < y < c.goal_y_max

        if x < r:
            if in_goal_y:
                if x <= 0.0:
                    return GOAL_LEFT
                return GOAL_NONE  # let it travel into the mouth
            s.puck_pos[0] = r
            s.puck_vel[0] = abs(s.puck_vel[0]) * self.wall_e
        elif x > c.length - r:
            if in_goal_y:
                if x >= c.length:
                    return GOAL_RIGHT
                return GOAL_NONE
            s.puck_pos[0] = c.length - r
            s.puck_vel[0] = -abs(s.puck_vel[0]) * self.wall_e
        return GOAL_NONE

    def _mallet_collision(self, m: int) -> None:
        c = self.cfg
        s = self.state
        rsum = c.puck_radius + c.mallet_radius
        delta = s.puck_pos - s.mallet_pos[m]
        dist = float(np.linalg.norm(delta))
        if dist >= rsum or dist < 1e-9:
            if dist < 1e-9:
                # Degenerate overlap: shove puck along +x.
                s.puck_pos = s.mallet_pos[m] + np.array([rsum, 0.0])
            return

        n = delta / dist
        v_rel = s.puck_vel - s.mallet_vel[m]
        vn = float(v_rel @ n)
        if vn < 0:  # approaching — reflect relative normal velocity (mallet ~ infinite mass)
            s.puck_vel = s.puck_vel - (1.0 + self.mallet_e) * vn * n
        # Positional correction so they no longer overlap.
        s.puck_pos = s.mallet_pos[m] + n * rsum
        s.last_touch = m
