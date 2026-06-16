"""Opponents that drive mallet 1 (the right player).

An opponent is any callable ``opponent(physics) -> np.ndarray`` returning a
*target velocity in world coordinates* for mallet 1. Three are provided:

- :class:`ScriptedPredictor` — analytic defend/attack bot (plan §4.2). The first
  baseline the RL agent must beat, and the initial self-play sparring partner.
- :class:`StillOpponent` / :class:`RandomOpponent` — trivial baselines.
- :class:`PolicyOpponent` — wraps a frozen SB3 model for self-play. The field is
  mirrored so the model (trained as the left attacker) sees itself on the left.
"""

from __future__ import annotations

import numpy as np

from .config import TableConfig
from .physics import AirHockeyPhysics
from .predictor import TrajectoryPredictor


class StillOpponent:
    def __call__(self, physics: AirHockeyPhysics) -> np.ndarray:
        return np.zeros(2)


class RandomOpponent:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def __call__(self, physics: AirHockeyPhysics) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=2) * physics.cfg.mallet_max_speed


class ScriptedPredictor:
    """Defends its own goal by interception; attacks when it controls the puck.

    ``defends`` is the goal side this opponent protects: 1 = right goal (x=L),
    which is the normal role for mallet 1. (Parameterised so the same logic can
    drive a scripted *left* player for testing.)
    """

    def __init__(self, cfg: TableConfig | None = None, mallet: int = 1,
                 defends: int = 1, aggression: float = 1.0):
        self.cfg = cfg or TableConfig()
        self.pred = TrajectoryPredictor(self.cfg)
        self.mallet = mallet
        self.defends = defends
        self.aggression = aggression

    def __call__(self, physics: AirHockeyPhysics) -> np.ndarray:
        c = self.cfg
        s = physics.state
        puck = s.puck_pos
        pvel = s.puck_vel
        mpos = s.mallet_pos[self.mallet]

        attack_left_goal = self.defends == 1          # we attack the opposite goal
        own_goal_x = c.length if self.defends == 1 else 0.0
        defense_x = (c.length * 0.82) if self.defends == 1 else (c.length * 0.18)
        on_our_side = (puck[0] > c.half_x()) if self.defends == 1 else (puck[0] < c.half_x())
        incoming = (pvel[0] > 0.2) if self.defends == 1 else (pvel[0] < -0.2)

        max_v = c.mallet_max_speed

        if incoming:
            # Predict where the puck crosses our defence line and go block it.
            y_int, t = self.pred.intercept(puck, pvel, defense_x)
            if y_int is None:
                y_int = puck[1]
            target = np.array([defense_x, y_int])
        elif on_our_side:
            # We have the puck — line up *behind* it and strike toward their goal.
            aim_dir = self.pred.aim_point(puck, attack_left_goal) - puck
            aim_dir = aim_dir / (np.linalg.norm(aim_dir) + 1e-9)
            behind = puck - aim_dir * (c.puck_radius + c.mallet_radius)
            dist_behind = float(np.linalg.norm(behind - mpos))
            if dist_behind < (c.puck_radius + c.mallet_radius) * 1.2:
                # Lined up: drive *through* the puck for a strike.
                return aim_dir * max_v * self.aggression
            target = behind
        else:
            # Puck on their side — fall back to a guarding rest position.
            rest_x = (c.length * 0.86) if self.defends == 1 else (c.length * 0.14)
            target = np.array([rest_x, np.clip(puck[1], c.width * 0.3, c.width * 0.7)])

        # Proportional controller: velocity toward the target position.
        err = target - mpos
        vel = err * 8.0
        spd = float(np.linalg.norm(vel))
        if spd > max_v:
            vel *= max_v / spd
        return vel


def mirror_state(physics: AirHockeyPhysics):
    """Return (puck_pos, puck_vel, mallet0, mvel0, mallet1, mvel1) reflected in x.

    After mirroring, the right player (mallet 1) appears as the left player
    attacking +x, so a left-trained policy can act on it directly.
    """
    c = physics.cfg
    s = physics.state

    def mx(p):
        out = p.copy()
        out[0] = c.length - out[0]
        return out

    def mv(v):
        out = v.copy()
        out[0] = -out[0]
        return out

    puck_pos = mx(s.puck_pos)
    puck_vel = mv(s.puck_vel)
    # Mallet 1 becomes "our" (left) mallet; mallet 0 becomes the opponent.
    m_self_pos = mx(s.mallet_pos[1]); m_self_vel = mv(s.mallet_vel[1])
    m_opp_pos = mx(s.mallet_pos[0]); m_opp_vel = mv(s.mallet_vel[0])
    return puck_pos, puck_vel, m_self_pos, m_self_vel, m_opp_pos, m_opp_vel


class PolicyOpponent:
    """Drive mallet 1 with a frozen SB3 policy via field mirroring."""

    def __init__(self, model, cfg: TableConfig | None = None, deterministic: bool = True):
        self.model = model
        self.cfg = cfg or TableConfig()
        self.deterministic = deterministic
        self.pred = TrajectoryPredictor(self.cfg)

    def __call__(self, physics: AirHockeyPhysics) -> np.ndarray:
        from .env import build_observation  # local import to avoid a cycle
        obs = build_observation(self.cfg, self.pred, *mirror_state(physics))
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        # Action is a target velocity in the mirrored (left-attacking) frame.
        tv = np.asarray(action, float) * self.cfg.mallet_max_speed
        tv[0] = -tv[0]  # un-mirror back to world coordinates
        return tv
