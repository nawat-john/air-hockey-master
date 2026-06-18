"""Gymnasium environment: single learning agent (left mallet) vs an opponent.

- Decision rate 10 Hz; each :meth:`step` runs ``cfg.substeps`` physics substeps
  (plan §3.2). The opponent picks its target once per decision.
- The agent always defends the LEFT goal and attacks +x, so no per-agent
  mirroring is needed; self-play opponents are mirrored instead (see
  :class:`airhockey.opponents.PolicyOpponent`).
- Observation includes analytic *predicted* features (intercept point + time),
  the lever that makes learning fast and strong (plan §3.3, §4.3).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import TableConfig
from .physics import AirHockeyPhysics, GOAL_NONE, GOAL_LEFT, GOAL_RIGHT
from .predictor import TrajectoryPredictor

Opponent = Callable[[AirHockeyPhysics], np.ndarray]

OBS_DIM = 16


def _norm_pos(cfg: TableConfig, p: np.ndarray) -> np.ndarray:
    return np.array([p[0] / cfg.length * 2 - 1, p[1] / cfg.width * 2 - 1])


def build_observation(cfg: TableConfig, pred: TrajectoryPredictor,
                      puck_pos, puck_vel, self_pos, self_vel, opp_pos, opp_vel) -> np.ndarray:
    """Build the 16-d observation from a LEFT-attacker perspective.

    Shared by the env and by mirrored self-play opponents so both see an
    identical feature layout.
    """
    vmax = cfg.mallet_max_speed
    pvmax = cfg.puck_max_speed

    defense_x = cfg.length * 0.12  # the agent's guard line in front of x=0
    y_int, t = pred.intercept(puck_pos, puck_vel, defense_x)
    if y_int is None:
        y_int, t = float(puck_pos[1]), 1.5
    int_norm = y_int / cfg.width * 2 - 1
    t_norm = float(np.clip(t / 1.5, 0.0, 1.0))  # cap at 1.5 s lookahead

    ahead = pred.position_at(puck_pos, puck_vel, 0.2)  # short-horizon lookahead

    obs = np.concatenate([
        _norm_pos(cfg, puck_pos),
        np.clip(puck_vel / pvmax, -1, 1),
        _norm_pos(cfg, self_pos),
        np.clip(self_vel / vmax, -1, 1),
        _norm_pos(cfg, opp_pos),
        np.clip(opp_vel / vmax, -1, 1),
        [int_norm, t_norm * 2 - 1],
        _norm_pos(cfg, ahead),
    ]).astype(np.float32)
    return obs


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: TableConfig | None = None,
                 opponent: Opponent | str = "scripted",
                 max_episode_seconds: float = 20.0,
                 action_mode: str = "velocity",
                 shaping: bool = True,
                 randomize: bool = False,
                 serve_mode: str = "random",
                 render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or TableConfig()
        self.pred = TrajectoryPredictor(self.cfg)
        self.physics = AirHockeyPhysics(self.cfg)
        self.action_mode = action_mode
        self.shaping = shaping
        self.randomize = randomize
        self.serve_mode = serve_mode
        self.render_mode = render_mode
        self.max_steps = int(round(max_episode_seconds * self.cfg.decision_hz))

        self._opponent_spec = opponent
        self.opponent = self._make_opponent(opponent)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._steps = 0
        self._idle_counter = 0
        self._renderer = None

    def _make_opponent(self, opponent: Opponent | str) -> Opponent:
        if isinstance(opponent, str):
            if opponent == "scripted":
                from .opponents import ScriptedPredictor
                return ScriptedPredictor(self.cfg)
            if opponent == "still":
                from .opponents import StillOpponent
                return StillOpponent()
            if opponent == "random":
                from .opponents import RandomOpponent
                return RandomOpponent(self.np_random)
            raise ValueError(f"unknown opponent spec: {opponent}")
        return opponent

    def set_opponent(self, opponent: Opponent | str) -> None:
        """Swap the opponent (used by self-play / league sampling)."""
        self._opponent_spec = opponent
        self.opponent = self._make_opponent(opponent)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.physics.rng = self.np_random
        if self.serve_mode == "to_agent":
            serve_to = 0
        elif self.serve_mode == "to_opponent":
            serve_to = 1
        else:
            serve_to = int(self.np_random.integers(0, 2))
        self.physics.reset(randomize=self.randomize, serve_to=serve_to)
        if isinstance(self._opponent_spec, str) and self._opponent_spec == "random":
            self.opponent = self._make_opponent("random")
        self._steps = 0
        self._idle_counter = 0
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        s = self.physics.state
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        prev_touch = s.last_touch

        tv1 = np.asarray(self.opponent(self.physics), dtype=float)  # held over substeps

        goal = GOAL_NONE
        for _ in range(cfg.substeps):
            tv0 = self._agent_target_vel(action)
            goal = self.physics.step(tv0, tv1)
            if goal != GOAL_NONE:
                break

        self._steps += 1
        reward, terminated = self._reward(goal, prev_touch)
        truncated = self._steps >= self.max_steps and not terminated

        info = {}
        if goal == GOAL_RIGHT:
            info["scored"] = "agent"
        elif goal == GOAL_LEFT:
            info["scored"] = "own" if s.last_touch == 0 else "opponent"

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    # --------------------------------------------------------------- helpers
    def _agent_target_vel(self, action: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if self.action_mode == "velocity":
            return action * cfg.mallet_max_speed
        # position mode: action maps to a point in the agent's half; P-controller.
        xlo, xhi = self.physics.mallet_x_bounds(0)
        ylo, yhi = cfg.mallet_radius, cfg.width - cfg.mallet_radius
        tx = xlo + (action[0] * 0.5 + 0.5) * (xhi - xlo)
        ty = ylo + (action[1] * 0.5 + 0.5) * (yhi - ylo)
        err = np.array([tx, ty]) - self.physics.state.mallet_pos[0]
        vel = err * 10.0
        spd = float(np.linalg.norm(vel))
        if spd > cfg.mallet_max_speed:
            vel *= cfg.mallet_max_speed / spd
        return vel

    def _obs(self) -> np.ndarray:
        s = self.physics.state
        return build_observation(
            self.cfg, self.pred,
            s.puck_pos, s.puck_vel,
            s.mallet_pos[0], s.mallet_vel[0],
            s.mallet_pos[1], s.mallet_vel[1],
        )

    def _reward(self, goal: int, prev_touch: int) -> tuple[float, bool]:
        cfg = self.cfg
        s = self.physics.state

        if goal == GOAL_RIGHT:
            return 10.0, True
        if goal == GOAL_LEFT:
            return (-15.0 if s.last_touch == 0 else -10.0), True

        if not self.shaping:
            return 0.0, False

        r = 0.0
        # New contact by the agent → possession + reward shooting toward +x.
        if s.last_touch == 0 and prev_touch != 0:
            r += 0.1
            r += 0.3 * max(0.0, float(s.puck_vel[0])) / cfg.puck_max_speed
            # Dense counterpart to the sparse own-goal penalty: punish striking
            # the puck back toward our own goal (−x). The 30.5% UTD4 policy spiked
            # to 52 own goals/200 eps from reckless strikes; the sparse −15 alone
            # was too weak a signal. Penalty > the +x bonus so forward always wins.
            r -= 0.4 * max(0.0, -float(s.puck_vel[0])) / cfg.puck_max_speed

        # Defensive positioning: when the puck is incoming on our side, reward
        # being between the puck and our goal (in y).
        if s.puck_vel[0] < -0.2 and s.puck_pos[0] < cfg.half_x():
            y_int, _ = self.pred.intercept(s.puck_pos, s.puck_vel, cfg.length * 0.12)
            if y_int is not None:
                err = abs(s.mallet_pos[0, 1] - y_int) / cfg.width
                r += 0.02 * (1.0 - err)

        # Anti-stall: penalise leaving the puck nearly still on our side.
        if float(np.linalg.norm(s.puck_vel)) < 0.3 and s.puck_pos[0] < cfg.half_x():
            self._idle_counter += 1
            if self._idle_counter > 10:
                r -= 0.05
        else:
            self._idle_counter = 0

        return r, False

    # --------------------------------------------------------------- render
    def render(self):
        if self._renderer is None:
            from .render import Renderer
            self._renderer = Renderer(self.cfg, mode=self.render_mode)
        return self._renderer.draw(self.physics)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
