"""Physics & predictor sanity tests — run these *before* training (plan §10).

    D:\\Code\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airhockey.config import TableConfig
from airhockey.physics import AirHockeyPhysics, GOAL_LEFT, GOAL_RIGHT, GOAL_NONE
from airhockey.predictor import TrajectoryPredictor, _fold
from airhockey.env import AirHockeyEnv, OBS_DIM


def _run(physics, n, tv0=(0, 0), tv1=(0, 0)):
    last = GOAL_NONE
    for _ in range(n):
        g = physics.step(np.array(tv0, float), np.array(tv1, float))
        if g != GOAL_NONE:
            last = g
            break
    return last


def test_no_tunnelling_fast_puck():
    """A very fast puck must not pass through a side wall."""
    cfg = TableConfig()
    phys = AirHockeyPhysics(cfg)
    phys.reset()
    phys.state.puck_pos = np.array([cfg.length / 2, cfg.width / 2])
    phys.state.puck_vel = np.array([0.0, 50.0])  # absurdly fast toward +y wall
    _run(phys, 200)
    r = cfg.puck_radius
    assert r - 1e-6 <= phys.state.puck_pos[1] <= cfg.width - r + 1e-6


def test_goal_detection():
    cfg = TableConfig()
    phys = AirHockeyPhysics(cfg)
    phys.reset()
    phys.state.puck_pos = np.array([0.3, cfg.width / 2])
    phys.state.puck_vel = np.array([-5.0, 0.0])  # straight at the left goal
    assert _run(phys, 100) == GOAL_LEFT

    phys.reset()
    phys.state.puck_pos = np.array([cfg.length - 0.3, cfg.width / 2])
    phys.state.puck_vel = np.array([5.0, 0.0])
    assert _run(phys, 100) == GOAL_RIGHT


def test_wall_bounce_not_goal():
    """A puck hitting the end wall away from the mouth must bounce, not score."""
    cfg = TableConfig()
    phys = AirHockeyPhysics(cfg)
    phys.reset()
    phys.state.puck_pos = np.array([0.3, cfg.puck_radius + 0.02])  # near the corner
    phys.state.puck_vel = np.array([-5.0, 0.0])
    assert _run(phys, 60) == GOAL_NONE
    assert phys.state.puck_vel[0] > 0  # reflected back


def test_mallet_imparts_velocity():
    """A moving mallet striking a still puck should send it flying."""
    cfg = TableConfig()
    phys = AirHockeyPhysics(cfg)
    phys.reset()
    phys.state.puck_pos = np.array([cfg.length * 0.3, cfg.width / 2])
    phys.state.puck_vel = np.array([0.0, 0.0])
    phys.state.mallet_pos[0] = np.array([cfg.length * 0.3 - 0.1, cfg.width / 2])
    # Drive the mallet hard toward +x for a moment.
    for _ in range(40):
        phys.step(np.array([cfg.mallet_max_speed, 0.0]), np.zeros(2))
    assert phys.state.puck_vel[0] > 1.0
    assert phys.state.last_touch == 0


def test_fold():
    assert abs(_fold(0.5, 0.0, 1.0) - 0.5) < 1e-9
    assert abs(_fold(1.2, 0.0, 1.0) - 0.8) < 1e-9   # reflected past the far wall
    assert abs(_fold(-0.3, 0.0, 1.0) - 0.3) < 1e-9  # reflected past the near wall
    assert abs(_fold(2.3, 0.0, 1.0) - 0.3) < 1e-9   # two reflections


def test_predictor_matches_sim():
    """Mirror-unfolding intercept should match a brute-force simulation."""
    cfg = TableConfig()
    pred = TrajectoryPredictor(cfg)
    phys = AirHockeyPhysics(cfg)
    phys.reset()
    pos = np.array([cfg.length * 0.7, cfg.width * 0.6])
    vel = np.array([-3.0, 2.2])  # heading toward the left wall, bouncing in y
    phys.state.puck_pos = pos.copy()
    phys.state.puck_vel = vel.copy()
    # The mirror-unfolding math assumes lossless, frictionless bounces and no
    # mallets; match that here so the test isolates the geometry (real play uses
    # e=0.9). Park both mallets in a corner well clear of the puck's path.
    phys.friction = 0.0
    phys.wall_e = 1.0
    phys.state.mallet_pos[0] = np.array([0.1, 0.1])
    phys.state.mallet_pos[1] = np.array([cfg.length - 0.1, 0.1])

    x_line = cfg.length * 0.12
    y_pred, t = pred.intercept(pos, vel, x_line)
    assert y_pred is not None
    # Simulate until the puck crosses x_line.
    while phys.state.puck_pos[0] > x_line:
        phys.step(np.zeros(2), np.zeros(2))
    assert abs(phys.state.puck_pos[1] - y_pred) < 0.05


def test_env_smoke():
    env = AirHockeyEnv(opponent="scripted")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        assert np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
