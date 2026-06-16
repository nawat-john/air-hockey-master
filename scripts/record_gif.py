"""Record a GIF of the agent playing (plan §7, day 2 evening).

    python scripts/record_gif.py --model runs/sac_final.zip --out demo.gif --seconds 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC

from airhockey.config import TableConfig
from airhockey.env import AirHockeyEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="if omitted, the scripted bot plays itself")
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--opponent", default="scripted")
    args = ap.parse_args()

    cfg = TableConfig()
    env = AirHockeyEnv(cfg=cfg, opponent=args.opponent, shaping=False, render_mode="rgb_array")

    if args.model:
        model = SAC.load(args.model, device="cpu")
        def policy(obs):
            a, _ = model.predict(obs, deterministic=True)
            return a
    else:
        from airhockey.opponents import ScriptedPredictor, mirror_state
        from airhockey.env import build_observation
        left_bot = ScriptedPredictor(cfg, mallet=0, defends=0)
        def policy(obs):  # drive the left mallet with the scripted bot directly
            tv = left_bot(env.physics)
            return np.clip(tv / cfg.mallet_max_speed, -1, 1)

    frames = []
    obs, _ = env.reset(seed=0)
    steps = int(args.seconds * cfg.decision_hz)
    for _ in range(steps):
        action = policy(obs)
        obs, r, term, trunc, info = env.step(action)
        frames.append(env.render())
        if term or trunc:
            obs, _ = env.reset()

    imageio.mimsave(args.out, frames, fps=int(cfg.decision_hz), loop=0)
    print(f"wrote {args.out} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
