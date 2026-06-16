"""Evaluate a trained agent's win rate and sub-metrics (plan §9).

    python scripts/evaluate.py --model runs/sac_final.zip --episodes 200
    python scripts/evaluate.py --model runs/sac_final.zip --opponent scripted --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC

from airhockey.config import TableConfig
from airhockey.env import AirHockeyEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--opponent", default="scripted", choices=["scripted", "still", "random"])
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    cfg = TableConfig()
    env = AirHockeyEnv(cfg=cfg, opponent=args.opponent, shaping=False,
                       render_mode="human" if args.render else None)
    model = SAC.load(args.model, device="cpu")

    wins = losses = draws = own_goals = 0
    shot_speeds = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            s = env.physics.state
            if s.last_touch == 0:
                shot_speeds.append(float(np.linalg.norm(s.puck_vel)))
            done = term or trunc
            scored = info.get("scored")
            if scored == "agent":
                wins += 1
            elif scored == "opponent":
                losses += 1
            elif scored == "own":
                losses += 1
                own_goals += 1
        if not (info.get("scored")):
            draws += 1

    n = args.episodes
    print(f"\n=== vs {args.opponent} over {n} episodes ===")
    print(f"win rate : {wins / n:.1%}  ({wins}W / {losses}L / {draws}D)")
    print(f"own goals: {own_goals}")
    if shot_speeds:
        print(f"avg agent-contact puck speed: {np.mean(shot_speeds):.2f} m/s")
    env.close()


if __name__ == "__main__":
    main()
