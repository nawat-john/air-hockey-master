"""Play air hockey with the mouse against the scripted bot (or a trained model).

    python scripts/play_human.py                 # vs scripted predictor
    python scripts/play_human.py --model runs/sac_final.zip

You control the LEFT (blue) mallet by moving the mouse; the bot is on the right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airhockey.config import TableConfig
from airhockey.physics import AirHockeyPhysics, GOAL_LEFT, GOAL_RIGHT, GOAL_NONE
from airhockey.render import Renderer, MARGIN, PPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="optional SB3 .zip to use as the opponent")
    args = ap.parse_args()

    cfg = TableConfig()
    physics = AirHockeyPhysics(cfg)
    renderer = Renderer(cfg, mode="human")
    pygame = renderer.pygame

    if args.model:
        from stable_baselines3 import SAC
        from airhockey.opponents import PolicyOpponent
        opponent = PolicyOpponent(SAC.load(args.model), cfg)
    else:
        from airhockey.opponents import ScriptedPredictor
        opponent = ScriptedPredictor(cfg)

    physics.reset(serve_to=int(np.random.randint(0, 2)))
    prev_mouse = np.array(physics.state.mallet_pos[0])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Mouse -> world position, confined to the agent's half.
        mx, my = pygame.mouse.get_pos()
        wx = (mx - MARGIN) / PPM
        wy = cfg.width - (my - MARGIN) / PPM
        xlo, xhi = physics.mallet_x_bounds(0)
        wx = float(np.clip(wx, xlo, xhi))
        wy = float(np.clip(wy, cfg.mallet_radius, cfg.width - cfg.mallet_radius))
        target = np.array([wx, wy])

        tv1 = np.asarray(opponent(physics), dtype=float)
        goal = GOAL_NONE
        for _ in range(cfg.substeps):
            # Track the mouse with a stiff P-controller (acts as the human "hand").
            err = target - physics.state.mallet_pos[0]
            tv0 = err * 18.0
            spd = np.linalg.norm(tv0)
            if spd > cfg.mallet_max_speed:
                tv0 = tv0 / spd * cfg.mallet_max_speed
            goal = physics.step(tv0, tv1)
            if goal != GOAL_NONE:
                break

        if goal == GOAL_RIGHT:
            renderer.score[0] += 1
            physics.reset(serve_to=0)
        elif goal == GOAL_LEFT:
            renderer.score[1] += 1
            physics.reset(serve_to=1)

        renderer.draw(physics, info="move mouse — ESC/close to quit")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

    renderer.close()


if __name__ == "__main__":
    main()
