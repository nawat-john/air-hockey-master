"""Pygame renderer — used for human play, debugging, and GIF capture."""

from __future__ import annotations

import numpy as np

from .config import TableConfig
from .physics import AirHockeyPhysics
from .predictor import TrajectoryPredictor

PPM = 360  # pixels per metre
MARGIN = 30

BG = (15, 18, 28)
TABLE = (24, 60, 110)
LINE = (90, 130, 180)
GOAL = (235, 200, 60)
PUCK = (240, 240, 245)
MALLET0 = (70, 200, 255)   # agent (left)
MALLET1 = (255, 90, 90)    # opponent (right)
PRED = (120, 255, 160)


class Renderer:
    def __init__(self, cfg: TableConfig | None = None, mode: str | None = "human",
                 show_prediction: bool = True):
        import pygame
        self.pygame = pygame
        self.cfg = cfg or TableConfig()
        self.mode = mode or "human"
        self.show_prediction = show_prediction
        self.pred = TrajectoryPredictor(self.cfg)

        self.w = int(self.cfg.length * PPM) + 2 * MARGIN
        self.h = int(self.cfg.width * PPM) + 2 * MARGIN

        pygame.init()
        if self.mode == "human":
            pygame.display.set_caption("Air Hockey")
            self.screen = pygame.display.set_mode((self.w, self.h))
        else:
            self.screen = pygame.Surface((self.w, self.h))
        self.font = pygame.font.SysFont("consolas", 20)
        self.clock = pygame.time.Clock()
        self.score = [0, 0]  # [agent, opponent]

    def to_px(self, p) -> tuple[int, int]:
        return (int(MARGIN + p[0] * PPM), int(MARGIN + (self.cfg.width - p[1]) * PPM))

    def draw(self, physics: AirHockeyPhysics, info: str = ""):
        pg = self.pygame
        c = self.cfg
        s = physics.state
        scr = self.screen
        scr.fill(BG)

        # Table + centre line.
        rect = pg.Rect(MARGIN, MARGIN, int(c.length * PPM), int(c.width * PPM))
        pg.draw.rect(scr, TABLE, rect)
        pg.draw.rect(scr, LINE, rect, 2)
        mid_x = MARGIN + int(c.half_x() * PPM)
        pg.draw.line(scr, LINE, (mid_x, MARGIN), (mid_x, MARGIN + int(c.width * PPM)), 2)
        pg.draw.circle(scr, LINE, self.to_px([c.half_x(), c.width / 2]), int(0.18 * PPM), 2)

        # Goals.
        for gx in (0.0, c.length):
            top = self.to_px([gx, c.goal_y_max])
            bot = self.to_px([gx, c.goal_y_min])
            pg.draw.line(scr, GOAL, top, bot, 6)

        # Predicted puck trajectory.
        if self.show_prediction:
            traj = self.pred.trajectory(s.puck_pos, s.puck_vel, horizon=0.8, n=24)
            pts = [self.to_px(p) for p in traj]
            if len(pts) > 1:
                pg.draw.lines(scr, PRED, False, pts, 1)

        # Mallets + puck.
        pg.draw.circle(scr, MALLET0, self.to_px(s.mallet_pos[0]), int(c.mallet_radius * PPM))
        pg.draw.circle(scr, MALLET1, self.to_px(s.mallet_pos[1]), int(c.mallet_radius * PPM))
        pg.draw.circle(scr, PUCK, self.to_px(s.puck_pos), int(c.puck_radius * PPM))

        label = self.font.render(f"{self.score[0]} : {self.score[1]}   {info}", True, (220, 220, 230))
        scr.blit(label, (MARGIN, 4))

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.cfg.decision_hz)
        return self.frame()

    def frame(self) -> np.ndarray:
        arr = self.pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))  # (H, W, 3)

    def close(self):
        self.pygame.quit()
