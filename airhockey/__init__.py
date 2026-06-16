"""Air hockey bot — physics core + trajectory predictor + RL environment.

See ``air_hockey_bot_plan.md`` for the full design rationale (Thai).
"""

from .config import TableConfig
from .physics import AirHockeyPhysics, PhysicsState, GOAL_NONE, GOAL_LEFT, GOAL_RIGHT
from .predictor import TrajectoryPredictor
from .env import AirHockeyEnv

__all__ = [
    "TableConfig",
    "AirHockeyPhysics",
    "PhysicsState",
    "TrajectoryPredictor",
    "AirHockeyEnv",
    "GOAL_NONE",
    "GOAL_LEFT",
    "GOAL_RIGHT",
]
