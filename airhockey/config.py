"""Physical and simulation constants for the air-hockey table.

All units are SI: metres, seconds, metres/second. The table lies in the x-y
plane with the long (length) axis along x and the short (width) axis along y.

    y = W  +---------------------+
           |                     |
           | goal           goal |
    y = 0  +---------------------+
          x = 0                 x = L

The LEFT goal is at x = 0 (defended by mallet 0, "the agent").
The RIGHT goal is at x = L (defended by mallet 1, "the opponent").
The agent therefore always attacks toward +x.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableConfig:
    # --- Geometry (metres) ---
    length: float = 2.0           # x extent
    width: float = 1.0            # y extent
    goal_width: float = 0.30      # opening centred on y = width / 2
    puck_radius: float = 0.03
    mallet_radius: float = 0.05

    # --- Puck dynamics ---
    puck_friction: float = 0.20   # linear damping coefficient (1/s); air hockey is near-frictionless
    wall_restitution: float = 0.90
    mallet_restitution: float = 0.95
    puck_max_speed: float = 8.0

    # --- Mallet dynamics ("no teleport" — bounded speed and acceleration) ---
    mallet_max_speed: float = 4.0
    mallet_max_accel: float = 40.0

    # --- Frequencies ---
    physics_hz: float = 200.0     # fine substeps to avoid tunnelling
    decision_hz: float = 10.0     # agent picks an action 10x per second

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def decision_dt(self) -> float:
        return 1.0 / self.decision_hz

    @property
    def substeps(self) -> int:
        """Physics substeps executed per agent decision."""
        return max(1, round(self.physics_hz / self.decision_hz))

    @property
    def goal_y_min(self) -> float:
        return (self.width - self.goal_width) / 2.0

    @property
    def goal_y_max(self) -> float:
        return (self.width + self.goal_width) / 2.0

    # Bands the puck/mallet *centres* can occupy (radius-inset).
    @property
    def puck_y_band(self) -> tuple[float, float]:
        return (self.puck_radius, self.width - self.puck_radius)

    def half_x(self) -> float:
        return self.length / 2.0
