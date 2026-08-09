# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

"""Custom height field terrain: gradually increasing slope along x-axis.

Shape (cross-section along x):
    - Left half (x < 0): flat at z = 0. Robot spawns here (terrain origin = center).
    - Right half (x >= 0): parabolic rise  z(x) = slope_max * x² / W
      where W = terrain width.

At each x position the instantaneous slope is:
    dz/dx(x) = 2 * slope_max * x / W

So the slope grows linearly from 0 (at center) to `slope_max` (at the right edge).

Why this is useful for safety-monitor experiments
-------------------------------------------------
* gz (body z-axis projected onto world gravity) continuously decreases as the
  robot tilts forward → clear proprioceptive anomaly signal that develops slowly.
* The depth camera looking forward sees roughly the same inclined ground ahead
  as the slope increases gradually → depth-based monitors see little change until
  it is too late.
* When slope_max > tan(robot_stability_limit) the robot eventually falls,
  providing a clean "failure without visual warning" scenario.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg


# ---------------------------------------------------------------------------
# Terrain-generation function  (defined first so the config can reference it)
# ---------------------------------------------------------------------------

@height_field_to_mesh
def increasing_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """Generate a terrain with gradually increasing slope along the x-axis.

    Left half of the tile (x < 0) is flat at z=0.
    Right half (x >= 0) rises parabolically: z(x) = slope_max * x² / W
    so the instantaneous slope dz/dx = 2*slope_max*x/W increases linearly
    from 0 at the centre to slope_max at the right edge.

    Args:
        difficulty: Scalar in [0, 1].  Interpolates ``slope_range``.
        cfg:        :class:`HfIncreasingSlopeTerrainCfg` instance.

    Returns:
        2-D int16 height field of shape ``(width_pixels, length_pixels)``.
    """
    # resolve slope for this difficulty level
    slope_max = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # pixel dimensions
    W = cfg.size[0]   # physical width  (x direction)  [m]
    width_pixels  = int(W / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # physical x coordinates centred at 0
    x_phys = np.linspace(-W / 2.0, W / 2.0, width_pixels)

    # parabolic height for x >= 0, flat for x < 0
    z_phys = np.where(x_phys >= 0, slope_max * x_phys ** 2 / W, 0.0)  # metres

    # convert to discrete height units
    z_discrete = np.rint(z_phys / cfg.vertical_scale).astype(np.int16)

    # broadcast across y (terrain is uniform in y direction)
    hf = np.tile(z_discrete[:, np.newaxis], (1, length_pixels))

    return hf


# ---------------------------------------------------------------------------
# Configuration dataclass  (defined after the function it references)
# ---------------------------------------------------------------------------

@configclass
class HfIncreasingSlopeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a parabolic, gradually-increasing-slope terrain.

    The terrain occupies a rectangular tile of size ``(W, L)`` (x × y).  The
    left half of the tile (x < 0) is flat; the right half rises parabolically
    so that the slope at the far-right edge equals ``slope_range[1]`` at
    maximum difficulty.

    Recommended spawn:  place the robot at the terrain origin (center of the
    tile, x = 0) facing the +x direction.

    Example::

        HfIncreasingSlopeTerrainCfg(
            proportion=1.0,
            size=(12.0, 12.0),
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_range=(0.3, 0.9),   # tan(~17°) → tan(~42°)
        )
    """

    function = increasing_slope_terrain

    slope_range: tuple[float, float] = MISSING
    """(min_slope, max_slope) at the right edge of the terrain (rise/run ratio).

    At difficulty=0 the edge slope equals ``slope_range[0]``;
    at difficulty=1 it equals ``slope_range[1]``.

    Typical values that cause a quadruped to fall:
      * Go2 / ANYmal: slope ≳ 0.7–0.9  (≈ 35–42°)
    """
