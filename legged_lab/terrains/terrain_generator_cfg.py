# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from math import pi
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from .grid_terrain_generator import GridTerrainGeneratorCfg
from .hf_increasing_slope import HfIncreasingSlopeTerrainCfg

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

FLAT_MESH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(10.0, 10.0),   # 足够大，150 step * 0.005s * 4 × 2.5m/s = 7.5m，保留余量
    border_width=10.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,  # 平地无需高分辨率，0.1 即可，与 plane 接触行为更接近
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="none",  # 平地所有顶点同高，height 无意义，使用白灰默认色
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

CLIFF_EVAL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(15.0, 15.0),
    border_width=10.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",  # 按高度着色
    use_cache=False,
    sub_terrains={
        # 纯危险地形：高台 0.35-0.55m，不含平地。
        # 用于 FNR/TPR 评估：不干预必然发生 collision，
        # FP = TN = 0 by construction，保证无偏 FNR。
        "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=1.0,
            box_height_range=(0.35, 0.55),
            platform_width=10.0,
            double_box=False,
        ),
    },
)

CLIFF_DETECTION_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(12.0, 12.0), #8 
    # 超出地形的缓冲地带
    border_width=20.0, #3
    num_rows=10, #20
    num_cols=10, #20
    horizontal_scale=0.1,  # 从 0.1 减小到 0.05，提高网格分辨率（5cm vs 10cm）
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        # 平地 - 提供 safe baseline
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.10,
        # ),
        
        # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.55,
            box_height_range=(0.40, 0.60),
            platform_width=8.0,
            double_box=False,
        ),
        
        # 深坑 - collision unsafe signal (撞墙)
        "deep_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.30, 0.60),
            platform_width=8.0,
            double_pit=False,
        ),
        
        # 楼梯 - collision unsafe signal (撞台阶)
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.18, 0.28),
            step_width=0.30,
            platform_width=8.0,
            border_width=1.0,
            holes=False,
        ),
        
        "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,     # 从 stairs 或 pit 各分 5% 出来
            slope_range=(0.15, 0.25),  # 足够触发 base/head collision
            platform_width=8.0,
        ),
        # small_steps: 移除 - 与 platform 视觉相似但 label=safe，污染 margin head
        # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        # "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.55,
        #     box_height_range=(0.05, 0.1),
        #     platform_width=4.0,
        #     double_box=False,
        # ),
        
    },
)

CLIFF_DETECTION_TERRAINS_LOW_SPEED_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0), #8 
    # 超出地形的缓冲地带
    border_width=20.0, #3
    num_rows=10, #20
    num_cols=10, #20
    horizontal_scale=0.1,  # 从 0.1 减小到 0.05，提高网格分辨率（5cm vs 10cm）
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        # # 平地 - 提供 safe baseline
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.10,
        # ),
        
        # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.55,
            box_height_range=(0.40, 0.60),
            platform_width=5.0,
            double_box=False,
        ),
        
        # 深坑 - collision unsafe signal (撞墙)
        "deep_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.30, 0.60),
            platform_width=5.0,
            double_pit=False,
        ),
        
        # # 楼梯 - collision unsafe signal (撞台阶)
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.18, 0.28),
            step_width=0.30,
            platform_width=5.0,
            border_width=1.0,
            holes=False,
        ),
        
        "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,     # 从 stairs 或 pit 各分 5% 出来
            slope_range=(0.15, 0.25),  # 足够触发 base/head collision
            platform_width=5.0,
        ),
        # small_steps: 移除 - 与 platform 视觉相似但 label=safe，污染 margin head
        # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        # "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.55,
        #     box_height_range=(0.05, 0.1),
        #     platform_width=4.0,
        #     double_box=False,
        # ),
        
    },
)

# 3×3 地形：中心一格是平地（MeshPlane），周围8格是随机不平地（HfRandomUniform）
# 布局（行优先）：
#   1 1 1
#   1 0 1   ← 0 = flat, 1 = uneven
#   1 1 1
RING_UNEVEN_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=3,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    spawn_tile_keys=["flat"],  # 只在中心平地格子 spawn
    grid_layout=[
        "uneven", "uneven", "uneven",
        "uneven", "flat",   "uneven",
        "uneven", "uneven", "uneven",
    ],
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.0,  # proportion 在 grid_layout 模式下无效，仅用于注册 key
        ),
        "uneven": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.0,
            noise_range=(0.0, 0.05),  # 加大起伏：15cm 对 Go2 腿长 30cm 是显著挑战
            noise_step=0.02,
            border_width=0.0,  # tile 边缘 0.5m 过渡带（高度平滑归零），消除与 flat 的突变台阶
        ),
    },
)

CLIFF_EVALUATION_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(20.0, 20.0), #8 
    # 超出地形的缓冲地带
    border_width=20.0, #3
    num_rows=4, #20
    num_cols=4, #20
    horizontal_scale=0.1,  # 从 0.1 减小到 0.05，提高网格分辨率（5cm vs 10cm）
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={

        
        # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.55,
            box_height_range=(0.40, 0.60),
            platform_width=10.0,
            double_box=False,
        ),
        
        # "deep_pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.1,
        #     pit_depth_range=(0.30, 0.60),
        #     platform_width=10.0,
        #     double_pit=False,
        # ),
        # "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.15,
        #     step_height_range=(0.18, 0.28),
        #     step_width=0.30,
        #     platform_width=10.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.10,     # 从 stairs 或 pit 各分 5% 出来
        #     slope_range=(0.2, 0.25),  # 足够触发 base/head collision
        #     platform_width=2.0,
        # ),
        
        
    },
)

# ---------------------------------------------------------------------------
# Gradually-increasing-slope terrain
# ---------------------------------------------------------------------------
# The tile is W×L metres.  The LEFT half (x < 0) is flat; the RIGHT half
# rises parabolically: z(x) = slope_max * x² / W.
#
# Instantaneous slope at position x:  dz/dx = 2*slope_max*x/W
#   → continuously increases from 0 at the center to slope_max at the edge.
#
# Experiment use-case
# -------------------
#  * Robot spawns at center (x=0, flat) and walks in the +x direction.
#  * gz decreases continuously → clean proprioceptive failure signal.
#  * Depth camera sees a nearly-uniform inclined surface ahead → low visual
#    alarm until the slope becomes extreme.
#  * Once slope > balance limit (~tan 35-40°) the robot falls.
#
# num_rows controls difficulty spread (curriculum=True gives rows 0..N-1
# mapped to difficulty 0..1, so the steepest row has slope_range[1]).

INCREASING_SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(12.0, 8.0),       # 12 m in x (6 m uphill travel) × 8 m in y
    border_width=10.0,
    num_rows=1,             # difficulty rows: slope_max from 0.3 → 0.9
    num_cols=1,            # 50 tiles total — manageable mesh size
    horizontal_scale=0.1,   # 10 cm resolution — sufficient for smooth parabola
    vertical_scale=0.005,
    slope_threshold=None,   # keep the smooth ramp, no vertical-face correction
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "increasing_slope": HfIncreasingSlopeTerrainCfg(
            proportion=1.0,
            slope_range=(0.3, 0.9),  # tan(~17°) at easiest → tan(~42°) hardest
            horizontal_scale=0.1,
            vertical_scale=0.005,
        ),
    },
)
