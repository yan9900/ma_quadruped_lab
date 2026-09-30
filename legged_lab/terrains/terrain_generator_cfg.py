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

# Friction-adaptive 采集与 cliff detection 共用的主地形。
#
# 用 GridTerrainGeneratorCfg 但 **grid_layout=None**：类型仍由下面的 proportion
# 逐格随机采样，和原始 TerrainGenerator 行为一致。用子类只为一件事——
# 原始 TerrainGenerator 生成完就把"哪一格是哪种地形"丢掉了（只留 terrain_origins /
# terrain_meshes / flat_patches，见 _add_sub_terrain），而 hazard_dir 分层分析需要
# 逐 episode 的 terrain_key。子类把实际采到的布局回写到 cfg.grid_layout，
# 采集器照常反查，不需要手工排布。
#
# 不变量：所有子地形的 platform_width 都是 8.0，所以采集器的
# --platform_half_width 4.0 对全部类型都正确，边缘触发只需要一个数。
#
# "flat" 故意不启用：MeshPlaneTerrainCfg 没有平台边缘，d_edge_ray 会对着一个
# 不存在的矩形边界计算，机器人越过后既不会坠落也不会碰撞——制动 episode 全是废片。
# 需要安全基线的话用 FLAT_MESH_TERRAINS_CFG 单独采。
CLIFF_DETECTION_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(12.0, 12.0), #12
    # 超出地形的缓冲地带
    border_width=20.0,
    # 10x10 = 100 格。采集时 20-40 个 env 并行，randomize_terrain_spawn 需要
    # 可 spawn 格数远多于 num_envs 才能保证每台机器人独占一格；同格的两台 Go2
    # 会进入彼此的 3m 深度相机。
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        # # 平地 - 提供 safe baseline
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.10,
        # ),
        
        # # 危险平台 (0.40-0.60m) - 主力 cliff fall unsafe signal
        # "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.55,
        #     box_height_range=(0.40, 0.60),
        #     platform_width=12.0,
        #     double_box=False,
        # ),
        # # 深坑 - collision unsafe signal (撞墙)
        # "deep_pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.15,
        #     pit_depth_range=(0.30, 0.60),
        #     platform_width=12.0,
        #     double_pit=False,
        # ),
        # 楼梯 - collision unsafe signal (撞台阶)
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.18, 0.40), #(0.18,0.28)
            step_width=0.30,
            platform_width=8.0,
            border_width=1.0,
            holes=False,
        ),
        # 倒金字塔斜坡 - 平台在最低处，越界=上坡，hazard_dir=+1
        "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.15, 0.35), #(0.15, 0.25)
            platform_width=8.0,
        ),
        # "flat": 见上方说明，制动采集里不能用
        # "small_steps": 移除 - 与 platform 视觉相似但 label=safe，污染 margin head
    },
)

# 按 RobotLab v1 训练配置复刻的上楼梯 / 上坡地形，用来测 RobotLab policy 在其训练分布内的能力。
# 来源：go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/mdp/terrains.py 的 stairs_up / slope_up。
# 与训练的差别：
#   - 训练格子是 9 m（8 m 地形 + 两侧各 0.5 m 子地形边框）；这里用 8 m、无子地形边框，相邻格子边缘高度都是 0。
#   - 训练里楼梯 slope_threshold=0.25、斜坡 10.0（按子地形分别设置）；IsaacLab 生成器会用全局值覆盖，
#     这里统一 0.75。只影响 <7.5 cm 的台阶（立面变成 10 cm 宽的小斜面），斜坡 ≤0.568 不受影响。
#   - 没有 terrain level 课程，difficulty 在 [0,1] 上均匀采样。
# 几何：平台在中心最低处，向外逐级/连续上升；采集时 --platform_half_width = platform_width/2
#（训练原值 3 m → 1.5；当前 6 m → 3.0）。
ROBOTLAB_TRAIN_STAIRS_SLOPE_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(11.0, 11.0),  # 训练 8 m；2026-09-29 加大平台做助跑，台阶级数仍为 8
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.08, 0.20),  # 训练范围 (0.05, 0.257)；2026-09-29 扫描碰头分界（此前固定 0.2 测负载）
            step_width=0.31,
            platform_width=6.0,  # 训练 3.0
        ),
        # 2026-09-29 暂时只保留楼梯（测实际能爬的级数）
        # "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.5,
        #     slope_range=(0.1, 0.568),
        #     platform_width=6.0,  # 训练 3.0
        # ),
    },
)

# docs/82 §7.2 的 pilot 子集：先做「下行边 / 单级上行边 / 阴性对照」，不含楼梯（头部接触定义未决，docs/81）。
# 几何（IsaacLab trimesh/mesh_terrains.py 核对）：
#   drop_box : 边长 platform_width 的方箱，顶面 = box_height，出生在顶面；下行边在离中心 platform_width/2 处。
#   step_pit : 中间 platform_width 见方的坑，坑底 = −depth，出生在坑底；上行墙在离中心 platform_width/2 处。
#   slope    : 高度 = h_max·xx·yy、平台按角点截平 ⇒ 平台宽取格宽 1/3 才是真坡（2/3 时几乎是平地，docs/81 §5）。
# 格子 8 m、平台 3 m ⇒ 边缘离出生点 1.5 m；采集时 --platform_half_width 1.5（d_edge 对照臂）。
GX_PILOT_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "drop_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.35, box_height_range=(0.05, 0.60), platform_width=3.0, double_box=False,
        ),
        "step_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.30, pit_depth_range=(0.05, 0.45), platform_width=3.0, double_pit=False,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.1, 0.4), platform_width=2.7,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.1, 0.4), platform_width=2.7,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)

# 目视检查：全部是 0.5 m 方箱（与 GX_PILOT 同尺寸），确认 v1 能否从 0.5 m 平台走下去。
GX_DROP05_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "drop_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=1.0, box_height_range=(0.5, 0.5), platform_width=3.0, double_box=False,
        ),
    },
)

def _exact_layout(counts: dict, seed: int = 0) -> list:
    """按精确格数生成打乱的 grid_layout（逐格按 proportion 抽样在 64 格下偏差很大：
    gx_main A 实际高平台 6/64 = 9%，目标 22%）。"""
    import random as _random
    layout = [k for k, n in counts.items() for _ in range(n)]
    _random.Random(seed).shuffle(layout)
    return layout


# 静态高程图 g_x 正式采集（docs/82、docs/83）：A 跨越 / D 低摩擦行走用小平台版。
# 阈值 h_down = 0.236 m、h_up = 0.031 m（docs/83 §3.4）⇒ 低平台 0.05–0.40 加密在 h_down 两侧，
# 高平台 0.40–0.90 补足"过不去的悬崖"（pilot 最高 0.58 m，几乎没有）；坑壁在 h_up = 3 cm 下全部算墙。
# 几何同 GX_PILOT：8 m 格、3 m 平台，边在离中心 1.5 m；坡平台 = 格宽 1/3。
GX_MAIN_SMALL_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    grid_layout=_exact_layout({"box_low": 20, "box_high": 14, "step_pit": 14, "slope_up": 7, "slope_down": 7, "flat": 2}),
    sub_terrains={
        "box_low": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.32, box_height_range=(0.05, 0.40), platform_width=3.0, double_box=False,
        ),
        "box_high": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.22, box_height_range=(0.40, 0.90), platform_width=3.0, double_box=False,
        ),
        "step_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.22, pit_depth_range=(0.05, 0.45), platform_width=3.0, double_pit=False,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.11, slope_range=(0.1, 0.4), platform_width=2.7,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.11, slope_range=(0.1, 0.4), platform_width=2.7,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.02),
    },
)

# 静态高程图 g_x 正式采集：B 高 μ 刹车 / C 低摩擦刹车用大平台版（spawn solve 需要长跑道）。
# 尺寸由 v1 平地制动拟合（latent-safety/results/d_brake_fit_v1.json）反推：最坏 v=2.0、μ_dyn=0.07、r=1.6 时
# 出生到边 D0 = v²/(2·1.1) + v·(0.25+0.6) + 1.6·(d_brake+0.11) ≈ 9.3 m，朝向 ±45° ⇒ 平台半宽 5 m。
# 只含方箱与坑：spawn solve 把出生点放在离中心数米处，而斜坡格只有中心是平的、采集器按格中心高度出生，
# 试运行中上坡格出生在地面以下（首帧头部接触）。坡/平地的阴性样本由 A、D（小平台，中心出生）提供。
# 低平台收窄到 0.15–0.40 m 并降到 25%：≤ 0.24 m 的边可跨越，刹不刹得住都不失败，对 V_stop 信息量低，这类阴性样本 A 里已大量存在。
GX_MAIN_BIG_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(14.0, 14.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    grid_layout=_exact_layout({"box_low": 16, "box_high": 26, "step_pit": 22}),   # 25% / 40% / 35%（2026-09-30 用户确认）
    sub_terrains={
        "box_low": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.25, box_height_range=(0.15, 0.40), platform_width=10.0, double_box=False,
        ),
        "box_high": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.40, box_height_range=(0.40, 0.90), platform_width=10.0, double_box=False,
        ),
        "step_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.35, pit_depth_range=(0.05, 0.45), platform_width=10.0, double_pit=False,
        ),
    },
)

# E4 方案复刻（go2_fricsweep_2700 采集时的 CLIFF_DETECTION，git 6a0695b）：12 m 格、10×10、平台 8 m（边在 ±4.0 m），
# dangerous_platform 0.40–0.60 / deep_pit 0.30–0.60 / slope 0.15–0.25 各按 E4 原值；
# E4 的 stairs（15%）换成 box_low（0.15–0.40，跨越 h_down=0.236 两侧）。2026-09-30 用户定。
# 用精确计数布局（_exact_layout），避免随机抽样把某类格子抽得过少。
GX_E4_TERRAINS_CFG = GridTerrainGeneratorCfg(
    curriculum=False,
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    grid_layout=_exact_layout({"dangerous_platform": 55, "deep_pit": 15, "box_low": 15, "slope": 15}),
    sub_terrains={
        "dangerous_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.55, box_height_range=(0.40, 0.60), platform_width=8.0, double_box=False,
        ),
        "deep_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.30, 0.60), platform_width=8.0, double_pit=False,
        ),
        "box_low": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.15, box_height_range=(0.15, 0.40), platform_width=8.0, double_box=False,
        ),
        "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.15, 0.25), platform_width=8.0,
        ),
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
            noise_step=0.05,
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
