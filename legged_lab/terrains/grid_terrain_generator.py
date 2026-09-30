"""
GridTerrainGenerator: 支持按固定网格布局（grid_layout）生成地形的扩展。

用法示例（3×3，中心位置是 cliff，其余全是 flat）：

    from terrains.grid_terrain_generator import GridTerrainGeneratorCfg

    MY_CFG = GridTerrainGeneratorCfg(
        num_rows=3, num_cols=3,
        ...
        sub_terrains={
            "flat":  terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            "cliff": terrain_gen.MeshBoxTerrainCfg(proportion=1.0, ...),
        },
        # 9 个名字，行优先（row-major）顺序填写 sub_terrains 的 key
        # 索引顺序：
        #   [0] [1] [2]     row=0, col=0/1/2
        #   [3] [4] [5]     row=1, col=0/1/2
        #   [6] [7] [8]     row=2, col=0/1/2
        grid_layout=[
            "flat",  "flat",  "flat",
            "flat",  "cliff", "flat",
            "flat",  "flat",  "flat",
        ],
    )

grid_layout 的长度必须等于 num_rows × num_cols。
每个元素是 sub_terrains 字典里的 key（字符串）。
如果 grid_layout=None（默认），行为与原始 TerrainGenerator 完全一致（按 proportion 随机采样）。
"""

from __future__ import annotations

import numpy as np
from isaaclab.terrains import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class GridTerrainGeneratorCfg(TerrainGeneratorCfg):
    """扩展 TerrainGeneratorCfg，增加 grid_layout 字段。"""

    # 覆盖 class_type，指向我们的子类 generator
    class_type: type = None  # 在 __post_init__ 里设置，避免循环引用

    # 行优先的地形布局，每个元素是 sub_terrains 的 key
    # None 表示使用原始随机采样逻辑（proportion 权重）
    grid_layout: list[str] | None = None

    # 允许 spawn 的 sub_terrain key 列表（仅在 enable_random_terrain_spawn=True 时有效）
    # None = 所有格子都可以 spawn；["flat"] = 只在 key 为 "flat" 的格子 spawn
    spawn_tile_keys: list[str] | None = None

    def __post_init__(self):
        # 如果 class_type 还是 None（用户没有手动设置），指向 GridTerrainGenerator
        if self.class_type is None:
            self.class_type = GridTerrainGenerator

        # 验证 grid_layout 长度
        if self.grid_layout is not None:
            expected = self.num_rows * self.num_cols
            if len(self.grid_layout) != expected:
                raise ValueError(
                    f"grid_layout 长度 ({len(self.grid_layout)}) 与 "
                    f"num_rows×num_cols ({self.num_rows}×{self.num_cols}={expected}) 不匹配。"
                )
            # 验证所有 key 在 sub_terrains 里（只能在 sub_terrains 非空时检查）
            if self.sub_terrains:
                unknown = [k for k in self.grid_layout if k not in self.sub_terrains]
                if unknown:
                    raise ValueError(
                        f"grid_layout 中存在未知的 sub_terrain key：{unknown}。"
                        f"可用的 key：{list(self.sub_terrains.keys())}"
                    )


# 最近一次生成的实际布局（行优先的 sub_terrain key 列表）。
#
# 为什么需要它：上游 TerrainGenerator 生成完只保留 terrain_mesh / terrain_origins /
# flat_patches，"哪一格是哪种地形"这个信息被丢弃（见 _add_sub_terrain）；而
# TerrainImporter 只把 generator 当局部变量用完即弃。下游要按地形分层分析就没法反查。
# 生成时同时回写 cfg.grid_layout 和这个模块级变量，两条路取其一即可。
LAST_REALIZED_LAYOUT: list[str] | None = None


class GridTerrainGenerator(TerrainGenerator):
    """TerrainGenerator + 布局记录。

    - ``grid_layout`` 给定时按固定布局生成（每格类型确定，proportion 被忽略）。
    - ``grid_layout=None`` 时按 proportion 逐格随机采样，行为与上游一致，
      **但把实际采到的布局记录下来**，使 terrain_key 反查在随机模式下同样可用。
    两种模式下 difficulty 都逐格随机。
    """

    def _generate_random_terrains(self):
        global LAST_REALIZED_LAYOUT

        sub_terrains_cfgs = self.cfg.sub_terrains  # OrderedDict: name -> cfg
        names = list(sub_terrains_cfgs.keys())
        layout = getattr(self.cfg, "grid_layout", None)

        if layout is None:
            # 按 proportion 逐格随机采样（与上游 TerrainGenerator 同一套逻辑）
            proportions = np.array([c.proportion for c in sub_terrains_cfgs.values()], dtype=float)
            proportions /= proportions.sum()
            n = self.cfg.num_rows * self.cfg.num_cols
            layout = [names[self.np_rng.choice(len(proportions), p=proportions)] for _ in range(n)]
            realized_is_sampled = True
        else:
            realized_is_sampled = False

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))

            terrain_key = layout[index]
            sub_cfg = sub_terrains_cfgs[terrain_key]

            # difficulty 随机采样（与原始逻辑相同）
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

            mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)

        # 回写，供下游反查 terrain_key
        LAST_REALIZED_LAYOUT = list(layout)
        self.cfg.grid_layout = list(layout)
        if realized_is_sampled:
            from collections import Counter
            counts = Counter(layout)
            print(f"[Terrain] grid_layout=None，按 proportion 采样得到 "
                  f"{self.cfg.num_rows}x{self.cfg.num_cols} 实际配比: "
                  + ", ".join(f"{k}={counts.get(k, 0)}" for k in names))
