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


class GridTerrainGenerator(TerrainGenerator):
    """支持 grid_layout 固定布局的 TerrainGenerator。"""

    def _generate_random_terrains(self):
        """如果 cfg 提供了 grid_layout 则按固定布局生成，否则走原始随机逻辑。"""
        if not (hasattr(self.cfg, "grid_layout") and self.cfg.grid_layout is not None):
            # 没有 grid_layout → 原始随机逻辑
            super()._generate_random_terrains()
            return

        layout = self.cfg.grid_layout
        sub_terrains_cfgs = self.cfg.sub_terrains  # OrderedDict: name -> cfg

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))

            terrain_key = layout[index]
            sub_cfg = sub_terrains_cfgs[terrain_key]

            # difficulty 随机采样（与原始逻辑相同）
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

            mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)
