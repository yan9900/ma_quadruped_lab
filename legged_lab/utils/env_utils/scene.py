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

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from legged_lab.sensors.ray_caster import RayCasterCfg
from legged_lab.sensors.camera import CAMERA_USD_CFG, CameraCfg


if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # 为什么用引号包裹类型名？
    # 在 Python 的类型注解中，如果你引用的类型在当前作用域还没定义（比如跨文件引用），
    # 可以用字符串包裹类型名，告诉类型检查器“稍后会有这个类型”。

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0, #half ce-like surface
                dynamic_friction=1.0, #half ce-like surface
            ),
            visual_material=None,  # 不绑定 MDL 材质，使 color_scheme 的 vertex color 生效
            debug_vis=False,
        )

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                # attach_yaw_only=True,
                ray_alignment='yaw',
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )

        if config.camera.enable_camera:
            # 如果启用物理asset，先创建USD模型
            # 这一步把d435的usd添加到了robot/prim_body_name/d435下，也就是捆绑了robot和d435的物理模型
            # 对于一个有实体的传感器来说，有两个path，即物理模型的prim_path和虚拟的sensor_path,二者路径不能相同
            if getattr(config.camera, 'use_physical_asset', False):
                # 使用真实的D435 USD文件
                self.d435_camera_asset = CAMERA_USD_CFG.replace(
                    prim_path="{ENV_REGEX_NS}/Robot/" + config.camera.prim_body_name + "/d435"
                )
                print(f"[DEBUG] Creating D435 USD asset at: {config.camera.prim_body_name}/d435")
                # sensor应该在USD资产内部，添加子路径避免冲突
                sensor_path = config.camera.prim_body_name + "/d435/front_cam"
            else:
                sensor_path = config.camera.prim_body_name + "/front_cam"
            
            # 创建相机sensor
            print(f"[DEBUG] Creating camera sensor with name: front_camera")
            print(f"[DEBUG] Camera sensor path: {'{ENV_REGEX_NS}/Robot/' + sensor_path}")
            # 实例化
            self.front_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + sensor_path,
                height=config.camera.height,
                width=config.camera.width,
                history_length=config.camera.history_length,
                update_period=config.camera.update_period,
                data_types=config.camera.data_types or ["distance_to_image_plane"],
                spawn=config.camera.spawn,
                offset=config.camera.offset,
                debug_vis=config.camera.debug_vis,
            )
