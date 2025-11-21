# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

"""Configuration for camera asset (physical USD model)."""

import os
from sklearn import base
from sympy import euler
import torch
from warp import quat
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

def quat_from_euler_zyx_rpy(roll, pitch, yaw):
    cy = torch.cos(yaw   * 0.5); sy = torch.sin(yaw   * 0.5)  # Z
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)  # Y
    cr = torch.cos(roll  * 0.5); sr = torch.sin(roll  * 0.5)  # X
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return torch.stack([qw, qx, qy, qz], dim=-1)

@configclass
class D435AssetCfg(AssetBaseCfg):
    """RealSense D435相机的物理asset配置
    
    这个配置定义了D435相机在USD场景中的物理实体，包括：
    - 外观模型（通过USD文件）
    - 位置和朝向
    - 物理属性
    
    与CameraCfg不同，这是物理asset配置，不是sensor配置。
    """
    
    # USD场景路径 - 相机将在这个路径下创建
    prim_path: str = "{ENV_REGEX_NS}/Robot/base/d435"
    
    # USD文件配置 - 修复单位和层次结构问题
    # Orientation设置不对，模型会消失
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "assets", "d435.usd"),
        scale=(1, 1, 1),  # 放大倍数
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,  
        ),
        
    )
    
    # 初始状态配置
    init_state: AssetBaseCfg.InitialStateCfg = AssetBaseCfg.InitialStateCfg(
        pos=(0.33, 0.0, 0.08),  # 相机在机器人base坐标系中的位置
        # @configclass自动生成的__init__只负责赋值，计算要交给__post_init__
        rot=None  
    )
    
    def __post_init__(self):
        
        euler_angles = torch.tensor([180.0, 70.0, -90.0])
        euler_rad = torch.deg2rad(euler_angles)
        base_quat = quat_from_euler_xyz(*tuple(euler_rad))
        # print(f"base_quat: {base_quat}")
        
        # 匹配Isaac Lab的坐标系
        flip_vector = torch.tensor([1.0, 1.0, 1.0, -1.0])
        final_quat = torch.tensor(base_quat) * flip_vector
        # print(f"final_quat: {final_quat}")

        self.init_state.rot = final_quat


# 预定义配置实例
CAMERA_USD_CFG = D435AssetCfg()

# 配置选择函数
def get_camera_asset_cfg(use_usd_model: bool = True) -> AssetBaseCfg:
    """获取相机asset配置
    
    Args:
        use_usd_model: 是否使用USD模型，False时使用简单几何体
    
    Returns:
        相机asset配置
    """
    if use_usd_model:
        return CAMERA_USD_CFG
