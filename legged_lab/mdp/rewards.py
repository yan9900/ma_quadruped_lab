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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional

# # Initialize Isaac Sim environment before importing isaaclab modules
# try:
#     import isaacsim  # This must be imported first to initialize the Isaac Sim environment
# except ImportError:
#     print("Warning: Isaac Sim not available. Some imports may fail.")

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv

# velocity tracking functions
# yaw_frame下的x/y速度跟踪
# yaw_frame 是指只考虑机器人朝向（yaw角）旋转后的坐标系
# root link frame 是机器人根节点的完整坐标系，包含 roll、pitch、yaw 三个旋转分量 
def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # 绑定机器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算 yaw_frame 下的线速度，quat->quaternion
    # inputs: root_quat_w, root_lin_vel_w
    # outputs: vel_yaw: [num_envs, 3] lin xy, ang z
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    # 做差
    # lin_vel_error shape: [num_envs,]
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    # 归一化
    # shape: [num_envs,]
    return torch.exp(-lin_vel_error / std**2)

# z方向world frame 角速度跟踪
def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    # 归一化
    # shape: [num_envs,]
    return torch.exp(-ang_vel_error / std**2)

# root / pose functions punishment-> bigger, worse
# 惩罚commands以外的三个量，ang xy和lin z
# root link frame下的z方向线速度
def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # l2 norm
    # shape: [num_envs,]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

# root link frame下的x/y方向角速度
def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # l2 norm
    # shape: [num_envs,]
    # x*x + y*y
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

# joints / actuation functions
# 能量-penalty term
def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节能量 tau * omega/ F * v ->瞬时功率
    # inputs: applied_torque, joint_vel
    # outputs: reward: [num_envs,]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward

# 关节加速度 - penalty term
def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # joint_acc shape: [num_envs, num_joints], 加速度的值为标量
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

# 关节控制频率 - penalty term
# buffer: [num_envs, history_length, num_actions]
# square() [num_envs, num_actions]
# sum(, dim = 1) [num_envs,]
# 最新的动作和上一个动作的差值的l2 norm
def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

# contact/force functions
# net_contact_forces [num_sensors, history_length, num_bodies, 3]
# norm(, dim=-1) [num_sensors, history_length, selected_bodies]
# max(, dim=1) [num_sensors, selected_bodies] 选出历史最大值
# max()会返回两个张量，[0]是最大值，[1]是索引
# is_contact [num_sensors, selected_bodies] bool tensor
# sum(, dim=1) [num_sensors,]
def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # print(f"net_contact_forces: {net_contact_forces.shape}")
    # print(f"is_contact: {is_contact.shape}")
    return torch.sum(is_contact, dim=1)

# is_contact本身就是bool tensor
# < 0.5说明一个接触的body都没有 -> fly
def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

# 衡量机器人身体姿态是否接近（水平），也就是身体的 roll 和 pitch 是否接近 0
# 重力向量在机器人根节点坐标系下的分量，shape [num_envs, 3]
def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

# 只有当某个环境需要重置（reset_buf=True），且不是因为超时（time_out_buf=False），结果才是 True
# 返回的也就是异常终止的项
def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf

# biped-specific functions
def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

# biped-specific functions
def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

# modifications for quadruped robots
# -------------------------
# Joints / Actuation
# -------------------------

# revolute joints
def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau), dim=1)
# prismatic joints
def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(qd), dim=1)

# joint position limits
# l1 norm, closer to limits, worse
def joint_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q  = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lo = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 0]
    hi = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 1]
    # clamp(min=0.0)相当于去除小于0的部分
    below = (lo - q).clamp(min=0.0)
    above = (q - hi).clamp(min=0.0)
    return torch.sum(below + above, dim=1)

# joint velocity limits
# l1 norm, closer to limits, worse
def joint_vel_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if not hasattr(asset.data, "joint_vel_limits"):
        return torch.zeros(qd.shape[0], device=asset.data.joint_vel.device)
    vmax = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    excess = (qd.abs() - vmax).clamp(min=0.0)
    return torch.sum(excess, dim=1)

# 防止顺拐
# 为什么没用asset_cfg.joint_ids?
def joint_mirror(env, mirror_joints: List[Tuple[List[int], List[int]]]) -> torch.Tensor:
    """
    mirror_joints: [([ids_left], [ids_right]), ...]
    度量 |q_left + q_right|（左右对称时应相反）。
    支持传入关节名称字符串，自动转换为ID。
    """
    asset: Articulation = env.scene["robot"]
    q = asset.data.joint_pos
    loss = 0.0
    for ids_a, ids_b in mirror_joints:
        # 如果传入的是字符串名称，转换为ID
        if isinstance(ids_a[0], str):
            ids_a = [asset.find_joints(name)[0][0] for name in ids_a]
        if isinstance(ids_b[0], str):
            ids_b = [asset.find_joints(name)[0][0] for name in ids_b]
        
        # 转换为tensor
        ids_a = torch.tensor(ids_a, device=q.device, dtype=torch.long)
        ids_b = torch.tensor(ids_b, device=q.device, dtype=torch.long)
        
        loss = loss + torch.sum(torch.abs(q[:, ids_a] + q[:, ids_b]), dim=1)
    return loss


# -------------------------
# Root / Pose
# -------------------------

def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) > 0:
        z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    else:
        z = asset.data.root_pos_w[:, 2]
    return torch.square(z - target_height)

def body_lin_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    a = asset.data.root_lin_acc_w[:, :3]
    return torch.sum(torch.square(a), dim=1)

# 重力在root link frame下的z分量 bigger, better
# 如果翻倒，z分量会变成负值 -> 惩罚
def upward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    return (-g_b[:, 2]) / torch.norm(g_b, dim=1)

# 奖励站稳
def stand_still_without_cmd(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_generator.command  # [num_envs, 3] xyz
    # still 是 bool tensor
    # cmd xy线速度+z角速度 < 0.1 判定为静止
    still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    v_lin = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    v_yaw = torch.abs(asset.data.root_ang_vel_w[:, 2])
    # cmd < 0.1, still = 1, 速度/角速度越小，奖励越大
    return -(v_lin + v_yaw) * still


# -------------------------
# Contacts / Forces
# -------------------------

def contact_forces(env, sensor_cfg: SceneEntityCfg, reduction: str = "sum") -> torch.Tensor:
    """
    汇总所选 body 的接触力范数。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N,B,3] 没有历史？
    fn = torch.norm(f, dim=-1)                                      # [N,B]
    if reduction == "sum":
        return torch.sum(fn, dim=1)
    elif reduction == "mean":
        return torch.mean(fn, dim=1)
    else:
        return torch.max(fn, dim=1)[0]


# -------------------------
# Feet / Gait (quadruped general)
# -------------------------

# 是否接触，重复了
def _contacts_bool(env, sensor_cfg: SceneEntityCfg, thresh: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # [N,T,B,3]
    fmax = forces_hist.norm(dim=-1).max(dim=1)[0]  # [N,B]
    return fmax > thresh

# 悬空时间奖励
def feet_air_time(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [N,B]
    reward = torch.sum(torch.clamp(air_time, max=threshold), dim=1)
    cmd = env.command_generator.command
    moving = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1
    return reward * moving

def feet_air_time_variance(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = torch.clamp(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], max=threshold)
    return torch.var(air_time, dim=1, unbiased=False)

def feet_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    c = _contacts_bool(env, sensor_cfg, thresh=threshold)
    return torch.sum(c, dim=1).float()

def feet_contact_without_cmd(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    c = _contacts_bool(env, sensor_cfg, thresh=threshold)
    num_c = torch.sum(c, dim=1).float()
    cmd = env.command_generator.command
    still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    return num_c * still

def feet_height(env, target_height: float, sensor_cfg: SceneEntityCfg,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    c = _contacts_bool(env, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)                         # [N,1]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                   # [N,B]
    z_rel  = foot_z - base_z                                                  # [N,B]
    swing  = (~c)
    gain   = torch.clamp(z_rel - target_height, min=0.0) * swing
    return torch.sum(gain, dim=1)

def feet_height_body(env, target_height: float, sensor_cfg: SceneEntityCfg,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    z_rel  = foot_z - base_z
    excess = (z_rel - target_height).clamp(min=0.0)
    return torch.sum(excess, dim=1)

def feet_gait(env, sensor_cfg: SceneEntityCfg,
              synced_feet_pair_ids: Optional[Tuple[Tuple[int,int], ...]] = None,
              threshold: float = 1.0) -> torch.Tensor:
    """
    以接触状态一致性作为分数（1=完全同步）。
    注意：synced_feet_pair_ids 索引应与 sensor_cfg.body_ids 的顺序一致。
    """
    c = _contacts_bool(env, sensor_cfg, thresh=threshold).float()  # [N,B]
    if not synced_feet_pair_ids:
        return torch.ones(c.shape[0], device=c.device)
    scores = []
    for a, b in synced_feet_pair_ids:
        scores.append(1.0 - torch.abs(c[:, a] - c[:, b]))
    return torch.stack(scores, dim=1).mean(dim=1)

# quadruped recovery rewards
# Orientation Posture
# Base Orientation ->已经存在，查看flat_orientation_l2

# Upright Orientation, 
def upright_orientation_root(
    env: BaseEnv, epsilon: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # 绑定机器人
    asset: Articulation = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2] #shape [num_instance,]
    return torch.exp(-torch.square(g_z+1) / (2*epsilon**2))

# Target Posture
def target_posture(
    env: BaseEnv, epsilon: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)->torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_stand = asset.data.default_joint_pos[:,asset_cfg.joint_ids]
    g_z = asset.data.projected_gravity_b[:, 2]
    joint_diff_squared = torch.sum(torch.square(q - q_stand), dim=1)  # [num_envs]
    # print(f"q: {q}")
    # print(f"q_stand: {q_stand}")
    # print(f"g_z: {g_z}")
    # 公式：exp(-(q - q_stand)²) if |g_z + 1| < ε, else 0
    # condition = torch.abs(g_z + 1) < epsilon
    # reward = torch.where(
    #     condition,
    #     torch.exp(-joint_diff_squared),
    #     torch.zeros_like(joint_diff_squared)
    # )
    mask = (torch.abs(g_z + 1) < epsilon).float()
    reward = mask * torch.exp(-joint_diff_squared)
    return reward

# Contact management
# Feet Contact
# 已经存在，查看undesired_contacts

# Body Contact
# 惩罚碰撞，不包含腿部，已经存在，但是需要传不同的body_ids(去除feet_ids)

# Stability Control
# Safety force,惩罚水平方向上的knee contacts,l2 norm
# 这里的body_ids要找knee
def safety_force(env: BaseEnv, 
                 sensor_cfg: SceneEntityCfg, 
                 reduction: str = "last",
                 window: int | None = None,
                 )->torch.Tensor:
    """
    reduction:
      - "last": 只用当前帧（最快，最常用）
      - "max": 近窗口最大值（对抖动更鲁棒）
      - "mean": 近窗口均值（平滑但会“稀释”峰值）
    window:
      - None: 用整个历史（通常不建议太长）
      - int: 例如 3~10 帧
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    knee_f_xy = net_contact_forces[:, :, sensor_cfg.body_ids, :2] # [N,T,B,2]
    knee_f_xy_sq = torch.sum(torch.square(knee_f_xy), dim=-1)  # [N,T,B]
    # 裁减到指定窗口
    if window is not None:
        knee_f_xy_sq = knee_f_xy_sq[:, -window:, :]
    # 按指定方式汇总
    if reduction == "last":
        # 当前帧 [N, B]
        knee_f_t = knee_f_xy_sq[:, -1, :]
    elif reduction == "max":
        # 窗口内最大 [N, B]
        knee_f_t = torch.max(knee_f_xy_sq, dim=1)[0]
    elif reduction == "mean":
        # 窗口内均值 [N, B]
        knee_f_t = torch.mean(knee_f_xy_sq, dim=1)
    else:
        raise ValueError("time_reduce must be 'last' | 'max' | 'mean'.")
    
    return torch.sum(knee_f_t, dim=1)

# Body Bias防止recover过程中不要出现太大的位置移动
# 指标：root link position在xy平面的l2 norm
def body_bias(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))->torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # curr pos
    p_xy_curr = asset.data.root_link_pos_w[:,:2] # [N,2]
    # init pos（scene的中心）
    p_xy_init = env.scene.env_origins[:, :2]       # [N,2]
    # l2距离并clip
    bias = torch.norm(p_xy_curr - p_xy_init, dim=1) # [N,]
    return torch.clamp(bias, 0, 4)

# Motion Constraints
# Position Limit，所有腿部关节，12
def position_limits(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q  = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lo = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 0]
    hi = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 1]
    # 超过上限或低于下限的关节数量
    above_max = (q > hi) # [N, 12]
    below_min = (q < lo) # [N, 12]
    # 按公式统计bool数量
    return (above_max | below_min).sum(dim=1).float() # [N,]

# Angular Velocity Limit
# 惩罚超过0.8rad/s的关节速度，l1 norm
def angular_velocity_limits(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids] # [N, 12]
    # 公式：max(|qd| - 0.8, 0)
    excess = (qd.abs() - 0.8).clamp(min=0.0) # [N, 12]
    # 每个环境取所有关节的最大值
    return excess.max(dim=1)[0]

# Joint Acc, 已经存在，参考joint_acc_l2

# Joint Vel
def joint_vel_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # joint_vel shape: [num_envs, num_joints], 速度的值为标量
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

# Action Smoothing, 已经存在，参考action_rate_l2
# Joint Torques
def joint_torques(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]  # [N, J]
    return torch.sum(torch.square(tau), dim=1)  # [N]