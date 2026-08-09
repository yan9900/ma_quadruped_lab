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
from unittest.mock import Base

# # Initialize Isaac Sim environment before importing isaaclab modules
# try:
#     import isaacsim  # This must be imported first to initialize the Isaac Sim environment
# except ImportError:
#     print("Warning: Isaac Sim not available. Some imports may fail.")

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
# from utils.env_utils import scene

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv

# velocity tracking functions
# Note: commands are given in base frame!!
# yaw_frame下的x/y速度跟踪
# yaw_frame 是指只考虑机器人朝向（yaw角）旋转后的坐标系
# root link frame 是机器人根节点的完整坐标系，包含 roll、pitch、yaw 三个旋转分量 
def track_lin_vel_xy_base_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # 绑定机器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算 base_frame 下的线速度，quat->quaternion
    # inputs: root_quat_w, root_lin_vel_w
    # outputs: vel_yaw: [num_envs, 3] lin xy, ang z
    # vel_yaw = math_utils.quat_apply_inverse(
    #     math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    # )
    # lin_vel_error shape: [num_envs,]
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # shape: [num_envs,]
    return torch.exp(-lin_vel_error / std**2)

# z方向base frame 角速度跟踪
def track_ang_vel_z_base_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_b[:, 2])
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
    """Penalize feet sliding relative to the world (ground).

    Measures the world-frame lateral velocity of each foot while in contact.
    This correctly penalizes ground-level slipping regardless of body motion.
    A foot planted on the ground should have near-zero world-frame velocity.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_slide_body_frame(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize lateral (sideways) foot sliding relative to the robot body.

    Computes foot velocity relative to the body, rotated into body frame, and penalizes
    the Y-axis (lateral) component while a foot is in contact. X-axis backward motion
    during stance is normal walking behavior and is intentionally excluded.
    Complements `feet_slide` (world-frame): together they catch both ground-slip and
    body-frame lateral drift.
    """
    # Penalize feet sliding
    contact_sensor : ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # 计算脚在body frame下的速度
    # cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    # footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    # for i in range(len(asset_cfg.body_ids)):
    #     footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
    #         asset.data.root_quat_w, cur_footvel_translated[:, i, :]
    #     )
    # foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
    #     env.num_envs, -1
    # )
    # reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    
    # 脚相对于机身的速度（世界坐标系下相减，再转到body frame）
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    # 只取Y轴（侧向）分量：正常站立/行走时X有退后速度属正常，Y侧滑才是打滑信号
    footvel_y = footvel_in_body_frame[:, :, 1]
    reward = torch.sum(footvel_y.abs() * contacts, dim=1)
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
def joint_mirror(env, mirror_joints: List[Tuple[List[str], List[str]]]) -> torch.Tensor:
    """
    mirror_joints: [([joint_names_left], [joint_names_right]), ...]
    度量左右对称关节的动作差异，用于防止顺拐。
    支持传入关节名称字符串，自动转换为ID。
    """
    asset: Articulation = env.scene["robot"]
    total_loss = torch.zeros(env.num_envs, device=asset.device)
    
    for joint_group_a, joint_group_b in mirror_joints:
        # 将关节名称转换为ID
        if isinstance(joint_group_a[0], str):
            ids_a = [asset.find_joints(name)[0][0] for name in joint_group_a]
        else:
            ids_a = joint_group_a
            
        if isinstance(joint_group_b[0], str):
            ids_b = [asset.find_joints(name)[0][0] for name in joint_group_b]
        else:
            ids_b = joint_group_b
        
        # 计算当前关节组的动作差异
        # action_buffer._circular_buffer.buffer: [num_envs, history, num_actions]
        actions_a = env.action_buffer._circular_buffer.buffer[:, -1, ids_a]  # [num_envs, len(ids_a)]
        actions_b = env.action_buffer._circular_buffer.buffer[:, -1, ids_b]  # [num_envs, len(ids_b)]
        
        # 计算对称关节组的L2差异
        group_diff = torch.sum(torch.square(actions_a - actions_b), dim=1)  # [num_envs]
        
        # 累积损失，归一化为关节组大小
        total_loss += group_diff / len(ids_a)
    
    return total_loss


# -------------------------
# Root / Pose
# -------------------------

def base_height_l2(env: BaseEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    # asset: Articulation = env.scene[asset_cfg.name]
    # if len(asset_cfg.body_ids) > 0:
    #     z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    # else:
    #     z = asset.data.root_pos_w[:, 2]
    # return torch.square(z - target_height)
    
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        ray_caster: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = ray_caster.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward

def root_lin_acc_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # body_lin_acc_w: [num_envs, num_bodies, 3]
    # [:, 0, 2] → body 0 (root/base link), index 2 = z-component
    a_z = asset.data.body_lin_acc_w[:, 0, 2]
    return torch.square(a_z)

# 重力在root link frame下的z分量 bigger, better
# 如果翻倒，z分量会变成负值 -> 惩罚
def upward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    return (-g_b[:, 2]) / torch.norm(g_b, dim=1)

# 奖励站稳
# def stand_still_without_cmd(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = env.command_generator.command  # [num_envs, 3] xyz
#     # still 是 bool tensor
#     # cmd xy线速度+z角速度 < 0.1 判定为静止
#     still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
#     v_lin = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
#     v_yaw = torch.abs(asset.data.root_ang_vel_w[:, 2])
#     # cmd < 0.1, still = 1, 速度/角速度越小，奖励越大
#     return -(v_lin + v_yaw) * still

def stand_still_without_cmd(
    env: BaseEnv,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_generator.command, dim=1) < command_threshold
    return reward


# -------------------------
# Contacts / Forces
# -------------------------

# 参考undersired_contacts
# def contact_forces(env, sensor_cfg: SceneEntityCfg, reduction: str = "sum") -> torch.Tensor:
#     """
#     汇总所选 body 的接触力范数。
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N,B,3] 没有历史？
#     fn = torch.norm(f, dim=-1)                                      # [N,B]
#     if reduction == "sum":
#         return torch.sum(fn, dim=1)
#     elif reduction == "mean":
#         return torch.mean(fn, dim=1)
#     else:
#         return torch.max(fn, dim=1)[0]


# -------------------------
# Feet / Gait (quadruped general)
# -------------------------

# 是否接触，重复了
def _contacts_bool(env: BaseEnv, sensor_cfg: SceneEntityCfg, thresh: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # [N,T,B,3]
    fmax = forces_hist.norm(dim=-1).max(dim=1)[0]  # [N,B]
    return fmax > thresh

# 悬空时间奖励
# current_air_time [N,B]的定义是：上次接触后悬空的时间
def feet_air_time(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 鼓励悬空时间
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [N,B]
    # reward = torch.sum(torch.clamp(air_time, max=threshold), dim=1) # [N,]
    # cmd = env.command_generator.command
    # moving = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1 # [N,] bool
    # return reward * moving
    
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # 只在脚刚落地时才有奖励，且悬空时间必须大于threshold
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids] # [N,B] bool tensor, first contact after being in the air
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids] # [N,B] float tensor, Time spent (in s) in the air before the last contact.
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1) # [N,] 
    # no reward for zero command
    reward *= torch.norm(env.command_generator.command[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_variance(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # air_time = torch.clamp(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], max=threshold)
    # return torch.var(air_time, dim=1, unbiased=False)
    
    # 奖励各脚悬空时间的一致性,惩罚频繁切换
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # assume > threshold -> complete air/contact phase
    # punish those feet that are frequently switching between air and contact
    reward = torch.var(torch.clip(last_air_time, threshold), dim=1) + torch.var(torch.clip(last_contact_time, threshold), dim=1)
    return reward

def feet_contact(env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    c = _contacts_bool(env, sensor_cfg, thresh=threshold)
    return torch.sum(c, dim=1).float()

def feet_contact_without_cmd(env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    # c = _contacts_bool(env, sensor_cfg, thresh=threshold)
    # num_c = torch.sum(c, dim=1).float()
    contact_sensor : ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids] # [N,B] bool tensor
    num_c = torch.sum(contact, dim=1).float()
    cmd = env.command_generator.command
    still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    return num_c * still

def feet_height(env: BaseEnv, target_height: float, tanh_mult: float, sensor_cfg: SceneEntityCfg,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # c = _contacts_bool(env, sensor_cfg)
    # asset: Articulation = env.scene[asset_cfg.name]
    # base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)                         # [N,1]
    # foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                   # [N,B]
    # z_rel  = foot_z - base_z                                                  # [N,B]
    # swing  = (~c)
    # gain   = torch.clamp(z_rel - target_height, min=0.0) * swing
    # return torch.sum(gain, dim=1)
    """Reward the swinging feet for clearing a specified height off the ground"""
    # in world frame
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_generator.command, dim=1) > 0.1
    return reward

def feet_height_body(env: BaseEnv, target_height: float, tanh_mult: float, sensor_cfg: SceneEntityCfg,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # asset: Articulation = env.scene[asset_cfg.name]
    # base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    # foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # z_rel  = foot_z - base_z
    # excess = (z_rel - target_height).clamp(min=0.0)
    # return torch.sum(excess, dim=1)
    
    # in body frame
    asset: Articulation = env.scene[asset_cfg.name]
    # [N,B,3] -> [N,B,3] - [N,1,3] -> [N,B,3]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_generator.command, dim=1) > 0.1
    return reward


# def feet_gait(env, sensor_cfg: SceneEntityCfg,
#               synced_feet_pair_ids: Optional[Tuple[Tuple[int,int], ...]] = None,
#               threshold: float = 1.0) -> torch.Tensor:
#     """
#     以接触状态一致性作为分数（1=完全同步）。
#     注意：synced_feet_pair_ids 索引应与 sensor_cfg.body_ids 的顺序一致。
#     """
#     c = _contacts_bool(env, sensor_cfg, thresh=threshold).float()  # [N,B]
#     if not synced_feet_pair_ids:
#         return torch.ones(c.shape[0], device=c.device)
#     scores = []
#     for a, b in synced_feet_pair_ids:
#         scores.append(1.0 - torch.abs(c[:, a] - c[:, b]))
#     return torch.stack(scores, dim=1).mean(dim=1)

# 同步奖励，强调的是air time的一致性，而不是关节
class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: BaseEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: BaseEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # example, input (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # [0][0] = FL, [0][1] = RR
        # [1][0] = FR, [1][1] = RL
        
        # for synchronous feet, the contact (air) times of two feet should match
        # sync_reward_0 : FL and RR
        # sync_reward_1 : FR and RL
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        # 奖励异步
        # async_reward_0 : FL and FR
        # async_reward_1 : RR and RL
        # async_reward_2 : FL and RL
        # async_reward_3 : RR and FR
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        # cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        cmd = torch.linalg.norm(env.command_generator.command, dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        # body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        # 当cmd和body_vel都大于阈值时-> 开始运动时，才计算reward，否则reward为0
        # torch.where(condition, x, y) -> condition为True时取x，否则取y
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


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

def action_smoothing(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # t动作
    action_t = env.action_buffer._circular_buffer.buffer[:, -1, asset_cfg.joint_ids]  # [N, J]
    # t-1动作
    action_t_1 = env.action_buffer._circular_buffer.buffer[:, -2, asset_cfg.joint_ids]  # [N, J]
    # t-2动作
    action_t_2 = env.action_buffer._circular_buffer.buffer[:, -3, asset_cfg.joint_ids]  # [N, J]
    
    action_change = action_t - action_t_1 - (action_t_1 - action_t_2)  # [N, J]
    return torch.sum(torch.square(action_change), dim=1)  # [N]