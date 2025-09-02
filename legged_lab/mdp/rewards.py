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

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv

# velocity tracking functions
def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# root / pose functions
def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

# joints / actuation functions
def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

# contact/force functions
def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


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

def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau), dim=1)

def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(qd), dim=1)

def joint_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q  = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lo = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 0]
    hi = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 1]
    below = (lo - q).clamp(min=0.0)
    above = (q - hi).clamp(min=0.0)
    return torch.sum(below + above, dim=1)

def joint_vel_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "joint_vel_limits"):
        return torch.zeros(q.shape[0], device=asset.data.joint_vel.device)
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vmax = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    excess = (qd.abs() - vmax).clamp(min=0.0)
    return torch.sum(excess, dim=1)

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

def upward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    return (-g_b[:, 2]) / torch.norm(g_b, dim=1)

def stand_still_without_cmd(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    当期望速度≈0时，惩罚自身的线/角速度（鼓励站稳）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_generator.command  # [N, 3] = [vx, vy, wz]
    still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    v_lin = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    v_yaw = torch.abs(asset.data.root_ang_vel_w[:, 2])
    return (v_lin + v_yaw) * still


# -------------------------
# Contacts / Forces
# -------------------------

def contact_forces(env, sensor_cfg: SceneEntityCfg, reduction: str = "sum") -> torch.Tensor:
    """
    汇总所选 body 的接触力范数。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N,B,3]
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

def _contacts_bool(env, sensor_cfg: SceneEntityCfg, thresh: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # [N,T,B,3]
    fmax = forces_hist.norm(dim=-1).max(dim=1)[0]  # [N,B]
    return fmax > thresh

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


