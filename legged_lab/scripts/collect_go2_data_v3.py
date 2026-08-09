#!/usr/bin/env python3
"""
Go2数据收集脚本 v3 - 速度跟随失效场景（V1: 摩擦力变化 / V2: 持续阻力）

基于 v2 扩展，新增：
  - V1 (--fault_type friction): T_fault 步骤后将足部摩擦系数降至 OOD 范围
  - V2 (--fault_type force_x): T_fault 步骤后对 base 施加持续 -x 方向外力
  - collection_mode=braking: 保持原数据结构，采集正常制动 command transition
  - T_fault 在 [t_fault_min, t_fault_max] 内随机化（防止 RSSM 学到时序伪相关）
  - 3D 速度跟随失效标签（vx/vy/yaw_rate 三分量分别检测）
  - t_fault / fault_type 写入 demo 元数据，供训练时分层采样使用
  - priv_state 扩展至 R^9：在原 R^7（base_lin_vel_b + feet_contact）基础上追加
      foot_static_friction (1): 当前 env 的足部静摩擦系数（正常轨迹为 DR 采样值；V1 触发后为 OOD 值）
      applied_force_x      (1): 当前 env 受到的 base x 向外力（N；V2 触发后非零，安全帧为 0）

数据格式与 v2 / generate_data_traj_cont.py 完全兼容，tools.py 通过
  demo.get('fault_type', 'fall') 判断是否跳过 gz/n_prefall 后处理。
priv_state 维度从 7 升至 9，旧数据集加载时在末尾 pad 零即可向后兼容。
"""

import argparse
import json
import pickle
import random
import torch
import numpy as np
from typing import Dict, List, Any
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from isaaclab.app import AppLauncher

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description="Go2 data collection script v3 - velocity tracking failure")
parser.add_argument("--task", type=str, default="go2_data_collection", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--num_episodes", type=int, default=25, help="Episodes per environment")
parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
parser.add_argument("--output_dir", type=str, default="./data/vel_failure", help="Output directory")
parser.add_argument("--save_interval", type=int, default=50, help="每收集多少个 episode 保存一次分片")
parser.add_argument("--no_final_merge", action="store_true", help="跳过末尾合并步骤")
parser.add_argument("--output_name", type=str, default=None, help="最终输出文件名（不含 .pkl）")

# ── V3: 扰动实验参数 ──────────────────────────────────────────────────────────
parser.add_argument("--fault_type", type=str, default="none",
    choices=["none", "friction", "force_x"],
    help="扰动类型: none=纯安全数据, friction=V1摩擦力变化, force_x=V2持续阻力")
parser.add_argument("--t_fault_min", type=int, default=30,
    help="扰动生效最早步数（含），建议 ≥ 20")
parser.add_argument("--t_fault_max", type=int, default=60,
    help="扰动生效最晚步数（含），建议 ≤ max_steps * 0.4")
parser.add_argument("--safe_ratio", type=float, default=0.2,
    help="每 episode 以此概率不施加扰动（纯安全轨迹比例）")
# V1: 摩擦力变化（每 episode 在范围内随机采样，增加扰动多样性）
parser.add_argument("--ood_static_friction_range", type=float, nargs=2, default=[0.08, 0.12],
    metavar=('LOW', 'HIGH'),
    help="V1: OOD 静摩擦系数范围 [LOW HIGH]，每 episode 均匀采样（训练正常范围 0.6-1.0）")
parser.add_argument("--ood_dynamic_friction_range", type=float, nargs=2, default=[0.04, 0.06],
    metavar=('LOW', 'HIGH'),
    help="V1: OOD 动摩擦系数范围 [LOW HIGH]，每 episode 均匀采样（训练正常范围 0.4-0.8）")
# V2: 持续阻力
parser.add_argument("--force_x_magnitude_min", type=float, default=-20.0,
    help="V2: base -x 方向持续力（牛顿，负值=阻力），建议 -30 ~ -80 N")
parser.add_argument("--force_x_magnitude_max", type=float, default=-80.0,
    help="V2: base -x 方向持续力（牛顿，负值=阻力），建议 -30 ~ -80 N")
parser.add_argument("--force_x_ramp_steps", type=int, default=0,
    help="V2: 从 0 线性斜坡到目标力值所需步数（0=瞬间施加）")
parser.add_argument("--min_fail_consecutive", type=int, default=5,
    help="V1/V2: 连续 failure 帧数达到此阈值才保留标签，消除速度噪声误标（0=不平滑）")
# Braking: 正常制动采集，不作为 fault_type，不改变 pkl demo 结构
parser.add_argument("--collection_mode", type=str, default="normal",
    choices=["normal", "braking"],
    help="数据采集模式: normal=原逻辑, braking=制动轨迹")
parser.add_argument("--brake_start_range", type=int, nargs=2, default=[50, 80],
    metavar=('LOW', 'HIGH'),
    help="braking: 制动开始步数范围 [LOW HIGH]，每 episode 均匀采样")
parser.add_argument("--brake_complete_ratio", type=float, default=0.5,
    help="braking: 完全制动轨迹比例；完全制动 target_vx=0，不完全制动 target_vx 从范围采样")
parser.add_argument("--brake_target_ratio_range", type=float, nargs=2, default=[0.2, 0.6],
    metavar=('LOW', 'HIGH'),
    help="braking: 不完全制动保留原始 vx command 的比例范围 [LOW HIGH]，例如 0.2 0.6")
# Stage-2 low-friction-brake: braking + fault_type=friction 组合模式
parser.add_argument("--post_fault_brake_delay_range", type=int, nargs=2, default=[15, 40],
    metavar=('LOW', 'HIGH'),
    help="low_friction_brake: t_brake = t_fault + delay，delay 从 [LOW HIGH] 均匀采样。"
         "保证先在低摩擦上走一段(检测延迟窗口)再制动。仅 collection_mode=braking & fault_type=friction 生效。")
parser.add_argument("--lowfric_walk", action="store_true", default=False,
    help="Stage-2 type-5 低摩擦 walk(z0/WM 表征用): normal+friction 下改用 fall/collision 语义"
         "(只有真摔=failure，打滑但没摔≠failure)，并写 fault_type='low_friction'/event_type='low_friction_walk'"
         "→ 训练侧走 fall 分支，保住 margin 的分级(否则低摩擦走路会被全标 unsafe)。")
# ─────────────────────────────────────────────────────────────────────────────

import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets.articulation import Articulation

import re as _re
def get_checkpoint_path(log_path, run_dir=".*", checkpoint=".*", other_dirs=None, sort_alpha=True):
    try:
        import os as _os
        runs = [_os.path.join(log_path, r) for r in _os.scandir(log_path) if r.is_dir() and _re.match(run_dir, r.name)]
        runs.sort() if sort_alpha else runs.sort(key=_os.path.getmtime)
        run_path = _os.path.join(runs[-1], *other_dirs) if other_dirs else runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in '{log_path}' matching '{run_dir}'.")
    model_checkpoints = [f for f in _os.listdir(run_path) if _re.match(checkpoint, f)]
    if not model_checkpoints:
        raise ValueError(f"No checkpoints in '{run_path}' matching '{checkpoint}'.")
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    return _os.path.join(run_path, model_checkpoints[-1])

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


# ========== 数据缓冲区 ==========

class TrajectoryBuffer:
    """单条轨迹缓冲区，格式与 v2 / generate_data_traj_cont.py 一致"""
    """
    单条轨迹的缓冲区，对应 generate_data_traj_cont.py 的一条 demo
    
    数据格式：
    demo = {
        'obs': {
            'image': [img_t0, img_t1, ...],        # List[np.ndarray]
            'state': [state_t0, state_t1, ...],    # List[np.ndarray] (42,): ang_vel, proj_grav, jpos, jvel, last_action
            'priv_state': [priv_t0, priv_t1, ...]  # List[np.ndarray] (9,): base_lin_vel_b(3), feet_contact(4), foot_static_friction(1), applied_force_x(1)
        },
        'rewards':[r_t0, r_t1, ...],               # List[float]
        'actions': [cmd_t0, cmd_t1, ...],          # List[np.ndarray] (3,): 速度指令 (vx, vy, yaw_rate)，WM 的控制输入
        'dones': [0, 0, ..., 1]                    # List[int]
        'failure': [0, 0, ..., 0] or [0, 0, ..., 1]
    }
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_list: List[np.ndarray] = []
        self.state_list: List[np.ndarray] = []
        self.priv_state_list: List[np.ndarray] = []
        self.reward_list: List[float] = []
        self.action_list: List[np.ndarray] = []
        self.done_list: List[int] = []
        self.failure_list: List[int] = []
        self.step_count = 0
        # 临时存储（step 内跨阶段传递）
        self._pending_obs = None
        self._pending_command = None

    def add_step(self, image, state, priv_state, reward, action, done, failure):
        self.image_list.append(image)
        self.state_list.append(state)
        self.priv_state_list.append(priv_state)
        self.reward_list.append(reward)
        self.action_list.append(action)
        self.done_list.append(done)
        self.failure_list.append(failure)
        self.step_count += 1

    def to_demo(self, env_id=None):
        demo = {
            'obs': {
                'image': self.image_list.copy(),
                'state': self.state_list.copy(),
                'priv_state': self.priv_state_list.copy()
            },
            'rewards': self.reward_list.copy(),
            'actions': self.action_list.copy(),
            'dones': self.done_list.copy(),
            'failure': self.failure_list.copy()
        }
        if env_id is not None:
            demo['env_id'] = env_id
        return demo

    def __len__(self):
        return self.step_count


class BatchedEnvData:
    """批量预提取所有环境的数据（单次 CUDA→CPU，避免 per-env sync）"""
    def __init__(self):
        self.ang_vel = None
        self.projected_gravity = None
        self.joint_pos = None
        self.joint_vel = None
        self.last_action = None
        self.base_lin_vel_w = None
        self.base_lin_vel_b = None
        self.feet_contact = None
        self.commands = None

    def fetch(self, env, robot: Articulation):
        self.ang_vel = robot.data.root_ang_vel_b.cpu().numpy()
        self.projected_gravity = robot.data.projected_gravity_b.cpu().numpy()
        self.joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos).cpu().numpy()
        self.joint_vel = robot.data.joint_vel.cpu().numpy()
        self.last_action = env.action_buffer._circular_buffer.buffer[:, -1, :].cpu().numpy()
        self.base_lin_vel_w = robot.data.root_lin_vel_w.cpu().numpy()
        self.base_lin_vel_b = robot.data.root_lin_vel_b.cpu().numpy()
        contact_forces = env.contact_sensor.data.net_forces_w_history
        feet_body_ids = env.feet_cfg.body_ids if hasattr(env, "feet_cfg") else slice(0, 4)
        feet_contact_t = (torch.max(torch.norm(contact_forces[:, :, feet_body_ids], dim=-1), dim=1)[0] > 0.5)
        self.feet_contact = feet_contact_t.cpu().numpy().astype(np.float32)
        self.commands = env.command_generator.command.cpu().numpy()


def extract_state_vector(batch: BatchedEnvData, env_idx: int) -> np.ndarray:
    return np.concatenate([
        batch.ang_vel[env_idx],
        batch.projected_gravity[env_idx],
        batch.joint_pos[env_idx],
        batch.joint_vel[env_idx],
        batch.last_action[env_idx]
    ])


def extract_priv_state_vector(batch: BatchedEnvData, env_idx: int) -> np.ndarray:
    """v2 兼容格式：R^7 (base_lin_vel_b + feet_contact)"""
    return np.concatenate([
        batch.base_lin_vel_b[env_idx],
        batch.feet_contact[env_idx]
    ])


def extract_priv_state_vector_v3(batch: BatchedEnvData, env_idx: int,
                                  foot_static_friction: float,
                                  applied_force_x: float) -> np.ndarray:
    """
    V3 扩展格式：R^9
      base_lin_vel_b       (3): 基座坐标系下 base 线速度
      feet_contact         (4): 四足接触状态（二值 0/1）
      foot_static_friction (1): 足部静摩擦系数（DR 采样值或 OOD 值；multiply 模式下等于有效摩擦）
      applied_force_x      (1): base x 向外力（N；安全帧为 0，V2 触发后为负值阻力）
    """
    return np.concatenate([
        batch.base_lin_vel_b[env_idx],
        batch.feet_contact[env_idx],
        np.array([foot_static_friction], dtype=np.float32),
        np.array([applied_force_x], dtype=np.float32)
    ])


def extract_image(env, env_idx: int) -> np.ndarray:
    try:
        if hasattr(env.scene, 'sensors') and 'front_camera' in env.scene.sensors:
            camera = env.scene.sensors['front_camera']
            if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                output = camera.data.output
                if 'rgb' in output:
                    return output['rgb'][env_idx].cpu().numpy()
                elif 'rgba' in output:
                    return output['rgba'][env_idx].cpu().numpy()[..., :3]
                elif 'distance_to_image_plane' in output:
                    return output['distance_to_image_plane'][env_idx].cpu().numpy()
    except Exception as e:
        print(f"[WARN] extract_image env{env_idx}: {e}")
    h = env.cfg.scene.camera.height
    w = env.cfg.scene.camera.width
    return np.zeros((h, w, 1), dtype=np.float32)


def atomic_save_shard(demos: List[Dict], shard_path: Path):
    tmp_path = shard_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(demos, f, protocol=4)
    tmp_path.rename(shard_path)
    print(f"   [分片保存] → {shard_path.name}  ({len(demos)} eps, "
          f"{shard_path.stat().st_size / 1024 / 1024:.1f} MB)")


def count_braking_stats(demos: List[Dict]):
    """返回 braking safe / fail reset 计数。"""
    n_total = len(demos)
    n_failed = sum(1 for d in demos if d.get('failure', [0]) and bool(d['failure'][-1]))
    n_safe = n_total - n_failed
    return n_safe, n_failed


def print_braking_stats(demos: List[Dict]):
    """打印 braking safe / fail reset 比例。"""
    n_safe, n_failed = count_braking_stats(demos)
    n_total = n_safe + n_failed
    denom = max(n_total, 1)
    print(f"   braking 统计: BRAKE_SAFE={n_safe}/{n_total} ({n_safe / denom:.1%}), "
          f"FAIL_RESET={n_failed}/{n_total} ({n_failed / denom:.1%})")


# ========== failure 标签后处理 ==========

def smooth_failure_labels(failure_list: List[int], min_consecutive: int) -> List[int]:
    """
    对 per-step 原始 failure 信号做连续帧平滑：
    只有连续 >= min_consecutive 帧的 failure run 才保留，孤立帧清零。
    episode 结束后对完整序列调用，避免速度噪声导致误标。

    示例 (min_consecutive=3):
      输入: [0,0,1,0,1,1,1,1,0,1,0]
      输出: [0,0,0,0,1,1,1,1,0,0,0]  # 长度>=3的run保留，短run清零
    """
    if min_consecutive <= 1:
        return list(failure_list)
    n = len(failure_list)
    result = [0] * n
    i = 0
    while i < n:
        if failure_list[i] == 1:
            j = i
            while j < n and failure_list[j] == 1:
                j += 1
            if (j - i) >= min_consecutive:
                for k in range(i, j):
                    result[k] = 1
            i = j
        else:
            i += 1
    return result


# ========== V2 兼容：原始 failure 判断（fault_type='none' 时使用）==========

def extract_failure(env, env_idx: int, is_done: bool, is_timeout: bool) -> int:
    """v2 原有逻辑：提前 reset（非 timeout）视为 failure"""
    if is_done and not is_timeout:
        return 1
    return 0


# ========== V3 新增：扰动相关函数 ==========

def apply_friction_fault(robot: Articulation, env_idx: int,
                          static_mu: float, dynamic_mu: float):
    """V1: 将指定 env 的所有材质摩擦系数修改为 OOD 值"""
    mats = robot.root_physx_view.get_material_properties()  # numpy (N, num_shapes, 3)
    mats[env_idx, :, 0] = static_mu
    mats[env_idx, :, 1] = dynamic_mu
    robot.root_physx_view.set_material_properties(
        mats, torch.tensor([env_idx], dtype=torch.int32)
    )


def restore_friction(robot: Articulation, env_idx: int, original_materials: np.ndarray):
    """V1: 将指定 env 的材质恢复为原始摩擦系数"""
    mats = robot.root_physx_view.get_material_properties()
    mats[env_idx] = original_materials[env_idx]
    robot.root_physx_view.set_material_properties(
        mats, torch.tensor([env_idx], dtype=torch.int32)
    )


def compute_vel_failure(batch: BatchedEnvData, env_idx: int,
                         current_step: int, t_fault: int,
                         is_safe: bool, v_cmd: np.ndarray,
                         applied_force_x: float = 0.0) -> int:
    """
    V1/V2 失效标签：基于 3D 速度跟随误差（任意分量超标 → failure=1）

    Grace period: t < t_fault + 5 时返回 0，避免过渡帧被误标为 failure。

    分量阈值：
      ex  = |v_cmd_x - v_x| / max(v_cmd_x, 0.5) > 0.3  （相对阈值，兜底 0.5m/s）
      ey  = |v_y| > 0.25 m/s                             （绝对阈值，避免正常步态侧摆误报）
      eyaw = |ang_vel_z| > 0.3 rad/s                     （绝对阈值）
    """
    if is_safe or current_step < t_fault + 5:
        return 0
    v = batch.base_lin_vel_b[env_idx]  # base frame velocity
    ex = abs(float(v_cmd[0]) - float(v[0])) / max(float(v_cmd[0]), 0.5) > 0.3
    ey = abs(float(v[1])) > 0.25
    eyaw = abs(float(batch.ang_vel[env_idx][2])) > 0.3
    print(f"   [Failure Check] env{env_idx} step={current_step} force_x={applied_force_x:.1f}N v_cmd={v_cmd} v={v} ex={ex} ey={ey} eyaw={eyaw}")
    return 1 if (ex or ey or eyaw) else 0


# ========== braking：command schedule ==========

def compute_braking_command(current_step: int,
                            brake_start_step: int,
                            original_cmd: np.ndarray,
                            target_vx: float) -> np.ndarray:
    """生成单步制动 command；制动后完整覆盖 [vx, vy, yaw]。"""
    if current_step < brake_start_step:
        return np.array(original_cmd, dtype=np.float32)
    return np.array([target_vx, 0.0, 0.0], dtype=np.float32)


def apply_command_override_to_obs(env, obs_dict, commands_t: torch.Tensor):
    """
    覆盖 command generator 并同步当前 policy/critic obs 中最新一帧的 command slice.

    BaseEnv 的单帧 actor obs 布局为:
      ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), action(12)
    因此 command slice 是 [6:9]。这里同时 patch obs buffer 和本轮 policy 输入，
    避免 policy 看到随机 command，而 dataset 记录 braking command。
    """
    env.command_generator.command[:, :] = commands_t
    scaled_commands = commands_t * env.obs_scales.commands
    cmd_start = 6
    cmd_end = 9

    actor_buf = env.actor_obs_buffer.buffer
    actor_buf[:, -1, cmd_start:cmd_end] = scaled_commands
    actor_frame_dim = actor_buf.shape[-1]
    actor_hist_len = actor_buf.shape[1]
    actor_flat_start = (actor_hist_len - 1) * actor_frame_dim + cmd_start
    if "policy" in obs_dict and obs_dict["policy"].shape[-1] >= actor_flat_start + 3:
        obs_dict["policy"][:, actor_flat_start:actor_flat_start + 3] = scaled_commands

    if hasattr(env, "critic_obs_buffer") and "critic" in obs_dict:
        critic_buf = env.critic_obs_buffer.buffer
        if critic_buf.shape[-1] >= cmd_end:
            critic_buf[:, -1, cmd_start:cmd_end] = scaled_commands
            critic_frame_dim = critic_buf.shape[-1]
            critic_hist_len = critic_buf.shape[1]
            critic_flat_start = (critic_hist_len - 1) * critic_frame_dim + cmd_start
            if obs_dict["critic"].shape[-1] >= critic_flat_start + 3:
                obs_dict["critic"][:, critic_flat_start:critic_flat_start + 3] = scaled_commands
    return obs_dict


# ========== 主采集函数 ==========

def collect_data():
    if args.collection_mode == 'braking' and args.fault_type == 'force_x':
        raise ValueError("--collection_mode braking 暂不支持 fault_type=force_x；仅支持 none 或 friction(=stage-2 低摩擦制动)。")
    if args.collection_mode == 'braking' and args.fault_type == 'friction':
        print("[INFO] Stage-2 low_friction_brake 组合模式: 先在 t_fault 掉低摩擦，再 t_brake=t_fault+delay 制动。")
    if args.lowfric_walk and not (args.collection_mode == 'normal' and args.fault_type == 'friction'):
        raise ValueError("--lowfric_walk 仅支持 --collection_mode normal --fault_type friction(低摩擦走路,不制动)。")
    if args.lowfric_walk:
        print("[INFO] Stage-2 type-5 低摩擦 walk: fall 语义 + low_friction 元数据(不制动，专供 terrain×low 的 z0/表征)。")

    # ── 环境配置 ──────────────────────────────────────────────────────────────
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False

    if not getattr(args, 'enable_cameras', False):
        env_cfg.scene.camera.enable_camera = False
        env_cfg.scene.camera.use_physical_asset = False
        print("[INFO] 相机已禁用（未传 --enable_cameras）")

    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt

    # 创建环境
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    num_envs = env.num_envs
    print(f"\n[INFO] 环境创建完成: {num_envs} envs, {env.num_actions} actions")

    # ── 加载策略 ──────────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] 加载策略: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print("[INFO] 策略加载成功")

    # ── 机器人引用 & V3 扰动初始化 ────────────────────────────────────────────
    robot: Articulation = env.scene["robot"]
    batch_data = BatchedEnvData()

    # V1: original_materials 在预热后读取（需等待 DR reset 后得到真实 per-env 摩擦值）
    original_materials = None
    if args.fault_type == 'friction':
        print(f"[INFO] V1 摩擦力实验")
        print(f"       OOD static range:  [{args.ood_static_friction_range[0]}, {args.ood_static_friction_range[1]}]")
        print(f"       OOD dynamic range: [{args.ood_dynamic_friction_range[0]}, {args.ood_dynamic_friction_range[1]}]")

    # V2: 预分配持续力缓冲区（每步更新后调用 apply_forces_and_torques_at_position）
    persistent_forces: torch.Tensor = None
    persistent_torques: torch.Tensor = None
    persistent_positions: torch.Tensor = None
    all_env_ids_tensor: torch.Tensor = None
    if args.fault_type == 'force_x':
        n_bodies = robot.num_bodies
        persistent_forces = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        persistent_torques = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        persistent_positions = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        all_env_ids_tensor = torch.arange(num_envs, dtype=torch.int32, device=env.device)
        print(f"[INFO] V2 持续阻力实验")
        print(f"       force_x range=[{args.force_x_magnitude_min}, {args.force_x_magnitude_max}] N, num_bodies={n_bodies}")

    # V3: per-env 扰动状态
    env_t_fault = [0] * num_envs              # 本 episode 的扰动触发步数
    env_is_safe = [True] * num_envs           # 本 episode 是否为安全轨迹
    env_fault_applied = [False] * num_envs    # V1/V2: 是否已触发扰动
    env_force_active = [False] * num_envs     # V2: 是否正在施加持续力
    env_sampled_static_mu = [0.0] * num_envs  # V1: 本 episode 采样的 OOD 静摩擦系数
    env_sampled_dynamic_mu = [0.0] * num_envs # V1: 本 episode 采样的 OOD 动摩擦系数
    env_sampled_force_x = [0.0] * num_envs    # V2: 本 episode 采样的 force_x 大小（负值）
    env_force_steps = [0] * num_envs           # V2: fault 触发后已经过的步数（用于斜坡计算）
    env_brake_start_step = [0] * num_envs
    env_brake_original_cmd = [np.zeros(3, dtype=np.float32) for _ in range(num_envs)]
    env_brake_target_vx = [0.0] * num_envs
    env_brake_is_complete = [True] * num_envs

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    def sample_braking_base_command() -> np.ndarray:
        """Sample a fresh non-braked command from the task command ranges."""
        ranges = env.cfg.commands.ranges
        vx = random.uniform(float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1]))
        vy = random.uniform(float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1]))
        yaw = random.uniform(float(ranges.ang_vel_z[0]), float(ranges.ang_vel_z[1]))
        return np.array([vx, vy, yaw], dtype=np.float32)

    def resample_episode_params(env_idx: int):
        """episode 开始时为该 env 重新采样扰动参数"""
        if args.fault_type == 'none':
            env_is_safe[env_idx] = True
        else:
            env_is_safe[env_idx] = (random.random() < args.safe_ratio)
        env_t_fault[env_idx] = random.randint(args.t_fault_min, args.t_fault_max)
        env_fault_applied[env_idx] = False
        env_force_active[env_idx] = False
        env_force_steps[env_idx] = 0
        # V1: 每 episode 独立采样 OOD 摩擦系数，增加扰动多样性
        if args.fault_type == 'friction':
            env_sampled_static_mu[env_idx] = random.uniform(
                args.ood_static_friction_range[0], args.ood_static_friction_range[1]
            )
            env_sampled_dynamic_mu[env_idx] = random.uniform(
                args.ood_dynamic_friction_range[0], args.ood_dynamic_friction_range[1]
            )
        # V2: 每 episode 独立采样 force_x 大小，增加扰动多样性
        if args.fault_type == 'force_x':
            env_sampled_force_x[env_idx] = random.uniform(
                args.force_x_magnitude_min, args.force_x_magnitude_max
            )

    def configure_braking_episode(env_idx: int):
        """reset 后采样本 episode 原始 command，并采样制动开始时间/目标速度。"""
        lo, hi = args.brake_start_range
        lo = max(0, min(lo, hi))
        hi = min(max(lo, hi), max(args.max_steps - 1, 0))
        # Stage-2 组合模式：非 safe 的 fault episode，制动锚定在 t_fault 之后 delay 步，
        # 保证"先正常走 → t_fault 掉低摩擦 → 走 delay 步(检测窗口) → t_brake 制动"。
        # safe(control) episode 或纯 braking 用原有 brake_start_range。
        if args.fault_type == 'friction' and not env_is_safe[env_idx]:
            d_lo, d_hi = args.post_fault_brake_delay_range
            d_lo = max(1, int(min(d_lo, d_hi)))
            d_hi = max(d_lo, int(d_hi))
            delay = random.randint(d_lo, d_hi)
            brake_start = env_t_fault[env_idx] + delay
            env_brake_start_step[env_idx] = int(min(brake_start, max(args.max_steps - 1, 0)))
        else:
            env_brake_start_step[env_idx] = random.randint(lo, hi)
        # Do not read the current command here: after a braking reset it may still
        # be the previous low target. Sample a fresh base command so ratios do not
        # compound across episodes.
        original_cmd = sample_braking_base_command()
        env.command_generator.command[env_idx] = torch.as_tensor(
            original_cmd, dtype=torch.float32, device=env.device
        )
        env_brake_original_cmd[env_idx] = original_cmd

        is_complete = random.random() < args.brake_complete_ratio
        if is_complete:
            target_vx = 0.0
        else:
            v0 = max(float(original_cmd[0]), 0.0)
            ratio_lo, ratio_hi = args.brake_target_ratio_range
            ratio_lo = max(0.0, min(float(ratio_lo), 1.0))
            ratio_hi = max(ratio_lo, min(float(ratio_hi), 1.0))
            ratio = random.uniform(ratio_lo, ratio_hi)
            target_vx = v0 * ratio
        env_brake_target_vx[env_idx] = target_vx
        env_brake_is_complete[env_idx] = is_complete

    def apply_braking_commands_for_current_buffers(obs_dict):
        commands = torch.zeros((num_envs, 3), dtype=torch.float32, device=env.device)
        for env_idx in range(num_envs):
            if episodes_completed[env_idx] >= args.num_episodes:
                commands[env_idx] = env.command_generator.command[env_idx]
                continue
            current_step = traj_buffers[env_idx].step_count
            commands[env_idx] = torch.as_tensor(
                compute_braking_command(
                    current_step,
                    env_brake_start_step[env_idx],
                    env_brake_original_cmd[env_idx],
                    env_brake_target_vx[env_idx],
                ),
                dtype=torch.float32,
                device=env.device,
            )
        return apply_command_override_to_obs(env, obs_dict, commands)

    for i in range(num_envs):
        resample_episode_params(i)

    # ── 轨迹缓冲区 & 计数器 ───────────────────────────────────────────────────
    traj_buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    episodes_completed = [0] * num_envs
    total_target = num_envs * args.num_episodes

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    if args.collection_mode == 'braking':
        braking_cfg_path = output_path / "braking_config.json"
        with open(braking_cfg_path, 'w') as f:
            json.dump({
                "collection_mode": args.collection_mode,
                "max_steps": args.max_steps,
                "brake_start_range": args.brake_start_range,
                "brake_complete_ratio": args.brake_complete_ratio,
                "brake_target_ratio_range": args.brake_target_ratio_range,
                "command_ranges_from_task": args.task,
                "fault_type": args.fault_type,
                "post_fault_brake_delay_range": args.post_fault_brake_delay_range if args.fault_type == 'friction' else None,
                "ood_static_friction_range": args.ood_static_friction_range if args.fault_type == 'friction' else None,
                "t_fault_range": [args.t_fault_min, args.t_fault_max] if args.fault_type == 'friction' else None,
                "safe_ratio": args.safe_ratio,
                "notes": "priv_state now 9D (base_lin_vel_b3 + feet_contact4 + foot_static_friction1 + applied_force_x1). "
                         "braking writes t_brake; fault_type=friction adds low_friction_brake with t_fault. "
                         "failure uses fall/collision (early-reset) semantics.",
            }, f, indent=2)
        print(f"[INFO] braking 配置已保存: {braking_cfg_path}")
    shard_idx = 0
    total_collected_saved = 0
    total_brake_safe_saved = 0
    total_brake_failed_saved = 0
    pending_demos: List[Dict] = []

    print(f"\n[INFO] 开始数据收集")
    print(f"   collection_mode={args.collection_mode}, fault_type={args.fault_type}, safe_ratio={args.safe_ratio}")
    if args.fault_type != 'none':
        print(f"   t_fault 范围: [{args.t_fault_min}, {args.t_fault_max}]")
    if args.collection_mode == 'braking':
        print(f"   braking start_range={args.brake_start_range}, complete_ratio={args.brake_complete_ratio}, "
              f"target_ratio_range={args.brake_target_ratio_range}")
    print(f"   每环境 episodes: {args.num_episodes}, 总目标: {total_target}")
    print(f"   每 episode 最大步数: {args.max_steps}")

    # ── 初始化 & 预热 ──────────────────────────────────────────────────────────
    all_env_ids = torch.arange(num_envs, device=env.device)
    env.reset(all_env_ids)
    obs_dict = env.get_observations()
    if args.collection_mode == 'braking':
        for env_idx in range(num_envs):
            configure_braking_episode(env_idx)
        obs_dict = apply_braking_commands_for_current_buffers(obs_dict)

    if getattr(args, 'enable_cameras', False):
        print("[INFO] 预热相机渲染器（5帧）...")
        for _ in range(5):
            simulation_app.update()

    WARMUP_STEPS = 50
    print(f"[INFO] 物理预热（{WARMUP_STEPS} 步）...")
    dummy_actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    for _ in range(WARMUP_STEPS):
        obs_dict, _, _, _ = env.step(dummy_actions)
    env.reset(all_env_ids)
    obs_dict = env.get_observations()
    if args.collection_mode == 'braking':
        for env_idx in range(num_envs):
            configure_braking_episode(env_idx)
        obs_dict = apply_braking_commands_for_current_buffers(obs_dict)
    print("[INFO] 预热完成，开始收集")

    # 读取 DR 初始化后的 per-env 摩擦系数（预热 reset 后才得到真实 DR 值）
    # DR event 是 mode="reset"，episode 内固定；V1 fault 触发后会被替换为 OOD 值
    _init_mats = robot.root_physx_view.get_material_properties()
    env_current_static_friction = [float(_init_mats[i, 0, 0]) for i in range(num_envs)]
    env_current_force_x = [0.0] * num_envs  # V2 fault 触发前为 0
    if args.fault_type == 'friction':
        original_materials = _init_mats.clone() if hasattr(_init_mats, 'clone') else _init_mats.copy()
        print(f"[INFO] 初始静摩擦（各 env，DR 随机化后）: {[f'{v:.3f}' for v in env_current_static_friction]}")

    step_count = 0

    # ── 主循环 ─────────────────────────────────────────────────────────────────
    while simulation_app.is_running():
        with torch.inference_mode():
            if sum(episodes_completed) >= total_target:
                print(f"\n[INFO] 数据收集完成! 共 {total_collected_saved + len(pending_demos)} 条轨迹")
                break

            # ===== Step 1: 获取动作 =====
            actions = policy(obs_dict)

            # ===== Step 2: 批量提取观测 & 检查 fault 触发 =====
            batch_data.fetch(env, robot)

            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue

                buffer = traj_buffers[env_idx]
                current_step = buffer.step_count  # 即将执行的步骤编号（0-indexed）

                # V3: 在 env.step 前检查 fault 触发时刻
                if (not env_is_safe[env_idx] and
                        not env_fault_applied[env_idx] and
                        current_step == env_t_fault[env_idx]):
                    if args.fault_type == 'friction':
                        apply_friction_fault(robot, env_idx,
                                             env_sampled_static_mu[env_idx],
                                             env_sampled_dynamic_mu[env_idx])
                        env_fault_applied[env_idx] = True
                        env_current_static_friction[env_idx] = env_sampled_static_mu[env_idx]  # priv_state 同步更新
                        print(f"   [V1] env{env_idx} step{current_step}: "
                              f"static={env_sampled_static_mu[env_idx]:.3f} "
                              f"dynamic={env_sampled_dynamic_mu[env_idx]:.3f}")
                    elif args.fault_type == 'force_x':
                        env_force_active[env_idx] = True
                        env_fault_applied[env_idx] = True
                        env_force_steps[env_idx] = 0
                        print(f"   [V2] env{env_idx} step{current_step}: 持续阻力已启动 (目标={env_sampled_force_x[env_idx]:.1f} N, ramp={args.force_x_ramp_steps}步)")

                # V2: 每步更新斜坡力（fault 触发后逐步增大到目标值）
                if env_force_active[env_idx]:
                    ramp = args.force_x_ramp_steps
                    if ramp <= 0:
                        ratio = 1.0
                    else:
                        ratio = min(env_force_steps[env_idx] / ramp, 1.0)
                    env_current_force_x[env_idx] = env_sampled_force_x[env_idx] * ratio
                    env_force_steps[env_idx] += 1

                # 提取当前帧观测（存入 pending，done 信号在 env.step 后才有）
                # priv_state (R^9)：本帧的 foot_static_friction / applied_force_x 已在 fault
                # 触发时同步更新，此处直接读取；V1 fault 在本步骤的 fetch 之后、step 之前触发，
                # 故本帧 priv_state 仍为触发前的正常摩擦值（下一帧才反映 OOD）
                current_command = batch_data.commands[env_idx].copy()
                image = extract_image(env, env_idx)
                state = extract_state_vector(batch_data, env_idx)
                # 统一记录 9D priv_state（含 foot_static_friction），braking 也记真实 DR/OOD 摩擦，
                # 供 stage-2 低摩擦制动的 friction 门控使用。
                priv_state = extract_priv_state_vector_v3(
                    batch_data, env_idx,
                    env_current_static_friction[env_idx],
                    env_current_force_x[env_idx]
                )
                buffer._pending_obs = (image.copy(), state.copy(), priv_state.copy())
                buffer._pending_command = current_command

            # ===== Step 2b: V2 持续力（在 env.step 前每步刷新）=====
            if args.fault_type == 'force_x':
                persistent_forces.zero_()
                for env_idx in range(num_envs):
                    if env_force_active[env_idx]:
                        # body 0 = root/base, 施加 -x 方向（opposing forward motion），已含斜坡系数
                        persistent_forces[env_idx, 0, 0] = env_current_force_x[env_idx]
                robot.root_physx_view.apply_forces_and_torques_at_position(
                    persistent_forces,
                    persistent_torques,
                    persistent_positions,
                    all_env_ids_tensor,
                    False               # is_global=True → world frame
                )

            # ===== Step 3: 执行物理步骤 =====
            next_obs_dict, rewards, dones, extras = env.step(actions)

            # ===== Step 4: 记录数据 & 处理 episode 结束 =====
            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue

                buffer: TrajectoryBuffer = traj_buffers[env_idx]
                image, state, priv_state = buffer._pending_obs
                command = buffer._pending_command
                current_step = buffer.step_count  # 当前步骤编号（add_step 前）

                # done 判断
                script_timeout = buffer.step_count >= args.max_steps - 1
                isaac_time_outs = extras.get("time_outs", torch.zeros(num_envs, dtype=torch.bool, device=env.device))
                isaac_timeout = bool(isaac_time_outs[env_idx].item())
                is_timeout = script_timeout or isaac_timeout
                is_env_done = bool(dones[env_idx].item() if isinstance(dones[env_idx], torch.Tensor) else dones[env_idx])
                is_done = is_timeout or is_env_done

                # ── failure 标签 ──────────────────────────────────────────────
                # braking(含 stage-2 低摩擦制动)始终用 fall/collision 语义(提前 reset=failure)，
                # 不用 vel-tracking OOD 判据——低摩擦但没摔的轨迹不应被误标为 failure。
                if args.collection_mode != 'braking' and args.fault_type in ('friction', 'force_x') \
                        and not args.lowfric_walk:
                    # V1/V2: 基于 3D 速度跟随误差
                    failure = compute_vel_failure(
                        batch_data, env_idx,
                        current_step,
                        env_t_fault[env_idx],
                        env_is_safe[env_idx],
                        command,
                        applied_force_x=env_current_force_x[env_idx]
                    )
                else:
                    # none/braking: 提前 reset = failure。制动碰撞 reset 是有意义的失败轨迹。
                    failure = extract_failure(env, env_idx, is_done, is_timeout)

                reward = rewards[env_idx].item()

                buffer.add_step(
                    image=image,
                    state=state,
                    priv_state=priv_state,
                    reward=reward,
                    action=command,
                    done=1 if is_done else 0,
                    failure=failure
                )

                if is_done:
                    # ── 组装 demo。写 t_brake / t_fault / fault_type / event_type 供分层采样。 ─────
                    demo = buffer.to_demo(env_id=env_idx)
                    if args.collection_mode == 'braking':
                        # 始终写 t_brake；fault episode 额外写 t_fault，标 low_friction_brake。
                        # fault_type 写 'low_friction_brake'/'brake'(均非 friction/force_x) →
                        # 训练侧 loader 走 fall 语义(n_prefall hindsight + gz onset + failure-based terminal)。
                        demo['t_brake'] = int(env_brake_start_step[env_idx])
                        is_lowfric = (args.fault_type == 'friction' and not env_is_safe[env_idx])
                        demo['t_fault'] = int(env_t_fault[env_idx]) if is_lowfric else -1
                        demo['fault_type'] = 'low_friction_brake' if is_lowfric else 'brake'
                        demo['event_type'] = demo['fault_type']
                    elif args.lowfric_walk:
                        # type-5 低摩擦 walk：fall 语义 + low_friction 元数据(非 'friction' → loader 走 fall 分支)。
                        is_lowfric = not env_is_safe[env_idx]
                        demo['t_fault'] = int(env_t_fault[env_idx]) if is_lowfric else -1
                        demo['fault_type'] = 'low_friction' if is_lowfric else 'none'
                        demo['event_type'] = 'low_friction_walk' if is_lowfric else 'normal'
                    elif args.fault_type in ('friction', 'force_x'):
                        demo['t_fault'] = env_t_fault[env_idx] if not env_is_safe[env_idx] else -1
                        demo['fault_type'] = (args.fault_type if not env_is_safe[env_idx] else 'none')

                        # ── V1/V2: 连续帧平滑（episode 结束后对完整序列处理）────
                        if args.min_fail_consecutive > 1:
                            demo['failure'] = smooth_failure_labels(
                                demo['failure'], args.min_fail_consecutive
                            )

                    pending_demos.append(demo)
                    episodes_completed[env_idx] += 1

                    n_fail = sum(demo['failure'])
                    if args.collection_mode == 'braking':
                        status = "FAIL_RESET" if n_fail > 0 else "BRAKE_SAFE"
                    elif args.fault_type == 'none':
                        status = "FAIL" if n_fail > 0 else "SAFE"
                    else:
                        status = "FAIL" if env_fault_applied[env_idx] and n_fail > 0 else \
                                 "SAFE" if env_is_safe[env_idx] else "NO_FAIL"
                    t_f = env_t_fault[env_idx] if not env_is_safe[env_idx] else -1
                    if args.collection_mode == 'braking':
                        is_lowfric = (args.fault_type == 'friction' and not env_is_safe[env_idx])
                        tf_str = f" t_fault={env_t_fault[env_idx]}" if is_lowfric else ""
                        step_str = f"brake_start={env_brake_start_step[env_idx]}{tf_str}"
                    else:
                        step_str = f"t_fault={t_f}"
                    print(f"   env{env_idx} | ep{episodes_completed[env_idx]}/{args.num_episodes} | "
                          f"步数={len(buffer)} {step_str} fail_frames={n_fail} | {status} | "
                          f"总进度={sum(episodes_completed)}/{total_target}")

                    # ── V1: 恢复摩擦力（reset 前撤销 OOD 设置，之后 DR 会重新随机化）────
                    if args.fault_type == 'friction' and env_fault_applied[env_idx]:
                        restore_friction(robot, env_idx, original_materials)

                    # ── V2: 停止持续力 ────────────────────────────────────────
                    # （下次循环 persistent_forces_np[:]=0.0 会自动清零该 env）
                    env_force_active[env_idx] = False

                    # 重置缓冲区 & 采样新 episode 参数
                    buffer.reset()
                    resample_episode_params(env_idx)

                    # 分片保存
                    if len(pending_demos) >= args.save_interval:
                        if args.collection_mode == 'braking':
                            n_safe_chunk, n_failed_chunk = count_braking_stats(pending_demos)
                            total_brake_safe_saved += n_safe_chunk
                            total_brake_failed_saved += n_failed_chunk
                        shard_file = shard_dir / f"shard_{shard_idx:04d}.pkl"
                        atomic_save_shard(pending_demos, shard_file)
                        shard_idx += 1
                        total_collected_saved += len(pending_demos)
                        pending_demos.clear()

                    # 脚本 timeout 时主动 reset（Isaac Lab 不会自动 reset）
                    if not is_env_done:
                        env.reset(torch.tensor([env_idx], device=env.device))
                    if args.collection_mode == 'braking':
                        configure_braking_episode(env_idx)

                    # 读取新 episode 经 DR 重新采样的摩擦系数（reset 已发生，DR 已更新）
                    # 同时更新 original_materials[env_idx]，供下次 restore_friction 使用
                    _post_mats = robot.root_physx_view.get_material_properties()
                    env_current_static_friction[env_idx] = float(_post_mats[env_idx, 0, 0])
                    env_current_force_x[env_idx] = 0.0
                    if args.fault_type == 'friction':
                        original_materials[env_idx] = _post_mats[env_idx]

            obs_dict = next_obs_dict
            if args.collection_mode == 'braking':
                obs_dict = apply_braking_commands_for_current_buffers(obs_dict)
            step_count += 1

            if step_count % 100 == 0:
                print(f"   Step {step_count} | 已收集: {sum(episodes_completed)}/{total_target} episodes")

    # ── 保存剩余分片 ──────────────────────────────────────────────────────────
    if pending_demos:
        if args.collection_mode == 'braking':
            n_safe_chunk, n_failed_chunk = count_braking_stats(pending_demos)
            total_brake_safe_saved += n_safe_chunk
            total_brake_failed_saved += n_failed_chunk
        shard_file = shard_dir / f"shard_{shard_idx:04d}.pkl"
        atomic_save_shard(pending_demos, shard_file)
        shard_idx += 1
        total_collected_saved += len(pending_demos)
        pending_demos.clear()

    if args.no_final_merge:
        print(f"\n[INFO] --no_final_merge 已设置，跳过合并")
        print(f"   分片目录: {shard_dir}  ({shard_idx} 个分片, {total_collected_saved} eps)")
        if args.collection_mode == 'braking':
            total = total_brake_safe_saved + total_brake_failed_saved
            denom = max(total, 1)
            print(f"   braking 统计: BRAKE_SAFE={total_brake_safe_saved}/{total} "
                  f"({total_brake_safe_saved / denom:.1%}), "
                  f"FAIL_RESET={total_brake_failed_saved}/{total} "
                  f"({total_brake_failed_saved / denom:.1%})")
        return

    # ── 合并所有分片 ──────────────────────────────────────────────────────────
    print(f"\n[INFO] 合并 {shard_idx} 个分片（共 {total_collected_saved} eps）...")
    if args.output_name:
        output_file = output_path / f"{args.output_name}.pkl"
    elif args.collection_mode == 'braking':
        output_file = output_path / f"go2_braking_{total_collected_saved}ep.pkl"
    else:
        output_file = output_path / f"go2_{args.fault_type}_{total_collected_saved}ep.pkl"

    tmp_output = output_file.with_suffix('.tmp')
    all_merged: List[Dict] = []
    with open(tmp_output, 'wb') as fout:
        for i in range(shard_idx):
            shard_file = shard_dir / f"shard_{i:04d}.pkl"
            with open(shard_file, 'rb') as f:
                chunk = pickle.load(f)
            all_merged.extend(chunk)
            print(f"   [{i+1}/{shard_idx}] 已读取 {len(chunk)} eps")
        pickle.dump(all_merged, fout, protocol=4)
    tmp_output.rename(output_file)

    print(f"[INFO] 最终文件: {output_file}")
    print(f"   总 episodes: {len(all_merged)}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 统计 failure/safe 分布。braking 数据不写 fault_type，按最后一帧 failure 判断。
    n_failed = sum(1 for d in all_merged if d.get('failure', [0]) and bool(d['failure'][-1]))
    n_safe = len(all_merged) - n_failed
    print(f"   failure 轨迹: {n_failed}, safe 轨迹: {n_safe}")
    if args.collection_mode == 'braking':
        print_braking_stats(all_merged)

    import shutil
    shutil.rmtree(shard_dir)
    print(f"[INFO] 已清理分片目录: {shard_dir}")

    # 打印示例
    if all_merged:
        s = all_merged[0]
        print(f"\n[INFO] 数据结构示例（第1条）:")
        print(f"   trajectory_length={len(s['actions'])}, top_keys={sorted(s.keys())}")
        print(f"   state shape:      {s['obs']['state'][0].shape}")
        print(f"   priv_state shape: {s['obs']['priv_state'][0].shape}")
        print(f"   action shape:     {s['actions'][0].shape}")
        print(f"   failure labels:   {s['failure'][:5]} ... {s['failure'][-5:]}")


if __name__ == "__main__":
    collect_data()
    simulation_app.close()
