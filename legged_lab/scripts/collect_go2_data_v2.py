#!/usr/bin/env python3
"""
Go2数据收集脚本 v2 - 修复数据格式问题
数据格式严格对齐 generate_data_traj_cont.py

核心思路：
- 每个环境独立收集轨迹
- 一条轨迹 = 一个 episode
- 总轨迹数 = num_envs * num_episodes
- 数据格式与 generate_data_traj_cont.py 完全一致
"""

import argparse
import pickle
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
parser = argparse.ArgumentParser(description="Go2 data collection script v2")
parser.add_argument("--task", type=str, default="go2_data_collection", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes per environment")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--output_dir", type=str, default="./data/go2_demo", help="Output directory")
parser.add_argument("--policy_task", type=str, default="go2_flat", help="Task to load policy from")

import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.assets.articulation import Articulation

from legged_lab.envs import *  # noqa:F401, F403  
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


class TrajectoryBuffer:
    """
    单条轨迹的缓冲区，对应 generate_data_traj_cont.py 的一条 demo
    
    数据格式：
    demo = {
        'obs': {
            'image': [img_t0, img_t1, ...],        # List[np.ndarray]
            'state': [state_t0, state_t1, ...],    # List[np.ndarray] 
            'priv_state': [priv_t0, priv_t1, ...]  # List[np.ndarray]
        },
        'actions': [ac_t0, ac_t1, ...],            # List[np.ndarray]
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
        self.action_list: List[np.ndarray] = []
        self.done_list: List[int] = []
        self.failure_list: List[int] = []
        self.step_count = 0
    
    def add_step(self, image: np.ndarray, state: np.ndarray, priv_state: np.ndarray, 
                 action: np.ndarray, done: int, failure: int):
        """添加一个时间步的数据"""
        self.image_list.append(image)
        self.state_list.append(state)
        self.priv_state_list.append(priv_state)
        self.action_list.append(action)
        self.done_list.append(done)
        self.failure_list.append(failure)
        self.step_count += 1
    
    def to_demo(self, env_id: int = None) -> Dict:
        """转换为 demo 格式"""
        demo = {
            'obs': {
                'image': self.image_list.copy(),
                'state': self.state_list.copy(),
                'priv_state': self.priv_state_list.copy()
            },
            'actions': self.action_list.copy(),
            'dones': self.done_list.copy(),
            'failure': self.failure_list.copy()
        }
        if env_id is not None:
            demo['env_id'] = env_id
        return demo
    
    def __len__(self):
        return self.step_count


def extract_state_vector(env, robot: Articulation, env_idx: int) -> np.ndarray:
    """
    提取状态向量，与 generate_data_traj_cont.py 的 state_obs 对应
    这里我们把所有可观测状态拼接成一个向量
    """
    # 收集各部分状态
    ang_vel = robot.data.root_ang_vel_b[env_idx].cpu().numpy()           # (3,)
    projected_gravity = robot.data.projected_gravity_b[env_idx].cpu().numpy()  # (3,)
    command = env.command_generator.command[env_idx].cpu().numpy()       # (3,)
    joint_pos = (robot.data.joint_pos[env_idx] - robot.data.default_joint_pos[env_idx]).cpu().numpy()  # (12,)
    joint_vel = robot.data.joint_vel[env_idx].cpu().numpy()              # (12,)
    last_action = env.action_buffer._circular_buffer.buffer[env_idx, -1, :].cpu().numpy()  # (12,)
    
    # 拼接成一个状态向量
    state = np.concatenate([
        ang_vel,           # 3
        projected_gravity, # 3
        command,           # 3
        joint_pos,         # 12
        joint_vel,         # 12
        last_action        # 12
    ])
    return state


def extract_priv_state_vector(env, robot: Articulation, env_idx: int) -> np.ndarray:
    """
    提取特权状态向量，与 generate_data_traj_cont.py 的 state_gt 对应
    包含仿真中才能获取的真实状态
    """
    root_lin_vel = robot.data.root_lin_vel_b[env_idx].cpu().numpy()      # (3,)
    root_pos = robot.data.root_pos_w[env_idx].cpu().numpy()              # (3,)
    root_quat = robot.data.root_quat_w[env_idx].cpu().numpy()            # (4,)
    
    # 脚部接触状态
    contact_forces = env.contact_sensor.data.net_forces_w_history[env_idx]  # (history, num_bodies, 3)
    feet_contact = (torch.max(torch.norm(contact_forces[:, :4], dim=-1), dim=0)[0] > 0.5).cpu().numpy().astype(np.float32)  # (4,)
    
    priv_state = np.concatenate([
        root_lin_vel,   # 3
        root_pos,       # 3  
        root_quat,      # 4
        feet_contact    # 4
    ])
    return priv_state


def extract_image(env, env_idx: int) -> np.ndarray:
    """
    提取图像数据
    返回单个 numpy 数组（与 generate_data_traj_cont.py 的 img_array 对应）
    """
    try:
        if hasattr(env.scene, 'sensors') and 'front_camera' in env.scene.sensors:
            camera = env.scene.sensors['front_camera']
            if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                output = camera.data.output
                
                # 优先使用 RGB
                if 'rgb' in output:
                    return output['rgb'][env_idx].cpu().numpy()
                elif 'rgba' in output:
                    rgba = output['rgba'][env_idx].cpu().numpy()
                    return rgba[..., :3]  # 只取 RGB 通道
                elif 'distance_to_image_plane' in output:
                    depth = output['distance_to_image_plane'][env_idx].cpu().numpy()
                    return depth
    except Exception as e:
        pass
    
    # 如果没有相机，返回空占位符
    return np.zeros((64, 64, 3), dtype=np.uint8)

def extract_failure(env, env_idx: int, is_done: bool, is_timeout: bool) -> int:
    """
    判断是否为失败轨迹
    
    逻辑：提前终止（非 timeout 的 reset）视为 failure
    - timeout 结束 → success (failure=0)
    - 提前 reset（如 base 碰撞）→ failure (failure=1)
    
    Args:
        env: 环境实例
        env_idx: 环境索引
        is_done: 是否结束（来自 env.step 的 dones）
        is_timeout: 是否因为达到最大步数而结束
    
    Returns:
        1 if failure, 0 otherwise
    """
    if is_done and not is_timeout:
        return 1  # 提前 reset = failure
    return 0


def collect_data():
    """主数据收集函数"""
    
    # ========== 环境设置 ==========
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False
    
    # 重要：确保相机每步都更新
    # step_dt = physics_dt * decimation = 0.005 * 4 = 0.02
    # 将 camera.update_period 设置为与 step_dt 相等，确保每步都更新相机
    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt
    print(f"[INFO] 设置相机更新周期为 {step_dt}s (与 step_dt 同步)")
    
    # 创建环境
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    
    num_envs = env.num_envs
    print(f"\n[INFO] 环境创建完成")
    print(f"   - 并行环境数: {num_envs}")
    print(f"   - 动作维度: {env.num_actions}")
    
    # ========== 加载策略 ==========
    _, policy_agent_cfg = task_registry.get_cfgs(args.policy_task)
    policy_agent_cfg = update_rsl_rl_cfg(policy_agent_cfg, args)
    
    log_root_path = os.path.abspath(os.path.join("logs", policy_agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, policy_agent_cfg.load_run, policy_agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    print(f"[INFO] 加载策略: {resume_path}")
    
    runner = OnPolicyRunner(env, policy_agent_cfg.to_dict(), log_dir=log_dir, device=policy_agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    
    print(f"[INFO] 策略加载成功!")
    
    # ========== 数据收集 ==========
    robot: Articulation = env.scene["robot"]
    
    # 每个环境一个轨迹缓冲区
    traj_buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    
    # 每个环境已完成的 episode 数
    episodes_completed = [0] * num_envs
    
    # 收集完成的所有 demos
    all_demos: List[Dict] = []
    
    # 目标：每个环境收集 num_episodes 条轨迹
    total_target = num_envs * args.num_episodes
    
    print(f"\n[INFO] 开始数据收集")
    print(f"   - 每环境 episodes: {args.num_episodes}")
    print(f"   - 总目标 episodes: {total_target}")
    print(f"   - 每 episode 最大步数: {args.max_steps}")
    
    # 初始化环境 - 需要传入所有环境的 ids
    all_env_ids = torch.arange(num_envs, device=env.device)
    env.reset(all_env_ids)
    obs_dict = env.get_observations()
    
    step_count = 0
    
    # 第一次需要先执行一个 step 来初始化传感器数据
    # 因为相机需要在 scene.update() 后才有数据
    dummy_actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    obs_dict, _, _, _ = env.step(dummy_actions)
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 检查是否收集完成
            total_collected = sum(episodes_completed)
            if total_collected >= total_target:
                print(f"\n[INFO] 数据收集完成! 共 {len(all_demos)} 条轨迹")
                break
            
            # ===== Step 1: 获取动作 (a_t) =====
            actions = policy(obs_dict)
            
            # ===== Step 2: 收集当前时刻的观测 (o_t) =====
            # 注意：在 step 之前收集观测，此时传感器数据是当前帧的
            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue  # 该环境已完成所有 episodes
                
                # 提取观测 - 图像数据在上一个 step 后已经更新
                image = extract_image(env, env_idx)
                state = extract_state_vector(env, robot, env_idx)
                priv_state = extract_priv_state_vector(env, robot, env_idx)
                
                # 暂存观测（还没有 done）
                traj_buffers[env_idx]._pending_obs = (image.copy(), state.copy(), priv_state.copy())
                traj_buffers[env_idx]._pending_action = actions[env_idx].cpu().numpy().copy()
            
            # ===== Step 3: 执行动作，获取 done 信号 =====
            next_obs_dict, rewards, dones, extras = env.step(actions)
            
            # ===== Step 4: 记录完整的 (o_t, a_t, done_t, failure_t) =====
            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue
                
                buffer = traj_buffers[env_idx]
                image, state, priv_state = buffer._pending_obs
                action = buffer._pending_action
                
                # 判断 done
                is_timeout = buffer.step_count >= args.max_steps - 1
                # dones来自于 base_env.step里的reset_buf
                is_env_done = dones[env_idx].item() if isinstance(dones[env_idx], torch.Tensor) else dones[env_idx]
                is_done = is_timeout or is_env_done
                
                # 判断 failure：提前 reset（非 timeout）视为 failure
                failure = extract_failure(env, env_idx, is_done, is_timeout)
                
                # 记录数据
                buffer.add_step(
                    image=image,
                    state=state,
                    priv_state=priv_state,
                    action=action,
                    done=1 if is_done else 0,
                    failure=failure
                )
                
                # 如果 episode 结束
                if is_done:
                    # 保存轨迹
                    demo = buffer.to_demo(env_id=env_idx)
                    all_demos.append(demo)
                    episodes_completed[env_idx] += 1
                    
                    # 显示是否为 failure 轨迹
                    status = "FAIL" if failure else "OK"
                    print(f"   环境 {env_idx} | Episode {episodes_completed[env_idx]}/{args.num_episodes} | "
                          f"步数: {len(buffer)} | {status} | 总进度: {sum(episodes_completed)}/{total_target}")
                    
                    # 重置缓冲区
                    buffer.reset()
                    
                    # 注意：Isaac Lab 会自动 reset done 的环境
            
            obs_dict = next_obs_dict
            step_count += 1
            
            # 进度显示
            if step_count % 200 == 0:
                print(f"   Step {step_count} | 已收集: {sum(episodes_completed)}/{total_target} episodes")
    
    # ========== 保存数据 ==========
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"go2_demos_{len(all_demos)}ep.pkl"
    
    print(f"\n[INFO] 保存数据到: {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_demos, f, protocol=4)
    
    # 验证
    with open(output_file, 'rb') as f:
        loaded = pickle.load(f)
    
    print(f"[INFO] 数据保存成功!")
    print(f"   - 总 episodes: {len(loaded)}")
    print(f"   - 文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 打印数据结构示例
    if loaded:
        sample = loaded[0]
        print(f"\n[INFO] 数据结构示例 (第一条轨迹):")
        print(f"   - 轨迹长度: {len(sample['actions'])}")
        print(f"   - state shape: {sample['obs']['state'][0].shape}")
        print(f"   - priv_state shape: {sample['obs']['priv_state'][0].shape}")
        print(f"   - action shape: {sample['actions'][0].shape}")
        print(f"   - image shape: {sample['obs']['image'][0].shape}")
        print(f"   - dones: [..., {sample['dones'][-3:]}")
        print(f"   - failure: [..., {sample['failure'][-3:]}")


if __name__ == "__main__":
    collect_data()
    simulation_app.close()
