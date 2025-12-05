#!/usr/bin/env python3

"""
Go2数据收集脚本 - 使用训练好的策略收集轨迹数据
基于play的仿真启动方式，结合generate_data_traj_cont的数据格式

使用示例:
1. 从go2_flat任务加载策略:
   python scripts/collect_go2_data.py --task go2_data_collection --policy_task go2_flat --num_episodes 10

2. 从其他任务加载策略:
   python scripts/collect_go2_data.py --task go2_data_collection --policy_task go2_rough --num_episodes 5

3. 自定义输出目录:
   python scripts/collect_go2_data.py --task go2_data_collection --policy_task go2_flat --output_dir ./data/custom --num_episodes 10

注意: 
- go2_data_collection 是数据收集环境配置
- go2_flat/go2_rough 等是训练任务，用于加载策略权重
- logs/ 目录中必须存在对应任务的训练检查点
"""

import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
import os
import sys
from pathlib import Path
# 添加路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments (在导入时执行，类似play.py)
parser = argparse.ArgumentParser(description="Go2 data collection script")
parser.add_argument("--task", type=str, default="go2_data_collection", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# 数据收集特定参数
parser.add_argument("--num_episodes", type=int, default=5, help="收集的回合数")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
parser.add_argument("--output_dir", type=str, default="./data/go2_demo", help="Output directory")
parser.add_argument("--policy_task", type=str, default="go2_flat", 
                   help="Task name to load policy from (e.g., go2_flat, go2_rough)")
parser.add_argument("--rotate_env_indices", action="store_true", help="让机器人在每个episode中spawn在不同的env_idx对应的地形位置")

# 类似play.py添加RSL-RL CLI参数
import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)

# AppLauncher参数
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动Isaac Sim (在导入时执行，类似play.py)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入必要模块 (类似play.py顺序)
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.assets.articulation import Articulation

# 导入任务注册和相关模块
from legged_lab.envs import *  # noqa:F401, F403  
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def collect_data():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg
    robot: Articulation
    """主数据收集函数，类似play.py的play()函数"""
    
    # 调试：打印可用的任务
    print(f"可用任务: {list(task_registry.task_classes.keys())}")
    
    # 获取任务配置 (类似play.py的方式)
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    
    # 应用命令行参数覆盖
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args)
    # 对环境的特殊设置
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False
    # env_cfg.scene.max_episode_length_s = args.max_steps * env_cfg.sim.dt
    

    # 确保相机启用
    if hasattr(env_cfg.scene, 'camera'):
        env_cfg.scene.camera.enable_camera = True
        env_cfg.scene.camera.debug_vis = True
        print(f"相机配置: enable_camera={env_cfg.scene.camera.enable_camera}")
    
    # 创建环境 (类似play.py的方式)
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    print(f"创建环境: {args.task}")
    print(f"环境数量: {args.num_envs}")
    print(f"动作维度: {env.num_actions}")
    
    
    
    # 加载训练好的策略 (类似play.py的方式)
    print("[INFO] Loading trained policy...")
    print(f"[INFO] Loading policy from task: {args.policy_task}")
    
    try:
        # 获取策略任务的agent配置
        _, policy_agent_cfg = task_registry.get_cfgs(args.policy_task)
        policy_agent_cfg = update_rsl_rl_cfg(policy_agent_cfg, args)
        
        log_root_path = os.path.join("logs", policy_agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from: {log_root_path}")
        
        resume_path = get_checkpoint_path(log_root_path, policy_agent_cfg.load_run, policy_agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        
        print(f"[INFO] Checkpoint path: {resume_path}")

        runner = OnPolicyRunner(env, policy_agent_cfg.to_dict(), log_dir=log_dir, device=policy_agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        print(f"[INFO] Policy loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] 策略加载失败: {e}")
        raise
    
    # 数据收集器类
    # 一个traj - 一个episode
    # total traj = num_envs * num_episodes
    # traj的思路是，通过method生成一条traj，再循环num_episodes次
    # go2可以多个环境并行
    # 一条traj包含:
    # obs: {'image': [], 'state': [], 'priv_state': []}
    # actions: []
    # dones: []
    class Go2DataCollector:
        def __init__(self, env: BaseEnv, policy=None):
            self.env = env
            self.policy = policy
            self.num_envs = env.num_envs
            self.collected_demos: List[Dict[str, Any]] = []
            
            # 为每个环境维护独立的数据收集状态
            self.env_episode_data = []
            self.env_episodes_collected = []
            self.env_current_step = []
            self.env_episode_complete = []
            
            for i in range(self.num_envs):
                self.env_episode_data.append({
                    'image_list': [],
                    'state_list': [],
                    'priv_state_list': [],
                    'action_list': [],
                    'done_list': []
                })
                self.env_episodes_collected.append(0)
                self.env_current_step.append(0)
                self.env_episode_complete.append(False)
        
        # reset的时候需要重置哪些东西？
        # go2回到初始状态，容器清空，step计数归零    
        def reset_episode(self, env_id=None):
            """重置episode数据收集"""
            if env_id is None:
                # 重置所有环境
                env_ids_to_reset = []
                for i in range(self.num_envs):
                    if self.env_episodes_collected[i] < args.num_episodes:
                        self.env_episode_data[i] = {
                            'image_list': [],
                            'state_list': [],
                            'priv_state_list': [],
                            'action_list': [],
                            'done_list': []
                        }
                        self.env_current_step[i] = 0
                        self.env_episode_complete[i] = False
                        env_ids_to_reset.append(i)
                
                if env_ids_to_reset:
                    env_ids_tensor = torch.tensor(env_ids_to_reset, device=self.env.device)
                    
                    # env_idx轮换（如果启用）
                    if args.rotate_env_indices:
                        self._rotate_env_terrain_assignment(env_ids_tensor)
                    
                    self.env.reset(env_ids_tensor)
            else:
                # 重置指定环境
                if self.env_episodes_collected[env_id] < args.num_episodes:
                    self.env_episode_data[env_id] = {
                        'image_list': [],
                        'state_list': [],
                        'priv_state_list': [],
                        'action_list': [],
                        'done_list': []
                    }
                    self.env_current_step[env_id] = 0
                    self.env_episode_complete[env_id] = False
                    
                    env_ids = torch.tensor([env_id], device=self.env.device)
                    self.env.reset(env_ids)
        
        def collect_current_observations(self) -> Dict[str, Any]:
            """收集当前时刻的观测数据，格式与你提供的obs结构一致"""
            
            # 获取机器人数据
            robot: Articulation = self.env.scene["robot"]
            
            # 获取相机数据
            camera_data = {}
            try:
                # 优先通过scene.sensors访问 (这是Isaac Lab的标准方式)
                if hasattr(self.env.scene, 'sensors') and 'front_camera' in self.env.scene.sensors:
                    camera = self.env.scene.sensors['front_camera']
                    # print(f"[DEBUG] Found camera via scene.sensors: {type(camera)}")
                    if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                        output_data = camera.data.output
                        # print(f"[DEBUG] Camera output keys: {list(output_data.keys())}")
                        
                        # 深度图像
                        if 'distance_to_image_plane' in output_data:
                            camera_data['depth'] = output_data['distance_to_image_plane'].clone()
                        
                        # RGB图像
                        if 'rgb' in output_data:
                            camera_data['rgb'] = output_data['rgb'].clone()
                        
                        if 'rgba' in output_data:
                            camera_data['rgba'] = output_data['rgba'].clone()
                    else:
                        print(f"[DEBUG] Camera data structure: data={hasattr(camera, 'data')}, output={hasattr(camera.data, 'output') if hasattr(camera, 'data') else 'N/A'}")
                        
                # 备选方案：通过env直接访问
                elif hasattr(self.env, 'front_camera'):
                    camera = self.env.front_camera
                    print(f"[DEBUG] Found camera via env: {type(camera)}")
                    if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                        output_data = camera.data.output
                        print(f"[DEBUG] Camera output keys: {list(output_data.keys())}")
                        
                        # 深度图像
                        if 'distance_to_image_plane' in output_data:
                            camera_data['depth'] = output_data['distance_to_image_plane'].clone()
                        
                        # RGB图像
                        if 'rgb' in output_data:
                            camera_data['rgb'] = output_data['rgb'].clone()
                        
                        if 'rgba' in output_data:
                            camera_data['rgba'] = output_data['rgba'].clone()
                else:
                    print(f"[DEBUG] Camera not found. Available sensors: {list(self.env.scene.sensors.keys()) if hasattr(self.env.scene, 'sensors') else 'No sensors'}")
                    print(f"[DEBUG] Env attributes: {[attr for attr in dir(self.env) if 'camera' in attr.lower()]}")
                            
            except Exception as e:
                print(f"[WARNING] 获取摄像头数据失败: {e}")
            
            # 构建观测数据（与你提供的格式一致）
            observations = {
                'image': camera_data,  # 摄像头数据
                'state': {
                    'ang_vel': robot.data.root_ang_vel_b.clone(),
                    'projected_gravity': robot.data.projected_gravity_b.clone(),
                    'command': self.env.command_generator.command.clone(),  # 直接从command_generator获取
                    'joint_pos': (robot.data.joint_pos - robot.data.default_joint_pos).clone(),
                    'joint_vel': (robot.data.joint_vel - robot.data.default_joint_vel).clone(),
                    'action': self.env.action_buffer._circular_buffer.buffer[:, -1, :].clone(),
                },
                'priv_state': {
                    'root_lin_vel_b': robot.data.root_lin_vel_b.clone(),
                    'feet_contact': torch.max(torch.norm(self.env.contact_sensor.data.net_forces_w_history[:, :, :4], dim=-1), dim=1)[0] > 0.5,  # 假设前4个body是脚
                }
            }
            
            return observations
        
        def get_actions(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            """获取动作 - 使用训练好的策略网络生成智能动作"""
            with torch.no_grad():
                actions = self.policy(obs_dict)
            return actions
        
        def process_step(self, obs_dict, actions):
            """处理单个仿真步骤的数据收集 - 现在收集所有环境的数据"""
            # 收集当前观测
            current_obs = self.collect_current_observations()
            
            # 执行动作
            next_obs_dict, rewards, dones, extras = self.env.step(actions)
            
            # 为每个环境收集数据
            any_episode_complete = False
            
            for env_idx in range(self.num_envs):
                # 跳过已完成所有episodes的环境
                if self.env_episodes_collected[env_idx] >= args.num_episodes:
                    continue
                
                # 跳过已完成当前episode的环境
                if self.env_episode_complete[env_idx]:
                    continue
                
                # 处理state数据
                state_dict = {}
                for k, v in current_obs['state'].items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v[env_idx].cpu().numpy()
                    else:
                        state_dict[k] = v[env_idx] if hasattr(v, '__getitem__') else v
                
                # 处理priv_state数据
                priv_state_dict = {}
                for k, v in current_obs['priv_state'].items():
                    if isinstance(v, torch.Tensor):
                        priv_state_dict[k] = v[env_idx].cpu().numpy()
                    else:
                        priv_state_dict[k] = v[env_idx] if hasattr(v, '__getitem__') else v
                
                # 处理image数据
                image_dict = {}
                for k, v in current_obs['image'].items():
                    if isinstance(v, torch.Tensor):
                        image_dict[k] = v[env_idx].cpu().numpy()
                    else:
                        image_dict[k] = v
                
                # 添加到当前环境的episode数据
                self.env_episode_data[env_idx]['image_list'].append(image_dict)
                self.env_episode_data[env_idx]['state_list'].append(state_dict)
                self.env_episode_data[env_idx]['priv_state_list'].append(priv_state_dict)
                self.env_episode_data[env_idx]['action_list'].append(actions[env_idx].cpu().numpy())
                
                # 检查是否完成
                is_done = dones[env_idx].cpu().numpy() if isinstance(dones[env_idx], torch.Tensor) else dones[env_idx]
                self.env_episode_data[env_idx]['done_list'].append(int(is_done))
                
                self.env_current_step[env_idx] += 1
                
                # 检查episode是否结束
                if is_done or self.env_current_step[env_idx] >= args.max_steps:
                    self.complete_env_episode(env_idx)
                    any_episode_complete = True
                    
                    # 如果该环境还需要更多episodes，立即重置
                    if self.env_episodes_collected[env_idx] < args.num_episodes:
                        self.reset_episode(env_idx)
            
            return next_obs_dict, any_episode_complete
        
        def complete_env_episode(self, env_idx):
            """完成指定环境的当前episode并保存到collected_demos"""
            episode_data = self.env_episode_data[env_idx]
            
            episode = {
                'obs': {
                    'image': episode_data['image_list'].copy(),
                    'state': episode_data['state_list'].copy(),
                    'priv_state': episode_data['priv_state_list'].copy()
                },
                'actions': episode_data['action_list'].copy(),
                'dones': episode_data['done_list'].copy(),
                'env_id': env_idx  # 标记来自哪个环境
            }
            
            self.collected_demos.append(episode)
            self.env_episodes_collected[env_idx] += 1
            self.env_episode_complete[env_idx] = True
            
            total_collected = sum(self.env_episodes_collected)
            total_target = args.num_episodes * self.num_envs
            
            print(f"环境{env_idx} Episode {self.env_episodes_collected[env_idx]} 完成: {len(episode['actions'])} 步 | 总进度: {total_collected}/{total_target}")
        
        def should_continue_collecting(self):
            """检查是否应该继续收集数据"""
            total_collected = sum(self.env_episodes_collected)
            total_target = args.num_episodes * self.num_envs
            return total_collected < total_target
        def save_data(self, output_file: str):
            """保存收集的数据，格式与generate_data_traj_cont.py一致"""
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为与generate_data_traj_cont.py相同的格式
            with open(output_file, 'wb') as f:
                pickle.dump(self.collected_demos, f)
            
            print(f"数据已保存到: {output_file}")
            print(f"   - 总episodes: {len(self.collected_demos)}")
            print(f"   - 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

            print(f"正在保存数据到: {output_file}")
            print(f"   - 总episodes: {len(self.collected_demos)}")
            
            # 使用更安全的保存方式
            try:
                # 先保存到临时文件
                temp_file = str(output_path) + '.tmp'
                print(f"   - 写入临时文件...")
                with open(temp_file, 'wb') as f:
                    # 使用 protocol 4 以支持大文件
                    pickle.dump(self.collected_demos, f, protocol=4)
                
                # 验证临时文件
                print(f"   - 验证文件完整性...")
                with open(temp_file, 'rb') as f:
                    test_load = pickle.load(f)
                    assert len(test_load) == len(self.collected_demos), "数据长度不匹配!"
                
                # 重命名为最终文件
                import shutil
                shutil.move(temp_file, output_file)
                
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"✓ 数据已成功保存!")
                print(f"   - 文件大小: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
                
            except Exception as e:
                print(f"✗ 保存失败: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
        def print_data_summary(self):
            """打印数据摘要"""
            if not self.collected_demos:
                print("没有收集到数据")
                return
            
            print("\n=== 并行数据收集摘要 ===")
            print(f"   - 总episodes: {len(self.collected_demos)}")
            print(f"   - 使用环境数: {self.num_envs}")
            
            # 统计每个环境的贡献
            env_contributions = {}
            for demo in self.collected_demos:
                env_id = demo.get('env_id', 0)
                env_contributions[env_id] = env_contributions.get(env_id, 0) + 1
            
            print(f"   - 各环境episodes: {dict(sorted(env_contributions.items()))}")
            
            # 统计步数
            total_steps = sum([len(ep['actions']) for ep in self.collected_demos])
            avg_steps = total_steps / len(self.collected_demos) if self.collected_demos else 0
            print(f"   - 总步数: {total_steps}")
            print(f"   - 平均步数/episode: {avg_steps:.1f}")
            
            # 检查数据结构
            if self.collected_demos:
                sample_ep = self.collected_demos[0]
                print(f"\n   单个episode结构:")
                print(f"   - 轨迹长度: {len(sample_ep['actions'])}")
                print(f"   - 动作维度: {sample_ep['actions'][0].shape}")
                print(f"   - 状态观测键: {list(sample_ep['obs']['state'][0].keys())}")
                print(f"   - 特权状态键: {list(sample_ep['obs']['priv_state'][0].keys())}")
                
                # 检查图像数据
                if sample_ep['obs']['image'] and sample_ep['obs']['image'][0]:
                    print(f"   - 图像数据键: {list(sample_ep['obs']['image'][0].keys())}")
                    for img_type, img_data in sample_ep['obs']['image'][0].items():
                        if isinstance(img_data, np.ndarray):
                            print(f"     - {img_type}: 形状{img_data.shape}, 类型{img_data.dtype}")

    # 创建数据收集器
    collector = Go2DataCollector(env, policy)
    
    # 获取初始观测
    obs_dict = env.get_observations()
    
    print(f"\n开始并行收集数据...")
    print(f"   - 环境数: {env.num_envs}")
    print(f"   - 每环境episodes: {args.num_episodes}")
    print(f"   - 总episodes: {args.num_episodes * env.num_envs}")
    print(f"   - 每episode最多步数: {args.max_steps}")
    print(f"   - 使用策略: {args.policy_task}")
    
    # 初始化所有环境
    collector.reset_episode()
    
    # 主收集循环
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 检查是否还需要收集数据
            if not collector.should_continue_collecting():
                print("所有数据收集完成!")
                break
            
            # 生成所有环境的动作
            actions = collector.get_actions(obs_dict)
            
            # 并行处理所有环境的数据收集
            obs_dict, any_episode_done = collector.process_step(obs_dict, actions)
            
            step_count += 1
            
            # 每100步显示一次进度
            if step_count % 100 == 0:
                total_collected = sum(collector.env_episodes_collected)
                total_target = args.num_episodes * collector.num_envs
                print(f"Step {step_count} | 已收集episodes: {total_collected}/{total_target}")
    
    # 打印数据摘要
    collector.print_data_summary()
    
    # 保存数据
    total_episodes = len(collector.collected_demos)
    output_file = os.path.join(args.output_dir, f"go2_parallel_{total_episodes}ep_{env.num_envs}envs.pkl")
    collector.save_data(output_file)
    
    print(f"\n数据收集完成! 文件保存在: {output_file}")


if __name__ == "__main__":
    collect_data()
    simulation_app.close()