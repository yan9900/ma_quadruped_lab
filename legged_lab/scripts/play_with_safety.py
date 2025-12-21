#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

"""
带安全过滤的Go2评估脚本

功能：
1. 加载训练好的PPO策略
2. 加载世界模型作为Safety Monitor
3. 实时过滤危险动作
4. 记录并输出安全统计信息

使用方法:
    # 基本用法 (使用默认世界模型)
    python scripts/play_with_safety.py --task go2_flat --enable_cameras
    
    # 指定世界模型路径
    python scripts/play_with_safety.py --task go2_flat --enable_cameras \
        --wm_path /path/to/rssm_ckpt.pt
    
    # 调整安全阈值
    python scripts/play_with_safety.py --task go2_flat --enable_cameras \
        --safety_threshold -0.2
    
    # 禁用安全过滤（仅监控）
    python scripts/play_with_safety.py --task go2_flat --enable_cameras \
        --monitor_only
    
    # 保存统计结果
    python scripts/play_with_safety.py --task go2_flat --enable_cameras \
        --save_stats ./results/safety_eval.json
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description="Play with Safety Monitor")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# Safety Monitor 参数
parser.add_argument("--wm_path", type=str, 
                    default="/home/lcy/latent-safety/logs/go2_world_model_v2/rssm_ckpt.pt",
                    help="Path to world model checkpoint")
parser.add_argument("--wm_config", type=str,
                    default="/home/lcy/latent-safety/configs.yaml",
                    help="Path to world model config")
parser.add_argument("--safety_threshold", type=float, default=0.0,
                    help="Safety threshold, g(x) < threshold triggers filtering")
parser.add_argument("--fallback_strategy", type=str, default="stop",
                    choices=["stop", "previous", "zero_velocity"],
                    help="Fallback action strategy when danger detected")
parser.add_argument("--monitor_only", action="store_true",
                    help="Only monitor safety values, don't filter actions")
parser.add_argument("--save_stats", type=str, default=None,
                    help="Path to save safety statistics JSON")
parser.add_argument("--print_interval", type=int, default=100,
                    help="Interval (steps) to print safety statistics")

# RSL-RL 参数
cli_args.add_rsl_rl_args(parser)
# AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path
from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play_with_safety():
    """主评估函数"""
    
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    
    # ==================== 环境配置 ====================
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 10.0  # 延长episode以观察安全行为
    env_cfg.scene.num_envs = 9
    env_cfg.scene.env_spacing = 10
    
    # 确保相机启用和完整配置
    if hasattr(env_cfg.scene, 'camera'):
        env_cfg.scene.camera.enable_camera = True
        env_cfg.scene.camera.use_physical_asset = True
        env_cfg.scene.camera.prim_body_name = "base"
        env_cfg.scene.camera.height = 64
        env_cfg.scene.camera.width = 64
        env_cfg.scene.camera.history_length = 2
        env_cfg.scene.camera.update_period = 0.025
        env_cfg.scene.camera.debug_vis = True
        env_cfg.scene.camera.data_types = ["distance_to_image_plane"]
        print(f"[INFO] Camera configured: enabled={env_cfg.scene.camera.enable_camera}, size={env_cfg.scene.camera.width}x{env_cfg.scene.camera.height}")
    else:
        print("[WARNING] No camera configuration found in scene!")
    
    # 命令配置
    env_cfg.commands.ranges.lin_vel_x = (0.5, 1.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.commands.rel_standing_envs = 0.0
    
    # 使用悬崖检测地形
    from legged_lab.terrains import CLIFF_DETECTION_TERRAINS_CFG
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = CLIFF_DETECTION_TERRAINS_CFG
    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 3
        env_cfg.scene.terrain_generator.num_cols = 3
    
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    # 创建环境
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    
    print(f"\n{'='*60}")
    print(f"Environment: {env_class_name}")
    print(f"Num envs: {env.num_envs}")
    print(f"Device: {env.device}")
    print(f"{'='*60}\n")

    # ==================== 加载策略 ====================
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[INFO] Policy loaded successfully from: {resume_path}")

    # ==================== 加载 Safety Monitor ====================
    print(f"\n{'='*60}")
    print("Loading Safety Monitor...")
    print(f"{'='*60}")
    
    from safety_monitor import Go2SafetyMonitor
    
    safety_monitor = Go2SafetyMonitor(
        world_model_path=args_cli.wm_path,
        config_path=args_cli.wm_config,
        threshold=args_cli.safety_threshold,
        fallback_strategy=args_cli.fallback_strategy,
        device=str(env.device),
        image_key="depth",
        state_key="policy",
        env=env  # 传入环境对象以直接获取相机数据
    )
    
    print(f"[INFO] Safety Monitor loaded:")
    print(f"  - World model: {args_cli.wm_path}")
    print(f"  - Threshold: {args_cli.safety_threshold}")
    print(f"  - Fallback strategy: {args_cli.fallback_strategy}")
    print(f"  - Monitor only: {args_cli.monitor_only}")
    print(f"{'='*60}\n")

    # ==================== 键盘控制 ====================
    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)

    # ==================== 评估循环 ====================
    obs_dict = env.get_observations()
    step_count = 0
    episode_count = 0
    
    # 记录每个环境的摔倒情况
    env_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    print("\n[INFO] Starting evaluation with safety monitoring...")
    print("[INFO] Press Ctrl+C to stop and show statistics.\n")
    
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # 1. 策略生成原始动作
                actions_raw = policy(obs_dict)
                
                # 2. 安全检查/过滤
                if args_cli.monitor_only:
                    # 仅监控模式：获取安全值但不过滤
                    g_values = safety_monitor.get_safety_values(obs_dict)
                    actions = actions_raw
                    was_filtered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                else:
                    # 过滤模式：危险时使用后备动作
                    actions, was_filtered, g_values = safety_monitor.filter_action(
                        obs_dict, actions_raw
                    )
                
                # 3. 执行动作
                obs_dict, rewards, dones, infos = env.step(actions)
                
                # 4. 更新统计
                step_count += 1
                
                # 检查摔倒 (根据实际环境的done条件)
                if dones.any():
                    episode_count += dones.sum().item()
                    safety_monitor.reset()  # 重置Safety Monitor状态
                
                # 5. 定期打印状态
                if step_count % args_cli.print_interval == 0:
                    g_mean = g_values.mean().item()
                    g_min = g_values.min().item()
                    filter_count = was_filtered.sum().item()
                    
                    print(f"[Step {step_count:6d}] "
                          f"g(x): mean={g_mean:+.3f}, min={g_min:+.3f} | "
                          f"Filtered: {filter_count}/{env.num_envs} | "
                          f"Episodes: {episode_count}")
                    
                    # 如果有环境触发过滤，打印详细信息
                    if filter_count > 0:
                        filtered_envs = torch.where(was_filtered)[0].tolist()
                        filtered_g = g_values[was_filtered].tolist()
                        print(f"  └─ Filtered envs: {filtered_envs}, g(x): {[f'{g:.3f}' for g in filtered_g]}")
                
    except KeyboardInterrupt:
        print("\n\n[INFO] Evaluation interrupted by user.")
    
    # ==================== 输出统计结果 ====================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    safety_monitor.print_stats()
    
    # 保存统计结果
    if args_cli.save_stats:
        stats = safety_monitor.get_stats_summary()
        stats['config'] = {
            'task': args_cli.task,
            'num_envs': env.num_envs,
            'wm_path': args_cli.wm_path,
            'threshold': args_cli.safety_threshold,
            'fallback_strategy': args_cli.fallback_strategy,
            'monitor_only': args_cli.monitor_only,
            'total_steps': step_count,
            'total_episodes': episode_count,
        }
        
        save_path = Path(args_cli.save_stats)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n[INFO] Statistics saved to: {save_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    play_with_safety()
    simulation_app.close()
