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

import argparse
import os

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 启动Omniverse应用（仿真环境）
# 用来控制主循环
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOTE: isaaclab_rl export functions not available in rsl_rl 3.0.0
# from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # 针对Fall Recovery的特殊配置
    if "fall_recovery" in env_class_name.lower():
        print("[INFO] Configuring for Fall Recovery task...")
        # 禁用噪声以便观察纯策略表现
        env_cfg.noise.add_noise = False
        # 禁用push robot事件，但保持reset_base以确保随机初始姿态
        env_cfg.domain_rand.events.push_robot = None
        # 确保保持Fall Recovery的随机初始姿态设置
        # env_cfg.domain_rand.events.reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0),
        #         # "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)
        #         "roll": (-0.0, -0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14)

        #     },
        #     "velocity_range": {
        #         "x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0),
        #         "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)
        #     }
        # }

        
        # 保持Fall Recovery的长episode时间
        env_cfg.scene.max_episode_length_s = 5.0
        # 较少环境数量用于演示观察
        env_cfg.scene.num_envs = 10
        env_cfg.scene.env_spacing = 3.0  # 增大间距以便观察
        
        # env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        # env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)  #躺下
        
 
        
        print("[INFO] Fall Recovery: Keeping random initial poses for recovery demonstration")
        print("[INFO] Robots will start from various fallen/inverted positions")
        
        # 速度命令设为0，专注于恢复站立而非移动
        env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
        
        # 保持平地地形以专注于恢复行为
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"
        
    else:
        # 原有的常规配置
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.events.push_robot = None
        env_cfg.scene.max_episode_length_s = 10.0
        env_cfg.scene.num_envs = 9 # num of robots
        env_cfg.scene.env_spacing = 2.5
        env_cfg.commands.ranges.lin_vel_x = (0.6, 1.2)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
        env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
        env_cfg.commands.rel_standing_envs = 0.0  # all envs have moving commands

        # env_cfg.scene.terrain_generator = None
        # env_cfg.scene.terrain_type = "plane"

        if env_cfg.scene.terrain_generator is not None:
            env_cfg.scene.terrain_generator.num_rows = 3
            env_cfg.scene.terrain_generator.num_cols = 3
            env_cfg.scene.terrain_generator.curriculum = True
            env_cfg.scene.terrain_generator.difficulty_range = (0.1, 0.3)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    # Note: rsl_rl 3.0.0 does not support export_policy_as_jit/onnx
    # Direct policy inference is used instead
    print(f"[INFO] Policy loaded successfully. Using direct inference mode.")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    # 获取初始观测
    obs_dict = env.get_observations()
    # 使用policy观测作为策略输入
    obs = obs_dict["policy"]

    # 为Fall Recovery任务添加统计信息和特殊处理
    if "fall_recovery" in env_class_name.lower():
        print(f"\n[INFO] Fall Recovery Play Mode Started")
        print(f"Number of environments: {env_cfg.scene.num_envs}")
        print(f"Episode length: {env_cfg.scene.max_episode_length_s}s")
        print(f"Command velocities: lin_x={env_cfg.commands.ranges.lin_vel_x}, lin_y={env_cfg.commands.ranges.lin_vel_y}")
        print(f"Terrain: {env_cfg.scene.terrain_type}")
        print("Watching for recovery behaviors from random initial poses...\n")
        
        step_count = 0
        reset_interval = 1000  # 重置间隔

    # 确保go2从一个静止的状态开始
    # define wait time
    dt = env_cfg.sim.dt
    wait_time = int(0.5 / dt)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Fall Recovery特殊处理：定期重置并输出状态信息
            if "fall_recovery" in env_class_name.lower():
                # if step_count < wait_time:
                #     actions = torch.zeros_like(policy(obs_dict))
                # elif step_count % reset_interval == 0:
                #     # 重置所有环境
                #     env_ids = torch.arange(env.num_envs, device=env.device)
                #     env.reset(env_ids)
                    
                #     obs_dict = env.get_observations()
                #     obs = obs_dict["policy"]
                #     step_count = 0
                    
                # else:
                #     actions = policy(obs_dict)
                actions = policy(obs_dict)
                
                # step_count += 1
            
            else:
                actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
