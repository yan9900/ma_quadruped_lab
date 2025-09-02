#!/usr/bin/env python3
"""
测试Fall Recovery环境配置的简单脚本
"""

import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    """Test the fall recovery environment setup."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Fall Recovery Environment")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--task", type=str, default="go2_fall_recovery", help="Task name")
    # Note: --headless is automatically added by AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    # Launch the simulator
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    # Import after launching
    from utils.task_registry import task_registry
    from legged_lab.envs.go2.go2_config import Go2FallRecoveryFlatEnvCfg, Go2FallRecoveryAgentCfg
    
    print(f"Testing task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    
    # Create environment
    env_cfg, agent_cfg = task_registry.get_cfgs(name=args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    
    # Print some key configuration values
    print("\n=== Fall Recovery Configuration ===")
    print(f"Max episode length: {env_cfg.scene.max_episode_length_s}s")
    print(f"Terminate contacts: {env_cfg.robot.terminate_contacts_body_names}")
    
    # Check reward weights
    print(f"\nKey reward weights:")
    print(f"  track_lin_vel_xy_exp: {env_cfg.reward.track_lin_vel_xy_exp.weight}")
    print(f"  track_ang_vel_z_exp: {env_cfg.reward.track_ang_vel_z_exp.weight}")  
    print(f"  upward: {env_cfg.reward.upward.weight}")
    print(f"  flat_orientation_l2: {env_cfg.reward.flat_orientation_l2.weight}")
    print(f"  body_orientation_l2: {env_cfg.reward.body_orientation_l2.weight}")
    print(f"  is_terminated: {env_cfg.reward.is_terminated.weight}")
    
    # Try to create the environment
    try:
        from legged_lab.envs.base.base_env import BaseEnv
        env = BaseEnv(cfg=env_cfg)
        print(f"\n✅ Environment created successfully!")
        print(f"   - Number of environments: {env.num_envs}")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Observation space: {env.observation_space}")
        
        # Run a few steps to test
        print("\n🧪 Running test steps...")
        env.reset()
        for i in range(5):
            actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(f"  Step {i+1}: reward_mean={rewards.mean().item():.3f}")
            
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
