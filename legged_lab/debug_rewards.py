#!/usr/bin/env python3

"""Debug script for reward calculation"""

import torch
from legged_lab.utils.task_registry import task_registry
from isaaclab.app import AppLauncher

# Initialize app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

def debug_rewards():
    # Get task configuration
    task_name = "go2_fall_recovery"
    env_cfg, agent_cfg = task_registry.get_cfgs(task_name)
    
    # Import the environment class
    env_cls = task_registry.get_task_class(task_name)
    env = env_cls(cfg=env_cfg, num_envs=4, device="cuda:0")
    
    print(f"\nEnvironment created with {env.num_envs} environments")
    print(f"Robot data available: {hasattr(env.robot, 'data')}")
    
    if hasattr(env.robot, 'data'):
        print(f"Robot gravity: {env.robot.data.projected_gravity_b.shape}")
        print(f"First env gravity: {env.robot.data.projected_gravity_b[0]}")
    
    # Test step
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, reward, done, info = env.step(actions)
    
    print(f"\nAfter step:")
    print(f"Total reward shape: {reward.shape}")
    print(f"Total reward values: {reward}")
    
    # Check individual rewards
    if hasattr(env, 'reward_manager'):
        print("\nIndividual reward terms:")
        reward_terms = env.reward_manager.active_terms
        for name in reward_terms:
            term = reward_terms[name]
            value = term.function(env, **term.params)
            print(f"  {name}: {value.mean().item():.6f} (weight: {term.weight})")
    
    env.close()

if __name__ == "__main__":
    debug_rewards()
    simulation_app.close()
