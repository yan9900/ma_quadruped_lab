#!/usr/bin/env python3

"""Simple debug script for reward calculation"""

import torch
from isaaclab.app import AppLauncher

# Initialize app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

def debug_rewards():
    # Import configurations directly
    from legged_lab.envs.go2.go2_config import Go2FallRecoveryFlatEnvCfg
    from legged_lab.envs.base.base_env import BaseEnv
    
    # Create environment directly with correct parameters
    env_cfg = Go2FallRecoveryFlatEnvCfg()
    env_cfg.scene.num_envs = 4  # Set number of environments
    env = BaseEnv(cfg=env_cfg, headless=True)
    
    print(f"Environment created with {env.num_envs} environments")
    print(f"Robot available: {hasattr(env, 'robot')}")
    
    if hasattr(env, 'robot'):
        print(f"Robot data available: {hasattr(env.robot, 'data')}")
        if hasattr(env.robot.data, 'projected_gravity_b'):
            print(f"Robot gravity shape: {env.robot.data.projected_gravity_b.shape}")
            print(f"First env gravity: {env.robot.data.projected_gravity_b[0]}")
    
    # Test reward functions directly
    print(f"\nReward manager available: {hasattr(env, 'reward_manager')}")
    if hasattr(env, 'reward_manager'):
        print(f"Active reward terms: {list(env.reward_manager.active_terms.keys())}")
        
        # Test step to compute rewards
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        
        # First simulation step to initialize everything
        env.sim.step()
        env.scene.update(env.physics_dt)
        
        # Now compute rewards
        total_rewards = env.reward_manager.compute(env.physics_dt)
        print(f"\nTotal reward shape: {total_rewards.shape}")
        print(f"Total reward values: {total_rewards}")
        
        # Test individual reward terms
        for term_name, term in env.reward_manager.active_terms.items():
            try:
                value = term.function(env, **term.params)
                print(f"  {term_name}: mean={value.mean().item():.6f}, weight={term.weight}")
            except Exception as e:
                print(f"  {term_name}: ERROR - {e}")
    
    env.close()

if __name__ == "__main__":
    debug_rewards()
    simulation_app.close()
