#!/usr/bin/env python3

"""Check robot initial state and reward calculation"""

import torch
from isaaclab.app import AppLauncher
import isaaclab.utils.math as math_utils

# Initialize app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

def check_robot_state():
    # Import required modules  
    from legged_lab.envs.go2.go2_config import Go2FallRecoveryFlatEnvCfg
    from legged_lab.envs.base.base_env import BaseEnv
    
    print("Creating Fall Recovery environment...")
    
    # Create environment with correct config inheritance
    env_cfg = Go2FallRecoveryFlatEnvCfg()
    env_cfg.scene.num_envs = 4
    
    try:
        env = BaseEnv(cfg=env_cfg, headless=True)
        print(f"✓ Environment created successfully with {env.num_envs} environments")
        
        # Check robot initial state after reset
        if hasattr(env, 'robot') and hasattr(env.robot, 'data'):
            print(f"\n=== Robot Initial State ===")
            
            # Position and orientation  
            pos = env.robot.data.root_pos_w
            quat = env.robot.data.root_quat_w
            print(f"Position (x,y,z): {pos[0].cpu()}")
            print(f"Quaternion (w,x,y,z): {quat[0].cpu()}")
            
            # Convert quaternion to Euler for easier understanding (simplified approach)
            # For fall recovery, we mainly care about upward vs downward orientation
            print(f"Quaternion analysis: w={quat[0][0].item():.3f} (should be close to 1.0 if upright)")
            
            # Check if robot is upside down using gravity vector
            gravity_b = env.robot.data.projected_gravity_b[0]
            print(f"Gravity in body frame: [{gravity_b[0].item():.3f}, {gravity_b[1].item():.3f}, {gravity_b[2].item():.3f}]")
            upward_score = (-gravity_b[2] / torch.norm(gravity_b)).item()
            print(f"Upward score: {upward_score:.3f} (1.0=upright, -1.0=upside down, 0=sideways)")
            
            # Velocity
            lin_vel = env.robot.data.root_lin_vel_w[0]
            ang_vel = env.robot.data.root_ang_vel_w[0]
            print(f"Linear velocity: {lin_vel.cpu()}")
            print(f"Angular velocity: {ang_vel.cpu()}")
            
        # Test a few simulation steps and check rewards
        print(f"\n=== Testing Reward Calculation ===")
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        
        for step in range(3):
            obs, reward, done, info = env.step(actions)
            print(f"\nStep {step + 1}:")
            print(f"  Total reward: {reward.cpu()}")
            
            # Check individual reward components
            if hasattr(env, 'reward_manager'):
                for term_name, term in env.reward_manager.active_terms.items():
                    try:
                        value = term.function(env, **term.params)
                        weighted_value = value * term.weight
                        print(f"  {term_name}: raw={value[0].item():.6f}, weighted={weighted_value[0].item():.6f}")
                    except Exception as e:
                        print(f"  {term_name}: ERROR - {e}")
        
        env.close()
        print("\n✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_robot_state() 
    simulation_app.close()
