from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
)

# --- Constants for Go2 robot ---
BASE_LINK_NAME = "base"
FOOT_REGEX = r".*_calf"

# --- Try to import GO2 asset cfg ---
try:
    from legged_lab.assets.unitree import GO2_CFG as GO2_CFG
except Exception:
    try:
        from legged_lab.assets.unitree import UNITREE_GO2_CFG as GO2_CFG
    except Exception:
        from robot_lab.assets.unitree import UNITREE_GO2_CFG as GO2_CFG


# =========================
# Fall Recovery Reward Configuration
# =========================
@configclass  
class Go2FallRecoveryRewardCfg:
    """Simplified Fall Recovery reward configuration"""
    # Reduced velocity tracking (less important during recovery)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})
    
    # Strong orientation incentives (critical for recovery)  
    upward = RewTerm(func=mdp.upward, weight=15.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    
    # Basic penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    
    # Relaxed joint penalties for dynamic recovery
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    
    # Strong termination penalty
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-300.0)


# =========================
# Fall Recovery Environment
# =========================
@configclass
class Go2FallRecoveryFlatEnvCfg(BaseEnvCfg):
    """Simplified Fall Recovery training environment"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set reward configuration
        self.reward = Go2FallRecoveryRewardCfg()
        
        # Robot & scene
        self.scene.robot = GO2_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.max_episode_length_s = 40.0  # Extended for recovery
        
        # Height scan off
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME
        
        # Robot-specific semantic groups
        self.robot.feet_body_names = [FOOT_REGEX]
        self.robot.terminate_contacts_body_names = []  # Disable contact termination
        
        # Fall Recovery: Random initial poses (can start upside down)
        self.domain_rand.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.6),
                "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0),
                "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)
            }
        }


# =========================  
# Agent Configuration
# =========================
@configclass
class Go2FallRecoveryAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_fall_recovery"
    wandb_project: str = "go2_fall_recovery"
    max_iterations: int = 3000
