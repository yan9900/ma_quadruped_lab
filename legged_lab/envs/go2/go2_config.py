from __future__ import annotations

import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
    CameraCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from legged_lab.terrains.terrain_generator_cfg import CLIFF_DETECTION_TERRAINS_CFG

# --- Constants for Go2 robot ---
BASE_LINK_NAME = "base"
FOOT_REGEX = r".*_foot"
CLIP_RANGE = (0.3, 3.0)  # 相机剪切范围


from legged_lab.assets.unitree import UNITREE_GO2_CFG as GO2_CFG



# =========================
# Fall Recovery Reward Configuration
# =========================
@configclass  
class Go2FallRecoveryRewardCfg:
    """Fall Recovery reward configuration based on research paper"""
    
    # Orientation Posture
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)  # Base Orientation
    upright_orientation_root = RewTerm(func=mdp.upright_orientation_root, weight=6.0, params={"epsilon": 0.6})  # Upright Orientation
    target_posture = RewTerm(func=mdp.target_posture, weight=4.0, params={"epsilon": 0.6})  # Target Posture epsilon=0.6

    # # Contact Management
    feet_contact = RewTerm(func=mdp.feet_contact, weight=0.3, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX])})  # Feet Contact
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-0.2, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*base.*", r".*_hip.*", r".*_thigh.*", r".*_calf.*"])})  # Body Contact

    # # Stability Control
    safety_force = RewTerm(func=mdp.safety_force, weight=-1.0e-2, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf.*"])})  # Safety Force (calf contacts)
    body_bias = RewTerm(func=mdp.body_bias, weight=-0.1)  # Body-bias
    
    # # Motion Constraints
    position_limits = RewTerm(func=mdp.position_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})  # Position Limits - All joints original -1.0
    angular_velocity_limits = RewTerm(func=mdp.angular_velocity_limits, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})  # Angular Velocity Limit - All joints
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})  # Joint Acceleration - All joints
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})  # Joint Velocity - All joints
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # Action Smoothing - No joint specification needed
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})  # Joint Torques - All joints original -5.0e-4 
    
    # # Strong termination penalty
    # is_terminated = RewTerm(func=mdp.is_terminated, weight=-300.0)


# =========================
# Fall Recovery Environment  
# =========================
@configclass
class Go2FallRecoveryFlatEnvCfg(BaseEnvCfg):  # 临时继承BaseEnvCfg
    """Fall Recovery environment configuration"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 设置reward配置
        self.reward = Go2FallRecoveryRewardCfg()
        
        # Robot & scene 配置
        self.scene.robot = GO2_CFG
        # self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        # self.scene.robot.init_state.rot = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z) 躺下
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.max_episode_length_s = 5.0  # in seconds
        
        # Height scan配置
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME
        
        # Robot specific配置
        self.robot.feet_body_names = [FOOT_REGEX]
        self.robot.terminate_contacts_body_names = []  # 禁用接触终止
        
        # 修复domain randomization events
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        
        # Fall Recovery: 更强的随机初始姿态 (确保从倒地状态开始)
        # 最终的pose和orientations分别是是通过
        # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]计算的
        #
        
        self.domain_rand.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0),
                "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)
                # "roll": (-3.14, -3.14), "pitch": (0, 0), "yaw": (-3.14, 3.14)
                
            },
            "velocity_range": {
                "x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)
            }
        }


# # =========================
# # Custom Environment Class for Fall Recovery
# # =========================
class Go2FallRecoveryEnv(BaseEnv):
    """Custom environment for Go2 fall recovery with paper-specific observations"""
    
    def compute_current_observations(self):
        """
        Compute observations according to the paper specification:
        
        Actor obs (p_t ∈ R^75):
        - o_t ∈ R^42: proprioceptive observation (body angular velocity, projected gravity, joint angles, joint velocities, previous action)
        - m̂_t ∈ R^4: predicted mass distribution (base, hip, thigh, calf) 
        - ĉ_t ∈ R^13: collision probabilities
        - ẑ_t ∈ R^16: latent representation
        
        Critic obs (s_t):
        - o_t ∈ R^42: proprioceptive observation
        - h_t ∈ R^187: height map scan dots (disabled for flat terrain)
        - m_t ∈ R^4: true masses
        - k_PD ∈ R^24: PD control gains
        - p_com ∈ R^2: center of mass position
        - c_t ∈ R^13: contact states
        - c_f ∈ R^4: contact forces of each foot
        - μ ∈ R^1: friction coefficient
        """
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # ===== Base proprioceptive observations (same for actor and critic) =====
        ang_vel = robot.data.root_ang_vel_b  # R^3
        projected_gravity = robot.data.projected_gravity_b  # R^3
        # command = self.command_generator.command  # R^3 (lin_vel_x, lin_vel_y, ang_vel_z)
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos  # R^12
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel  # R^12
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]  # R^12
        
        # Base proprioceptive observation o_t ∈ R^42
        proprioceptive_obs = torch.cat([
            ang_vel * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            # command * self.obs_scales.commands,
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            action * self.obs_scales.actions,
        ], dim=-1)  # R^42

        # ===== Actor-specific observations =====
        # For now, we'll use placeholder values for the paper-specific components
        # In a full implementation, these would come from specialized network modules
        
        # Predicted mass distribution m̂_t ∈ R^4 (base, hip, thigh, calf masses)
        # Placeholder: use normalized robot mass distribution
        predicted_masses = torch.ones(self.num_envs, 4, device=self.device) * 0.25  # R^4
        
        # Collision probabilities ĉ_t ∈ R^13 (body components collision probabilities)
        # Placeholder: based on current contact forces
        collision_probs = torch.zeros(self.num_envs, 13, device=self.device)  # R^13
        contact_threshold = 1.0
        if hasattr(self, 'termination_contact_cfg') and len(self.termination_contact_cfg.body_ids) > 0:
            body_contacts = torch.norm(net_contact_forces[:, -1, self.termination_contact_cfg.body_ids], dim=-1)
            collision_probs[:, :len(self.termination_contact_cfg.body_ids)] = torch.sigmoid(body_contacts - contact_threshold)
        
        # Latent representation ẑ_t ∈ R^16
        # Placeholder: encoded state information
        latent_repr = torch.zeros(self.num_envs, 16, device=self.device)  # R^16
        
        # Combine actor observations
        actor_obs = torch.cat([
            proprioceptive_obs,  # R^42
            predicted_masses,    # R^4
            collision_probs,     # R^13  
            latent_repr         # R^16
        ], dim=-1)  # Total: R^75

        # ===== Critic-specific observations =====
        
        # Height map scan (disabled for flat terrain)
        height_scan = torch.zeros(self.num_envs, 0, device=self.device)  # R^0 for flat terrain
        
        # True masses m_t ∈ R^4 (base, hip, thigh, calf)
        # Placeholder: use actual robot mass distribution
        true_masses = torch.ones(self.num_envs, 4, device=self.device) * 0.25  # R^4
        
        # PD control gains k_PD ∈ R^24 (12 joints × 2 gains each: kp, kd)
        pd_gains = torch.zeros(self.num_envs, 24, device=self.device)  # R^24
        # Fill with placeholder PD gains
        pd_gains[:, :12] = 25.0  # kp gains
        pd_gains[:, 12:] = 0.5   # kd gains
        
        # Center of mass position p_com ∈ R^2 (x, y)
        com_pos = robot.data.root_pos_w[:, :2]  # R^2
        
        # Contact states c_t ∈ R^13 (body components contact states)
        contact_states = torch.zeros(self.num_envs, 13, device=self.device)  # R^13
        if hasattr(self, 'termination_contact_cfg') and len(self.termination_contact_cfg.body_ids) > 0:
            body_forces = torch.norm(net_contact_forces[:, -1, self.termination_contact_cfg.body_ids], dim=-1)
            contact_states[:, :len(self.termination_contact_cfg.body_ids)] = (body_forces > 0.5).float()
        
        # Feet contact forces c_f ∈ R^4
        feet_forces = torch.zeros(self.num_envs, 4, device=self.device)  # R^4
        if hasattr(self, 'feet_cfg') and len(self.feet_cfg.body_ids) > 0:
            feet_contact_forces = torch.norm(net_contact_forces[:, -1, self.feet_cfg.body_ids], dim=-1)
            feet_forces[:, :len(self.feet_cfg.body_ids)] = feet_contact_forces
        
        # Friction coefficient μ ∈ R^1
        friction_coeff = torch.ones(self.num_envs, 1, device=self.device) * 0.8  # R^1
        
        # Combine critic observations
        critic_obs_components = [
            proprioceptive_obs,  # R^42
            true_masses,         # R^4
            pd_gains,           # R^24
            com_pos,            # R^2
            contact_states,     # R^13
            feet_forces,        # R^4
            friction_coeff      # R^1
        ]
        
        # Add height scan only if enabled
        if height_scan.shape[1] > 0:
            critic_obs_components.insert(1, height_scan)  # Insert after proprioceptive_obs
            
        critic_obs = torch.cat(critic_obs_components, dim=-1)  # Total: R^90 (without height scan)
        print(actor_obs.shape, critic_obs.shape)
        # print("action", action.shape, action)
        return actor_obs, critic_obs


# =========================  
# Agent Configuration
# =========================
@configclass
class Go2FallRecoveryAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_fall_recovery"
    wandb_project: str = "go2_fall_recovery"
    logger: str = "tensorboard"  # 使用tensorboard避免wandb问题
    max_iterations: int = 10000   # 增加训练迭代次数，fall recovery需要更多训练
    num_steps_per_env = 32       # 增加每个环境的步数，给更多时间学习recovery
    
    def __post_init__(self):
        super().__post_init__()

        # policy观测用于actor，critic观测用于critic
        self.obs_groups = {
            "policy": ["policy"],     # Actor使用env返回的"policy"观测
            "critic": ["critic"]      # Critic使用env返回的"critic"观测
        }

# =========================
# Original GO2 Configurations (for compatibility)
# =========================
@configclass
class Go2RewardCfg:
    """Basic GO2 reward configuration"""
    #===================================kinematic rewards===================================
    # Velocity Tracking Rewards
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_base_frame_exp, weight=1.5, params={"std": 0.5}) 

    # Root Penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Orientation "upward"
    upward = RewTerm(func=mdp.upward, weight=1.0)
    # newly added
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-5.0, 
                            #  params={"target_height": 0.33, "sensor_cfg": SceneEntityCfg("height_scanner", body_names=[BASE_LINK_NAME])})
                             params={"target_height": 0.5})

    #===================================joint rewards===================================
    # Joint Penalties
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    joint_vel_l2        = RewTerm(func=mdp.joint_vel_l2,     weight=0.0)
    joint_acc_l2        = RewTerm(func=mdp.joint_acc_l2,     weight=-2.5e-7)
    joint_pos_limits    = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    joint_vel_limits    = RewTerm(func=mdp.joint_vel_limits, weight=0.0)
    joint_power         = RewTerm(func=mdp.energy,           weight=-2.0e-5)  # energy == |tau * qdot|
    joint_pos_penalty   = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0,
                                #   params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[r".*_hip.*"])})   # 用默认位形的 L1 偏差作形状正则
                                  params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[r".*_hip_joint"])})   # 用默认位形的 L1 偏差作形状正则, 只惩罚hip关节
    
    #===================================action smoothness===================================
    # Action Penalties
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    #===================================contact rewards===================================
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, 
                                 params={"threshold": 1.0, 
                                         "sensor_cfg": SceneEntityCfg("contact_sensor", 
                                                                       body_names=[r".*base.*", r".*_hip.*", r".*_thigh.*", r".*_calf.*"])})
    
    # ===================================gait related rewards===================================
    # # ---- Feet behavior (quadruped) ----
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]), "threshold": 0.5}
    )
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]), "threshold": 0.5}
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]), "threshold": 1.0}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_body_frame, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=[FOOT_REGEX]),
        }
    )
    feet_height = RewTerm(
        func=mdp.feet_height, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]),
                "asset_cfg":  SceneEntityCfg("robot",          body_names=[FOOT_REGEX]),
                "target_height": 0.05,"tanh_mult":2.0}
    )
    feet_height_body = RewTerm(
        func=mdp.feet_height_body, weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]),
                "asset_cfg":  SceneEntityCfg("robot",          body_names=[FOOT_REGEX]),
                "target_height": -0.2,"tanh_mult":2.0}
    )
    # # feet_gait = RewTerm(
    # #     func=mdp.feet_gait, weight=0.5,
    # #     params={
    # #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]),
    # #         # 先按名字给出成对关系（构建期解析为 id 对更稳妥）
    # #         "synced_feet_pair_ids": None,  # 若框架不自动解析，可在运行前注入 id 对
    # #         "threshold": 1.0,
    # #     }
    # # )
    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.5, #0.5
        params={
            "std": math.sqrt(0.5),
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            # "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")), #trotting
            # "synced_feet_pair_names": (("FL_foot", "FR_foot"), ("RL_foot", "RR_foot")), #bounding
            "synced_feet_pair_names": (("FL_foot", "RL_foot"), ("FR_foot", "RR_foot")), # pacing
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[FOOT_REGEX],
            ),
        },
    )
    
    joint_mirror        = RewTerm(
        func=mdp.joint_mirror, weight=-0.05,
        params={"mirror_joints": [
            # 这里建议在构建阶段把正则解析成 id 列表再传入
            # 下方为“左右对称”成对关系（示意），可据你的 URDF 实际 joint 命名调整
            # 和feet_gait的顺序一致？
            # trotting
            # (["FL_hip_joint","FL_thigh_joint","FL_calf_joint"],
            #  ["RR_hip_joint","RR_thigh_joint","RR_calf_joint"]),
            # (["FR_hip_joint","FR_thigh_joint","FR_calf_joint"],
            #  ["RL_hip_joint","RL_thigh_joint","RL_calf_joint"]),
            # bounding
            # (["FL_hip_joint","FL_thigh_joint","FL_calf_joint"],
            #  ["FR_hip_joint","FR_thigh_joint","FR_calf_joint"]),
            # (["RL_hip_joint","RL_thigh_joint","RL_calf_joint"],
            #  ["RR_hip_joint","RR_thigh_joint","RR_calf_joint"]),
            # pacing
            (["FL_hip_joint","FL_thigh_joint","FL_calf_joint"],
             ["RL_hip_joint","RL_thigh_joint","RL_calf_joint"]),
            (["FR_hip_joint","FR_thigh_joint","FR_calf_joint"],
             ["RR_hip_joint","RR_thigh_joint","RR_calf_joint"]),
        ]}
    )
    
    #====================================standing still rewards====================================
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd, weight = -2.0, 
        params={"command_threshold": 0.1})

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd, weight=0.1, 
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[FOOT_REGEX]), "threshold": 1.0}
    )

@configclass
class Go2FlatEnvCfg(BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.reward = Go2RewardCfg()
        self.scene.robot = GO2_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None

        # Height scan off in flat
        self.scene.height_scanner.enable_height_scan = False
        # 将扫描原点/体绑定到 base（如果你的 HeightScanner 需要）
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME

        # Robot-specific semantic groups
        self.robot.feet_body_names = [FOOT_REGEX]
        # 若你不打算用“非法接触终止”，可不设置或改为更宽松的体：
        self.robot.terminate_contacts_body_names = []  # 或 [r".*base.*", r".*_hip.*"]


        # Domain randomization / events
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        # 如需 base reset 扰动/外力脉冲/COM 偏置，可按你之前的范围逐步加到 events.* 中
        # 例（若你的框架提供这些事件名）：
        # self.domain_rand.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5,0.5),"y": (-0.5,0.5),"z": (0.0,0.2),
        #                    "roll": (-3.14,3.14),"pitch": (-3.14,3.14),"yaw": (-3.14,3.14)},
        #     "velocity_range": {"x": (-0.5,0.5),"y": (-0.5,0.5),"z": (-0.5,0.5),
        #                        "roll": (-0.5,0.5),"pitch": (-0.5,0.5),"yaw": (-0.5,0.5)},
        # }
        
        # 速度太小可能会影响gait
        self.commands.ranges.lin_vel_x = (-1.0, 2.5)  # 前后速度
        # 全部收到行走指令，不存在静止站立指令
        self.commands.rel_standing_envs = 0.0
@configclass
class Go2FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_flat"
    wandb_project: str = "go2_flat"
    logger: str = "tensorboard"  # 使用tensorboard避免wandb问题
    max_iterations: int = 10000   # 增加训练迭代次数，fall recovery需要更多训练
    
    def __post_init__(self):
        super().__post_init__()
        # RSL-RL 需要知道如何映射观测组到策略网络
        self.obs_groups = {
            "policy": ["policy"],     # Actor使用env返回的"policy"观测
            "critic": ["critic"]      # Critic使用env返回的"critic"观测
        }

# ---- Rough ----
@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Import here to avoid circular imports
        # from legged_lab.terrains import ROUGH_TERRAINS_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME

        # Observation history for rough terrain
        self.robot.actor_obs_history_length  = 1
        self.robot.critic_obs_history_length = 1

        # Rough terrain specific reward adjustments
        self.reward.track_lin_vel_xy_exp.weight = 2.5  # 降低速度跟踪权重，rough terrain更难
        self.reward.track_ang_vel_z_exp.weight  = 1.2
        # self.reward.lin_vel_z_l2.weight         = -2.0
        # self.reward.feet_slide.weight = -0.1  # 增加防滑惩罚
        # self.reward.upward.weight = 1.0  # 增加保持直立奖励
        
        # 为 rough terrain 添加更多 domain randomization
        # self.domain_rand.events.reset_base.params = {
        #     "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.1),
        #                    "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
        #     "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
        #                        "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
        # }

@configclass
class Go2RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_rough"
    wandb_project: str = "go2_rough"
    logger: str = "tensorboard"  # 使用tensorboard避免wandb问题
    max_iterations: int = 10000   
    
    def __post_init__(self):
        super().__post_init__()
        # RSL-RL 需要知道如何映射观测组到策略网络
        self.obs_groups = {
            "policy": ["policy"],     # Actor使用env返回的"policy"观测
            "critic": ["critic"]      # Critic使用env返回的"critic"观测
        }

# =========================
# Data Collection Configuration
# =========================
@configclass
class Go2DataCollectionEnvCfg(BaseEnvCfg):
    """Go2 environment configuration for data collection with camera"""
    # __init__中只能完成简单赋值，如果有复杂的计算需要放在__post_init__中
    def __post_init__(self):
        super().__post_init__()
        
        # 基础环境配置
        self.scene.robot = GO2_CFG
        self.scene.terrain_type = "generator"  
        self.scene.terrain_generator = CLIFF_DETECTION_TERRAINS_CFG  
        self.scene.max_episode_length_s = 10.0  # 数据收集时间
        
        # enable camera
        self.scene.camera.enable_camera = True
        self.scene.camera.use_physical_asset = True  # 是否使用物理USD模型
        self.scene.camera.prim_body_name = "base"  # 如果没有启用usd模型，则绑定到base，否则绑定到相机自身
        self.scene.camera.height = 60  # 图像高度
        self.scene.camera.width = 106   # 图像宽度
        self.scene.camera.history_length = 2  # 历史长度
        self.scene.camera.update_period = 0.025  # 40 FPS (0.005*5)
        self.scene.camera.debug_vis = True  # 可视化调试
        self.scene.camera.data_types = ["distance_to_image_plane"]  #深度
        
        # 相机硬件配置
        self.scene.camera.spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=CLIP_RANGE
        )
        if self.scene.camera.use_physical_asset:
            self.scene.camera.offset = CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), 
                rot=torch.as_tensor([1.0, 0.0, 0.0, 0.0]), # wxyz
                convention="ros"
            )
        else:
            euler_angles = torch.tensor([180.0, 70.0, -90.0])
            euler_rad = torch.deg2rad(euler_angles)
            base_quat = quat_from_euler_xyz(*tuple(euler_rad))
            flip_vector = torch.tensor([1.0, 1.0, 1.0, -1.0])
            final_quat = torch.tensor(base_quat) * flip_vector
            
            self.scene.camera.offset = CameraCfg.OffsetCfg(
                pos=(0.33, 0.0, 0.08), 
                rot=final_quat, 
                convention="ros"
            )
            
        # Height Scanner (可选，用于地形信息)
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME
        
        # Robot配置
        self.robot.feet_body_names = [FOOT_REGEX]
        self.robot.terminate_contacts_body_names = [BASE_LINK_NAME]  # 数据收集时不终止
        
        # 简化奖励（数据收集不需要奖励）
        # self.reward = Go2RewardCfg()
        self.reward = None
        
        
        # 减少domain randomization（数据收集阶段保持稳定）
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        
        # 多样化的初始状态（用于数据多样性）
        self.domain_rand.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            }
        }
        # commands variations
        # 不考虑后退
        self.commands.ranges.lin_vel_x = (0, 2.0)
        self.commands.ranges.lin_vel_y = (0, 1.0) 
        self.commands.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.rel_standing_envs = 0.0  # %全部运动
        
        print(f"camera orientation (w,x,y,z): {self.scene.camera.offset.rot}")