from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG

# --- Constants for Go2 robot ---
BASE_LINK_NAME = "base"
FOOT_REGEX     = r".*_calf"

# --- Try to import GO2 asset cfg from common locations ---
try:
    from legged_lab.assets.unitree import GO2_CFG as GO2_CFG
except Exception:
    try:
        from legged_lab.assets.unitree import UNITREE_GO2_CFG as GO2_CFG
    except Exception:
        # fallback to your previous path
        from robot_lab.assets.unitree import UNITREE_GO2_CFG as GO2_CFG  # type: ignore
        GO2_CFG = UNITREE_GO2_CFG  # alias for consistency


# =========================
# Reward configuration
# =========================
@configclass
class Go2RewardCfg(RewardCfg):
    # ---- Original configuration (commented out for Fall Recovery setup) ----
    # # ---- Velocity tracking ----
    # track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    # track_ang_vel_z_exp  = RewTerm(func=mdp.track_ang_vel_z_world_exp,       weight=1.5, params={"std": 0.5})

    # # ---- Root / pose penalties ----
    # lin_vel_z_l2        = RewTerm(func=mdp.lin_vel_z_l2,   weight=-2.0)
    # ang_vel_xy_l2       = RewTerm(func=mdp.ang_vel_xy_l2,  weight=-0.05)
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)  # disabled by weight
    # base_height_l2      = RewTerm(func=mdp.base_height_l2, weight=0.0, params={
    #     "target_height": 0.33, "asset_cfg": SceneEntityCfg("robot", body_names=["base"])
    # })

    # ---- Fall Recovery Reward Configuration (Based on Literature) ----
    # Reduced velocity tracking weights during recovery phase
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})  # Reduced from 3.0
    track_ang_vel_z_exp  = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})  # Reduced from 1.5

    # Enhanced orientation recovery rewards (critical for fall recovery)
    upward = RewTerm(func=mdp.upward, weight=10.0)  # Significantly increased for strong recovery incentive
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)  # Enable strong penalty for tilted pose
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-5.0)  # Strong penalty for body not upright
    
    # Base height management for proper standing (critical for recovery)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={
        "target_height": 0.33, "asset_cfg": SceneEntityCfg("robot", body_names=["base"])
    })

    # Root pose penalties (more permissive during recovery)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)  # Reduced from -2.0
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)  # Reduced from -0.05
    body_lin_acc_l2     = RewTerm(func=mdp.body_lin_acc_l2, weight=0.0)

    # ---- Joints / actuation ----
    joint_torques_l2    = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    joint_vel_l2        = RewTerm(func=mdp.joint_vel_l2,     weight=0.0)
    joint_acc_l2        = RewTerm(func=mdp.joint_acc_l2,     weight=-2.5e-7)
    joint_pos_limits    = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    joint_vel_limits    = RewTerm(func=mdp.joint_vel_limits, weight=0.0)
    joint_power         = RewTerm(func=mdp.energy,           weight=-2.0e-5)  # energy == |tau * qdot|
    joint_pos_penalty   = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0)   # 用默认位形的 L1 偏差作形状正则
    joint_mirror        = RewTerm(
        func=mdp.joint_mirror, weight=-0.05,
        params={"mirror_joints": [
            # 这里建议在构建阶段把正则解析成 id 列表再传入
            # 下方为“左右对称”成对关系（示意），可据你的 URDF 实际 joint 命名调整
            (["FR_hip_joint","FR_thigh_joint","FR_calf_joint"],
             ["RL_hip_joint","RL_thigh_joint","RL_calf_joint"]),
            (["FL_hip_joint","FL_thigh_joint","FL_calf_joint"],
             ["RR_hip_joint","RR_thigh_joint","RR_calf_joint"]),
        ]}
    )

    # ---- Action smoothing ----
    action_rate_l2      = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ---- Contacts & forces ----
    undesired_contacts  = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r"^(?!.*_calf).*"]),
                "threshold": 1.0}
    )
    # 若尚未实现 mdp.contact_forces，可用 mdp.body_force 作为近似（只看竖直分量）
    contact_forces      = RewTerm(
        func=mdp.contact_forces, weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"])}
    )

    # ---- Feet behavior (quadruped) ----
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]), "threshold": 0.5}
    )
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]), "threshold": 0.5}
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]), "threshold": 1.0}
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd, weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]), "threshold": 1.0}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=[r".*_calf"]),
        }
    )
    feet_height = RewTerm(
        func=mdp.feet_height, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]),
                "asset_cfg":  SceneEntityCfg("robot",          body_names=[r".*_calf"]),
                "target_height": 0.05}
    )
    feet_height_body = RewTerm(
        func=mdp.feet_height_body, weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]),
                "asset_cfg":  SceneEntityCfg("robot",          body_names=[r".*_calf"]),
                "target_height": -0.2}
    )
    feet_gait = RewTerm(
        func=mdp.feet_gait, weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]),
            # 先按名字给出成对关系（构建期解析为 id 对更稳妥）
            "synced_feet_pair_ids": None,  # 若框架不自动解析，可在运行前注入 id 对
            "threshold": 1.0,
        }
    )

    # ---- Orientation "upward" (keep) ----
    upward = RewTerm(func=mdp.upward, weight=1.0)

    # ---- Critical: Enable termination penalty to strongly discourage falling ----
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # Strong penalty for termination


# =========================
# Fall Recovery Reward Configuration (New)
# =========================
@configclass
class Go2FallRecoveryRewardCfg(RewardCfg):
    """
    Reward configuration optimized for Fall Recovery training based on research literature.
    Key changes from standard locomotion:
    1. Reduced velocity tracking weights (recovery prioritized over speed)
    2. Significantly increased orientation recovery rewards
    3. Enabled termination penalty to discourage falling
    4. Relaxed joint penalties to allow more dynamic recovery motions
    5. More permissive contact thresholds for recovery scenarios
    """
    
    # ---- Velocity tracking (reduced importance during recovery) ----
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})  # Reduced from 3.0
    track_ang_vel_z_exp  = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})  # Reduced from 1.5

    # ---- Critical: Strong orientation recovery incentives ----
    upward = RewTerm(func=mdp.upward, weight=15.0)  # Massively increased from 1.0
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)  # Strong penalty for tilted pose
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-8.0)  # Very strong penalty for body not upright
    
    # ---- Base height management (critical for standing) ----
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-3.0, params={
        "target_height": 0.33, "asset_cfg": SceneEntityCfg("robot", body_names=["base"])
    })  # Strong penalty for incorrect height

    # ---- Root pose penalties (relaxed for recovery dynamics) ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)  # Further reduced from -2.0
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)  # Further reduced from -0.05
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=0.0)

    # ---- Joint penalties (significantly relaxed for recovery motions) ----
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)  # Much reduced from -2.5e-5
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)  # Much reduced from -2.5e-7
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)  # Much reduced from -5.0
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0.0)
    joint_power = RewTerm(func=mdp.energy, weight=-5.0e-6)  # Much reduced from -2.0e-5
    joint_pos_penalty = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2)  # Much reduced from -1.0
    
    # ---- Action smoothing (reduced for dynamic recovery) ----
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)  # Much reduced from -0.01

    # ---- Contact penalties (very relaxed for recovery) ----
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-0.2,  # Much reduced from -1.0
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r"^(?!.*_calf).*"]),
                "threshold": 2.0}  # Much increased threshold
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces, weight=-3.0e-5,  # Reduced from -1.5e-4
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"])}
    )

    # ---- Feet behavior (adapted for recovery scenarios) ----
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=0.02,  # Much reduced from 0.1
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]), "threshold": 0.5}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.02,  # Much reduced from -0.1
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[r".*_calf"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[r".*_calf"]),
        }
    )

    # ---- Critical: Strong termination penalty ----
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-300.0)  # Even stronger penalty for fall recovery


# ---- Flat Environment (Original - commented out) ----
# @configclass
# class Go2FlatEnvCfg(BaseEnvCfg):
#     reward = Go2RewardCfg()
# 
#     def __post_init__(self):
#         super().__post_init__()
#         # Robot & scene
#         self.scene.robot = GO2_CFG
#         self.scene.terrain_type = "plane"
#         self.scene.terrain_generator = None
# 
#         # Height scan off in flat
#         self.scene.height_scanner.enable_height_scan = False
#         self.scene.height_scanner.prim_body_name = BASE_LINK_NAME
# 
#         # Robot-specific semantic groups
#         self.robot.feet_body_names = [FOOT_REGEX]
#         self.robot.terminate_contacts_body_names = []  # 或 [r".*base.*", r".*_hip.*"]

# ---- Fall Recovery Flat Environment (New) ----
@configclass
class Go2FallRecoveryFlatEnvCfg(BaseEnvCfg):
    """Fall Recovery training environment with randomized initial poses and relaxed termination."""
    reward = Go2FallRecoveryRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Robot & scene
        self.scene.robot = GO2_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        
        # Extended episode length for recovery attempts
        self.scene.max_episode_length_s = 40.0  # Increased from default 20.0

        # Height scan off in flat
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME

        # Robot-specific semantic groups
        self.robot.feet_body_names = [FOOT_REGEX]
        # Fall Recovery: Disable contact termination or make very permissive
        self.robot.terminate_contacts_body_names = []  # Allow all contact without termination

        # Fall Recovery: Aggressive initial pose randomization (critical for fall recovery training)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        
        # Override reset_base event for fall recovery training
        self.domain_rand.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.6),  # Allow low starting heights
                "roll": (-3.14, 3.14),    # Full range - robot can start upside down
                "pitch": (-3.14, 3.14),   # Full range - robot can start upside down  
                "yaw": (-3.14, 3.14)      # Full range
            },
            "velocity_range": {
                "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0),
                "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)
            }
        }
        
        # More aggressive joint position randomization
        self.domain_rand.events.reset_robot_joints.params = {
            "position_range": (0.2, 1.8),  # Wider range than default (0.5, 1.5)
            "velocity_range": (-0.5, 0.5)  # Some initial joint velocity
        }
        
        # Enable periodic base reset for variety in fall recovery scenarios
        if hasattr(self.domain_rand.events, 'reset_base_position'):
            self.domain_rand.events.reset_base_position.mode = "interval"
            self.domain_rand.events.reset_base_position.interval_range_s = (10.0, 20.0)

# ---- Original Flat Environment (kept for reference) ----
@configclass
class Go2FlatEnvCfg(BaseEnvCfg):
    reward = Go2RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Robot & scene
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

        # Actions（若 RobotCfg 暴露了 action 配置项，可打开下列注释）
        # try:
        #     self.robot.action.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        #     self.robot.action.joint_pos.clip  = {".*": (-100.0, 100.0)}
        #     self.robot.action.joint_pos.joint_names = [
        #         "FR_hip_joint","FR_thigh_joint","FR_calf_joint",
        #         "FL_hip_joint","FL_thigh_joint","FL_calf_joint",
        #         "RR_hip_joint","RR_thigh_joint","RR_calf_joint",
        #         "RL_hip_joint","RL_thigh_joint","RL_calf_joint",
        #     ]
        # except AttributeError:
        #     pass  # 某些版本的 BaseEnvCfg 不在此处挂 action；如无则在 RobotCfg 侧配置

        # Domain randomization / events（按 H1 风格）
        # 已知 H1 演示了 add_base_mass；这里保持最小化、可逐步加：
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        # 如需 base reset 扰动/外力脉冲/COM 偏置，可按你之前的范围逐步加到 events.* 中
        # 例（若你的框架提供这些事件名）：
        # self.domain_rand.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5,0.5),"y": (-0.5,0.5),"z": (0.0,0.2),
        #                    "roll": (-3.14,3.14),"pitch": (-3.14,3.14),"yaw": (-3.14,3.14)},
        #     "velocity_range": {"x": (-0.5,0.5),"y": (-0.5,0.5),"z": (-0.5,0.5),
        #                        "roll": (-0.5,0.5),"pitch": (-0.5,0.5),"yaw": (-0.5,0.5)},
        # }

@configclass
class Go2FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_flat"
    wandb_project:  str = "go2_flat"


# ---- Rough ----
@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Terrain & sensors
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME

        # Observation history（与 H1 Rough 同风格）
        self.robot.actor_obs_history_length  = 1
        self.robot.critic_obs_history_length = 1

        # Rough-specific reward tweaks（保留你原 Rough 的权重）
        self.reward.track_lin_vel_xy_exp.weight = 3.0
        self.reward.track_ang_vel_z_exp.weight  = 1.5
        self.reward.lin_vel_z_l2.weight         = -2.0
        # 你也可以在 rough 下适度提高足端相关 shaping：
        # self.reward.feet_slide.weight = -0.15
        # self.reward.contact_forces.weight = -2.0e-4

@configclass
class Go2RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "go2_rough"
    wandb_project:  str = "go2_rough"

    def __post_init__(self):
        super().__post_init__()

@configclass
class Go2FallRecoveryAgentCfg(BaseAgentCfg):
    """Agent configuration for fall recovery training."""
    experiment_name: str = "go2_fall_recovery"
    wandb_project: str = "go2_fall_recovery"
    
    def __post_init__(self):
        super().__post_init__()
        # Longer training for fall recovery (complex behavior)
        self.max_iterations = 3000  # Increase from default if needed
        
        # Fall recovery may need longer episodes for learning  
        self.algorithm.value_loss_coef = 2.0  # Higher value function learning
        self.algorithm.learning_rate = 5e-4   # Slightly higher LR
        
        # More exploration for complex recovery behaviors
        self.algorithm.entropy_coef = 0.01
        # 你可以按 H1 Rough 一样先上 RNN，再观察是否有收益
        # self.policy.class_name = "ActorCriticRecurrent"
        # self.policy.actor_hidden_dims = [256, 256, 128]
        # self.policy.critic_hidden_dims = [256, 256, 128]
        # self.policy.rnn_hidden_size = 256
        # self.policy.rnn_num_layers = 1
        # self.policy.rnn_type = "lstm"

