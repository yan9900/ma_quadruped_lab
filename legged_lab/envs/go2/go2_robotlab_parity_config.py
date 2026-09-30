"""Nominal flat Go2 task aligned with ``go2_rl_robotlab``.

This is intentionally independent of :class:`Go2FlatEnvCfg`: the existing Go2
tasks include gait shaping and other task-specific settings that are outside
this RobotLab-aligned flat experiment.
"""

from __future__ import annotations

import copy
from tkinter import FLAT

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import UNITREE_GO2_CFG
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, CameraCfg, EventCfg

from legged_lab.terrains.terrain_generator_cfg import CLIFF_DETECTION_TERRAINS_CFG, FLAT_MESH_TERRAINS_CFG, RING_UNEVEN_TERRAINS_CFG

BASE_LINK_NAME = "base"
FOOT_REGEX = r".*_foot"
BASE_HEIGHT_TARGET = 0.38
CAMERA_CLIP_RANGE = (0.3, 3.0)

# RobotLab's reference joint-name list.  In this flat task it is used only to
# select reward joints; policy observations/actions keep LeggedLab's USD order.
JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


@configclass
class Go2RobotLabRewardCfg:
    """RobotLab reward set with fixed reference weights (no curriculum)."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_frame_exp, weight=2.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_base_frame_exp, weight=1.0, params={"std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_robotlab, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 5.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[r".*_thigh", r".*_calf"]
            ),
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_soft,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_REGEX]),
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*_hip_joint"]),
            "stand_still_scale": 1.0,
            "command_threshold": 0.1,
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*_(thigh|calf)_joint"]),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )


@configclass
class Go2RobotLabEventCfg:
    """RobotLab reset and domain-randomization events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=r"^(?!.*base).*$"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
            },
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={"offset_range": (-0.035, 0.035)},
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6),
            }
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


class Go2RobotLabParityEnv(BaseEnv):
    """Parity task using LeggedLab's established flat actor/critic schema."""

    pass


@configclass
class Go2RobotLabFlatEnvCfg(BaseEnvCfg):
    """Flat-plane configuration with RobotLab domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.reward = Go2RobotLabRewardCfg()

        # Keep the local USD but align RobotLab's pose, solver, and controller.
        # The remaining URDF/USD and actuator-model differences are in the manifest.
        self.scene.robot = copy.deepcopy(UNITREE_GO2_CFG)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        self.scene.robot.actuators["legs"].stiffness = 25.0
        self.scene.robot.actuators["legs"].damping = 0.5

        self.scene.num_envs = 16384
        self.scene.env_spacing = 0.5
        # self.scene.terrain_type = "plane"
        self.scene.terrain_type = "generator"  
        # self.scene.terrain_generator = CLIFF_DETECTION_TERRAINS_CFG
        self.scene.terrain_generator = FLAT_MESH_TERRAINS_CFG
        # self.scene.terrain_generator = None
        self.scene.max_episode_length_s = 10.0
        self.scene.friction_combine_mode = "average"
        self.scene.restitution_combine_mode = "average"
        self.scene.static_friction = 1.0
        self.scene.dynamic_friction = 1.0
        self.scene.restitution = 0.0
        self.scene.camera.enable_camera = False
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME
        self.scene.height_scanner.resolution = 0.1
        self.scene.height_scanner.size = (1.6, 1.0)

        self.robot.feet_body_names = [FOOT_REGEX]
        self.robot.terminate_contacts_body_names = [BASE_LINK_NAME]
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        self.robot.action_scale = 0.25

        self.normalization.obs_scales.lin_vel = 2.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.commands = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.actions = 1.0
        self.normalization.obs_scales.height_scan = 2.5
        self.normalization.height_scan_offset = 0.5
        self.normalization.clip_observations = 100.0
        self.normalization.clip_actions = 100.0
        self.noise.add_noise = True
        self.noise.noise_scales.ang_vel = 0.2
        self.noise.noise_scales.projected_gravity = 0.05
        self.noise.noise_scales.joint_pos = 0.03
        self.noise.noise_scales.joint_vel = 2.0

        self.commands.resampling_time_range = (5.0, 5.0)
        self.commands.heading_command = False
        self.commands.rel_heading_envs = 0.0
        self.commands.rel_standing_envs = 0.0
        self.commands.debug_vis = False
        # Static full RobotLab range. UniformVelocityCommand changes commands by
        # a hard step every five seconds; no command ramp is applied.
        # self.commands.ranges.lin_vel_x = (-2.0, 2.5)
        self.commands.ranges.lin_vel_x = (2.5, 2.5)
        # self.commands.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-0.0, 0.0)
        # Match RobotLab's EventCfg exactly.  Its motor-level actuator delay is
        # tracked separately because BaseEnv's delay buffer is policy-step based.
        self.domain_rand.events = Go2RobotLabEventCfg()
        self.domain_rand.action_delay.enable = False
        # Keep two history slots for action-rate/smoothness rewards.  With
        # ``enable=False`` the selected time lag remains zero.
        self.domain_rand.action_delay.params = {"max_delay": 2, "min_delay": 0}

        self.sim.dt = 0.005
        self.sim.decimation = 4
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024


@configclass
class Go2RobotLabDataCollectionEnvCfg(Go2RobotLabFlatEnvCfg):
    """RobotLab-policy environment with the v3 depth-data collection scene.

    Policy-facing observations, default pose, actuator settings, and solver
    settings come from :class:`Go2RobotLabFlatEnvCfg`.  Camera, terrain,
    command, reset, and fault-baseline material settings intentionally follow
    the existing Go2 data-collection task.
    """

    def __post_init__(self):
        super().__post_init__()

        # Data-collection terrain and episode layout.
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = CLIFF_DETECTION_TERRAINS_CFG
        # self.scene.terrain_generator = RING_UNEVEN_TERRAINS_CFG
        self.scene.terrain_generator.curriculum = False
        self.scene.enable_random_terrain_spawn = True
        self.scene.max_episode_length_s = 10.0

        # Non-physical, base-mounted depth camera.  Avoid adding the D435 USD
        # as a rigid body because its mass would change the policy dynamics.
        self.scene.camera.enable_camera = True
        self.scene.camera.use_physical_asset = False
        self.scene.camera.prim_body_name = BASE_LINK_NAME
        self.scene.camera.height = 64
        self.scene.camera.width = 64
        self.scene.camera.history_length = 2
        self.scene.camera.update_period = self.sim.dt * self.sim.decimation
        self.scene.camera.debug_vis = True
        self.scene.camera.data_types = ["distance_to_image_plane"]
        self.scene.camera.spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=CAMERA_CLIP_RANGE,
        )
        euler_rad = torch.deg2rad(torch.tensor([180.0, 70.0, -90.0]))
        base_quat = quat_from_euler_xyz(*tuple(euler_rad))
        final_quat = torch.as_tensor(base_quat) * torch.tensor([1.0, 1.0, 1.0, -1.0])
        self.scene.camera.offset = CameraCfg.OffsetCfg(
            pos=(0.33, 0.0, 0.08),
            rot=final_quat,
            convention="ros",
        )

        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = BASE_LINK_NAME

        # Keep v3's fall/collision termination semantics for collected labels.
        self.robot.feet_body_names = [FOOT_REGEX]
        # 头部接触 >1 N 也算失败（v3 语义）。注意：RobotLab 训练只终止 base
        # （其 URDF 的 Head 关节 dont_collapse=true，头是独立 body），盲走上台阶时头常磕立面，
        # 所以在这个定义下 RobotLab policy 的上台阶成功率会低于它的物理能力。
        # 2026-09-29 曾临时改为只 base：[r".*base.*"]
        self.robot.terminate_contacts_body_names = [r".*base.*", "Head_upper", "Head_lower"]

        # Use the controlled data-collection event set instead of RobotLab's
        # broad training-time DR.  In particular, normal friction stays well
        # separated from the v3 OOD friction injected at t_fault, and periodic
        # pushes do not confound the explicit fault label.
        self.domain_rand.events = EventCfg()
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [r".*base.*"]
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.reset_base.params = {
            # "pose_range": {
            #     "x": (0.5, 2.5),
            #     "y": (0.5, 2.5),
            #     "z": (0.0, 0.0),
            #     "roll": (0.0, 0.0),
            #     "pitch": (0.0, 0.0),
            #     "yaw": (0.0, 1.57),
            # },
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Multiplication keeps the v3 foot-material fault effective against a
        # terrain material of 1.0.  RobotLab's average mode would turn a 0.1
        # foot coefficient into roughly 0.55 effective friction.
        self.scene.friction_combine_mode = "multiply"
        self.scene.restitution_combine_mode = "multiply"
        self.scene.static_friction = 1.0
        self.scene.dynamic_friction = 1.0
        self.scene.restitution = 0.0

        # Stay within the command support saved with the RobotLab checkpoint;
        # lateral/yaw ranges retain the collection task's focused envelope.
        # self.commands.resampling_time_range = (5.0, 5.0)
        self.commands.heading_command = False
        self.commands.rel_heading_envs = 0.0
        self.commands.rel_standing_envs = 0.0
        self.commands.debug_vis = False
        self.commands.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.ranges.ang_vel_z = (-0.3, 0.3)


@configclass
class Go2RobotLabFlatAgentCfg(BaseAgentCfg):
    """Standard PPO settings from RobotLab's ``PPORunnerCfg``."""

    num_steps_per_env: int = 24
    max_iterations: int = 300000
    save_interval: int = 500
    experiment_name: str = "go2_flat_robotlab_parity"
    wandb_project: str = "go2_flat_robotlab_parity"
    logger: str = "tensorboard"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False
        self.algorithm.entropy_coef = 0.01
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
