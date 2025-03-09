from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, BaseSceneCfg, RobotCfg, DomainRandCfg,
    RewardCfg, HeightScannerCfg, AddRigidBodyMassCfg, PhysxCfg, SimCfg, MLPPolicyCfg, RNNPolicyCfg
)
from legged_lab.assets.fftai import GR2_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class GR2RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot_pitch.*).*"), "threshold": 1.0})
    fly = RewTerm(func=mdp.fly, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot_pitch.*"), "threshold": 1.0})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.15, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot_pitch.*"), "threshold": 0.4})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot_pitch.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_pitch.*")})
    feet_force = RewTerm(func=mdp.body_force, weight=-3e-3, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot_pitch.*"), "threshold": 500, "max_reward": 400})
    feet_too_near = RewTerm(func=mdp.feet_too_near_humanoid, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_pitch.*"]), "threshold": 0.25})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot_pitch.*")})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.15, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"])})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*head.*", ".*waist_yaw.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"])})
    joint_deviation_legs = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*ankle.*"])})


@configclass
class GR2FlatEnvCfg(BaseEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link"
        ),
        robot=GR2_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG
    )
    robot = RobotCfg(
        terminate_contacts_body_names=[".*torso.*"],
        feet_names=[".*foot_pitch.*"]
    )
    domain_rand = DomainRandCfg(
        add_rigid_body_mass=AddRigidBodyMassCfg(
            enable=True,
            params={
                "body_names": [".*torso.*"],
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add"
            }
        )
    )
    reward = GR2RewardCfg()


@configclass
class GR2FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "gr2_flat"
    wandb_project: str = "gr2_flat"


@configclass
class GR2RoughEnvCfg(GR2FlatEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link"
        ),
        robot=GR2_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG
    )
    robot = RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        terminate_contacts_body_names=[".*torso.*"],
        feet_names=[".*ankle_roll.*"]
    )
    reward = GR2RewardCfg(
        track_lin_vel_xy_exp=RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5}),
        track_ang_vel_z_exp=RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}),
        lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    )


@configclass
class GR2RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "gr2_rough"
    wandb_project: str = "gr2_rough"
    policy = RNNPolicyCfg()
