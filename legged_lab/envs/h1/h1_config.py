from legged_lab.envs.base.base_env_config import (
    BaseEnvConfig, BaseAgentConfig, SceneConfig, RobotConfig, DomainRandConfig,
    RewardConfig, HeightScannerConfig, AddRigidBodyMassConfig, PhysxConfig, SimConfig
)
from legged_lab.assets.unitree import H1_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class H1SceneConfig(SceneConfig):
    height_scanner: HeightScannerConfig = HeightScannerConfig(
        prim_body_name="torso_link"
    )
    robot: str = H1_CFG
    terrain_type: str = "generator"
    terrain_generator: str = GRAVEL_TERRAINS_CFG


@configclass
class H1RobotConfig(RobotConfig):
    terminate_contacts_body_names: list = [".*torso.*"]


@configclass
class H1DomainRandConfig(DomainRandConfig):
    add_rigid_body_mass: AddRigidBodyMassConfig = AddRigidBodyMassConfig(
        enable=True,
        params={
            "body_names": [".*torso.*"],
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add"
        }
    )


@configclass
class H1RewardConfig(RewardConfig):
    track_lin_vel_xy_exp: RewTerm = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp: RewTerm = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2: RewTerm = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2: RewTerm = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy: RewTerm = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2: RewTerm = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2: RewTerm = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts: RewTerm = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0})
    fly: RewTerm = RewTerm(func=mdp.fly, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 1.0})
    flat_orientation_l2: RewTerm = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty: RewTerm = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time: RewTerm = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.5, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle.*"), "threshold": 0.4})
    feet_slide: RewTerm = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle.*")})
    feet_force: RewTerm = RewTerm(func=mdp.body_force, weight=-3e-3, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 500, "max_reward": 400})
    dof_pos_limits: RewTerm = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip: RewTerm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])})
    joint_deviation_arms: RewTerm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*torso.*", ".*_shoulder.*", ".*_elbow.*"])})
    joint_deviation_legs: RewTerm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])})


@configclass
class H1SimConfig(SimConfig):
    physx: PhysxConfig = PhysxConfig(gpu_max_rigid_patch_count=10 * 2**15)


@configclass
class H1FlatEnvCfg(BaseEnvConfig):
    scene: H1SceneConfig = H1SceneConfig()
    robot: H1RobotConfig = H1RobotConfig()
    domain_rand: H1DomainRandConfig = H1DomainRandConfig()
    reward: H1RewardConfig = H1RewardConfig()
    sim: H1SimConfig = H1SimConfig()

@configclass
class H1FlatAgentCfg(BaseAgentConfig):
    experiment_name: str = "h1_flat"
    wandb_project: str = "h1_flat"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
