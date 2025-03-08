from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, SceneCfg, RobotCfg, DomainRandCfg,
    RewardCfg, HeightScannerCfg, AddRigidBodyMassCfg, PhysxCfg, SimCfg
)
from legged_lab.assets.agility import CASSIE_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class CASSIESceneCfg(SceneCfg):
    height_scanner: HeightScannerCfg = HeightScannerCfg(
        enable_height_scan=False,
        prim_body_name="pelvis"
    )
    robot: str = CASSIE_CFG
    terrain_type: str = "generator"
    terrain_generator: str = GRAVEL_TERRAINS_CFG


@configclass
class CASSIERobotCfg(RobotCfg):
    terminate_contacts_body_names: list = ["pelvis"]
    feet_names: list = [".*toe"]


@configclass
class CASSIEDomainRandCfg(DomainRandCfg):
    add_rigid_body_mass: AddRigidBodyMassCfg = AddRigidBodyMassCfg(
        enable=True,
        params={
            "body_names": ["pelvis"],
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add"
        }
    )


@configclass
class CASSIERewardCfg(RewardCfg):
    track_lin_vel_xy_exp: RewTerm = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp: RewTerm = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2: RewTerm = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2: RewTerm = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy: RewTerm = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2: RewTerm = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2: RewTerm = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts: RewTerm = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*toe.*).*"), "threshold": 1.0})
    fly: RewTerm = RewTerm(func=mdp.fly, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*toe.*"), "threshold": 1.0})
    flat_orientation_l2: RewTerm = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty: RewTerm = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time: RewTerm = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.5, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*toe.*"), "threshold": 0.4})
    feet_slide: RewTerm = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*toe.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*toe.*")})
    feet_force: RewTerm = RewTerm(func=mdp.body_force, weight=-3e-3, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*toe.*"), "threshold": 500, "max_reward": 400})
    dof_pos_limits: RewTerm = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_abduction_.*", "hip_rotation_.*"])})
    joint_deviation_toes = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["toe_joint_.*"])})


@configclass
class CASSIEFlatEnvCfg(BaseEnvCfg):
    scene: CASSIESceneCfg = CASSIESceneCfg()
    robot: CASSIERobotCfg = CASSIERobotCfg()
    domain_rand: CASSIEDomainRandCfg = CASSIEDomainRandCfg()
    reward: CASSIERewardCfg = CASSIERewardCfg()


@configclass
class CASSIEFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "cassie_flat"
    wandb_project: str = "cassie_flat"
