from .base_config import BaseConfig
from dataclasses import MISSING
import math
from isaaclab.utils import configclass


class BaseEnvConfig(BaseConfig):
    device = "cuda:0"

    class scene:
        class height_scanner:
            enable_height_scan = False
            prim_body_name = MISSING
            resolution = 0.1
            size = (1.6, 1.0)
            debug_vis = False

        max_episode_length_s = 20.0
        num_envs = 4096
        env_spacing = 2.5
        robot = MISSING
        terrain_type = MISSING  # "generator" or "plane"
        terrain_generator = MISSING
        max_init_terrain_level = 5

    class robot:
        action_scale = 0.25
        terminate_contacts_body_names = []

    @configclass
    class reward:
        pass

    class normalization:
        class obs_scales:
            ang_vel = 1.0
            projected_gravity = 1.0
            commands = 1.0
            joint_pos = 1.0
            joint_vel = 1.0
            actions = 1.0
            height_scan = 1.0

        clip_observations = 100.0
        clip_actions = 100.0
        height_scan_offset = 0.5

    class commands:
        resampling_time_range = (10.0, 10.0)
        rel_standing_envs = 0.2
        rel_heading_envs = 1.0
        heading_command = True
        heading_control_stiffness = 0.5
        debug_vis = True

        class ranges:
            lin_vel_x = (-1.0, 1.0)
            lin_vel_y = (-0.6, 0.6)
            ang_vel_z = (-1.0, 1.0)
            heading = (-math.pi, math.pi)

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            ang_vel = 0.2
            projected_gravity = 0.05
            joint_pos = 0.01
            joint_vel = 1.5
            height_scan = 0.1

    class domain_rand:

        class reset_robot_joints:
            params = {"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}

        class reset_robot_base:
            params = {
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
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
            }

        class randomize_robot_friction:
            enable = True
            params = {
                "static_friction_range": [0.6, 1.0],
                "dynamic_friction_range": [0.4, 0.8],
                "restitution_range": [0.0, 0.005],
                "num_buckets": 64,
            }

        class add_rigid_body_mass:
            enable = True
            params = {
                "body_names": MISSING,
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add",
            }

        class push_robot:
            enable = True
            push_interval_s = 15.0
            params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}

    class sim:
        dt = 0.005
        decimation = 4

        class physx:
            gpu_max_rigid_patch_count = 10 * 2**15


class BaseAgentConfig(BaseConfig):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = MISSING
    empirical_normalization = False
    device = "cuda:0"
    run_name = ""
    logger = "wandb"
    wandb_project = MISSING
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    class policy:
        class_name = "ActorCritic"
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        activation = "elu"

    class algorithm:
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
