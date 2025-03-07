from .base_config import BaseConfig
from dataclasses import MISSING


class BaseEnvConfig(BaseConfig):
    device = "cuda:0"

    class sim:
        dt = 0.005
        decimation = 4

        class physx:
            solver_type = 1
            max_position_iteration_count = 8
            max_velocity_iteration_count = 8
            bounce_threshold_velocity = 0.2
            gpu_max_rigid_contact_count = 2**24
            gpu_found_lost_pairs_capacity = 2**22
            gpu_found_lost_aggregate_pairs_capacity = 2**26

    class scene:
        episode_length_s = 20.0
        num_envs = 4096
        env_spacing = 2.5
        robot = MISSING
        terrain_type = MISSING  # "generator" or "plane"
        terrain_generator = MISSING
        max_init_terrain_level = 0

    class normalization:

        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0

        clip_observations = 100.
        clip_actions = 100.
        action_scale = 0.25


class Policy:
    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


class Algorithm:
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
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
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
