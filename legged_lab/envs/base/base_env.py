from legged_lab.envs.base.base_env_config import BaseEnvConfig
import torch
from rsl_rl.env import VecEnv
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene
from isaaclab.assets.articulation import Articulation
from legged_lab.envs.env_utils.scene import SceneCfg
import numpy as np


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvConfig, hedless):
        self.cfg: BaseEnvConfig

        self.cfg = cfg
        self.hedless = hedless

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(
                solver_type=cfg.sim.physx.solver_type,
                max_position_iteration_count=cfg.sim.physx.max_position_iteration_count,
                max_velocity_iteration_count=cfg.sim.physx.max_velocity_iteration_count,
                bounce_threshold_velocity=cfg.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=cfg.sim.physx.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=cfg.sim.physx.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

        scene_cfg = SceneCfg(config=cfg.scene)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor = self.scene["contact_sensor"]

        self._init_buffers()

    def _init_buffers(self):
        self.extras = {}

        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs

        self.max_episode_length = np.ceil(self.cfg.scene.episode_length_s / self.step_dt)
        self.default_joint_pos = self.robot.data.default_joint_pos
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_observations = self.cfg.normalization.clip_observations
        self.action_scale = self.cfg.normalization.action_scale
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.sim_step_counter = 0

    def compute_observations(self):
        a = torch.zeros(self.num_envs, 64, device=self.device, dtype=torch.float)
        c = torch.zeros(self.num_envs, 64, device=self.device, dtype=torch.float)
        return a, c

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        extras = {}
        extras["observations"] = {"critic": critic_obs}
        return actor_obs, extras

    def reset(self):
        pass

    def step(self, actions: torch.Tensor):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        processed_actions = self.actions * self.action_scale + self.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.rew_buf, self.reset_buf, self.extras
