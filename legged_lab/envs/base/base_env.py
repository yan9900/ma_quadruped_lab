from legged_lab.envs.base.base_env_config import BaseEnvCfg
import torch
from rsl_rl.env import VecEnv
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene
from isaaclab.assets.articulation import Articulation
from legged_lab.envs.env_utils.scene import SceneCfg
import numpy as np
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.envs.mdp.events import randomize_rigid_body_material, randomize_rigid_body_mass, reset_joints_by_scale, reset_root_state_uniform, push_by_setting_velocity
from isaaclab.managers import RewardManager
from isaaclab.utils.buffers import CircularBuffer


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, hedless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.hedless = hedless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_maneger = RewardManager(self.cfg.reward, self)

        self._init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.apply_domain_random_at_start(env_ids)
        self.reset(env_ids)

    def _init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names)
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.actions

        actor_obs = torch.cat([
            ang_vel * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            command * self.obs_scales.commands,
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            action * self.obs_scales.actions
        ], dim=-1,
        )

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - self.cfg.normalization.height_scan_offset)
            actor_obs = torch.cat([actor_obs, height_scan * self.obs_scales.height_scan], dim=-1)

        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        critic_obs = torch.cat([
            actor_obs,
            robot.data.root_lin_vel_b,
            feet_contact
        ], dim=-1)

        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = self.compute_current_observations()
        if self.add_noise:
            actor_obs = self.add_noise_to_obs(actor_obs)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        self.actor_obs_buffer.append(actor_obs)
        self.critic_obs_buffer.append(critic_obs)
        self.extras["observations"] = {"critic": self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)}

    def add_noise_to_obs(self, obs):
        obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        return obs

    def get_observations(self):
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self.compute_observations()
        return self.actor_obs_buffer.buffer.reshape(self.num_envs, -1), self.extras

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        reset_joints_by_scale(
            env=self,
            env_ids=env_ids,
            position_range=self.cfg.domain_rand.reset_robot_joints.params["position_range"],
            velocity_range=self.cfg.domain_rand.reset_robot_joints.params["velocity_range"],
            asset_cfg=self.robot_cfg,
        )
        reset_root_state_uniform(
            env=self,
            env_ids=env_ids,
            pose_range=self.cfg.domain_rand.reset_robot_base.params["pose_range"],
            velocity_range=self.cfg.domain_rand.reset_robot_base.params["velocity_range"],
            asset_cfg=self.robot_cfg,
        )

        self.command_generator.reset(env_ids)
        reward_extras = self.reward_maneger.reset(env_ids)
        self.extras['logs'] = reward_extras
        self.extras["time_outs"] = self.time_out_buf

        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.last_actions[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

    def step(self, actions: torch.Tensor):
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()

        cliped_actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.hedless:
            self.sim.render()

        self.post_physics_step()

        return self.actor_obs_buffer.buffer.reshape(self.num_envs, -1), self.reward_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.post_physics_step_callback()

        self.check_termination()
        self.reward_buf = self.reward_maneger.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)
        self.compute_observations()

    def post_physics_step_callback(self):
        self.command_generator.compute(self.step_dt)
        if self.cfg.domain_rand.push_robot.enable:
            env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.push_robot.push_interval_s / self.step_dt) == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) != 0:
                push_by_setting_velocity(
                    env=self,
                    env_ids=env_ids,
                    velocity_range=self.cfg.domain_rand.push_robot.params["velocity_range"],
                    asset_cfg=self.robot_cfg,
                )

    def check_termination(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        self.reset_buf = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1,), dim=1,)[0] > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def apply_domain_random_at_start(self, env_ids):
        if self.cfg.domain_rand.randomize_robot_friction.enable:
            self.cfg.domain_rand.randomize_robot_friction.params["asset_cfg"] = (self.robot_cfg)
            rand_rb_material = randomize_rigid_body_material(self.cfg.domain_rand.randomize_robot_friction, self)
            rand_rb_material(
                env=self,
                env_ids=env_ids,
                static_friction_range=self.cfg.domain_rand.randomize_robot_friction.params["static_friction_range"],
                dynamic_friction_range=self.cfg.domain_rand.randomize_robot_friction.params["dynamic_friction_range"],
                restitution_range=self.cfg.domain_rand.randomize_robot_friction.params["restitution_range"],
                num_buckets=self.cfg.domain_rand.randomize_robot_friction.params["num_buckets"],
                asset_cfg=self.robot_cfg
            )

        if self.cfg.domain_rand.add_rigid_body_mass.enable:
            robot_cfg = SceneEntityCfg(name="robot", body_names=self.cfg.domain_rand.add_rigid_body_mass.params["body_names"])
            robot_cfg.resolve(self.scene)
            randomize_rigid_body_mass(
                env=self,
                env_ids=env_ids,
                asset_cfg=robot_cfg,
                mass_distribution_params=self.cfg.domain_rand.add_rigid_body_mass.params["mass_distribution_params"],
                operation=self.cfg.domain_rand.add_rigid_body_mass.params["operation"],
            )

    def init_obs_buffer(self):
        actor_obs, _ = self.compute_current_observations()
        noise_vec = torch.zeros_like(actor_obs[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = (noise_scales.projected_gravity * noise_level * self.obs_scales.projected_gravity)
        noise_vec[6:9] = 0
        noise_vec[9 : 9 + self.num_actions] = (noise_scales.joint_pos * noise_level * self.obs_scales.joint_pos)
        noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (noise_scales.joint_vel * noise_level * self.obs_scales.joint_vel)
        noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
        if self.cfg.scene.height_scanner.enable_height_scan:
            noise_vec[9 + self.num_actions * 3 :] = (noise_scales.height_scan * noise_level * self.obs_scales.height_scan)
        self.noise_scale_vec = noise_vec

        self.actor_obs_buffer = CircularBuffer(max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device)
        self.critic_obs_buffer = CircularBuffer(max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device)
