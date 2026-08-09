# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
# import RayCaster from isaaclab.sensors if you want to use unmodified version
from isaaclab.sensors import ContactSensor
from isaaclab.sim import PhysxCfg, SimulationContext
# self defined sensors
from legged_lab.sensors.camera import Camera
from legged_lab.sensors.ray_caster import RayCaster
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        # cfg.scene是在BaseEnvCfg里定义的所有场景相关参数
        # SceneCfg则是一个包装类，用来把BaseSceneCfg转换成isaaclab（InteractiveScene）能用的配置
        # 注意区分SceneCfg和SceneEntityCfg,SceneCfg关注的是场景整体配置，而SceneEntityCfg关注的是场景中的具体实体
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
        if self.cfg.scene.camera.enable_camera:
            # 使用 Isaac Lab 的 Camera 类
            self.front_camera: Camera = self.scene.sensors["front_camera"]

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
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self): 
        #实际控制过程中，action和obs都存在延时
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        
        # action buffer用来模仿控制延迟，这里只是初始化
        # action_buffer是一个delaybuffer实例
        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        # 随机初始化延迟时间
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # init obs buffer
        # SceneEntityCfg独立于BaseEnvCfg用来配置和解析场景中的实体（如机器人、传感器等）
        # robot_cfg, termination_contact_cfg, feet_cfg都是SceneEntityCfg实例
        # 创建一个配置对象，描述“我要操作名为 robot 的场景实体”
        # resolve()是将对应的body_names转化为body_ids（索引）
        
        # robot 不指定 body_names → 绑定 robot 下所有 body。
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        
        # contact_sensor 是一个实体，里面可以有多个 body，你可以用 body_names 精确指定关注的部分（如脚、传感器片段等）
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.init_obs_buffer()

    # 计算当前观测值
    # actor_obs包括
    """
    root的角速度 3
    root的投影重力 3
    机器人命令 来自 command generator 3
    joint的位置 做差的形式，减去默认关节位置 12
    joint的速度 做差的形式，减去默认关节速度 12
    机器人的最新动作 12
    """
    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            # 沿着最后一个维度拼接
            dim=-1,
        )

        # critic_obs比actor_obs多了 
        # root的线速度 和 feet_contact 3 + 4
        # feet_contact为bool类型，表示脚是否接触地面
        # net_contact_forces的形状为[num_envs, history_length, num_bodies, 3]
        # torch.norm的形状为[num_envs, history_length, num_feet]
        # torch.max(,dim=1)寻找历史时间步里，某个body上受力最大的那个时间步
        # 经过torch.max之后，变成了[num_envs, num_feet]
        # [0]表示取最大值，>0.5表示受力大于0.5则认为接触地面
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    # 与current的区别在于，compute_observations会把历史的obs也考虑进来
    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        #逐项相加，[-1,1] * noise_scale
        #critic obs是在actor obs的基础上加了root的线速度和feet_contact，所以也加上了噪音
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec
        
        # obs_buffer是一个circularbuffer实例
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # self.actor_obs_buffer.buffer 的 shape 是 [num_envs, history_length, obs_dim]，
        # 即每个环境有 history_length 步，每步有 obs_dim 个观测量。
        # reshape(self.num_envs, -1) 把每个环境的所有历史观测拼接成一个一维长向量，
        # 最终 shape 变成 [num_envs, history_length * obs_dim]
        # 这样做的目的是：让策略网络一次性接收所有历史观测信息，而不是只用当前帧。
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        # _w:world coordinate,[:,2]-> height, [nums_envs, 3]-> unsqueeze(1) -> [num_envs, 1]只保留高度信息
        # self.height_scanner.data.ray_hits_w 是 height_scanner 发射的所有激光射线击中点的世界坐标，
        # shape 通常为 [num_envs, num_rays, 3]，这里取整数索引，变成 [num_envs, num_rays]，只保留高度信息
        # 根据广播原理，二者可以相减
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # 这样加上的噪音，critic_obs不会受到影响
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        # 定义：self.clip_obs = self.cfg.normalization.clip_observations
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    # 重置环境
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # 每次 reset 时都会执行 self.extras["log"] = dict()，即新建一个空字典。
        # 这样可以保证每次重置时日志内容都是最新的，不会残留上一次的信息。
        # 后续会通过 .update() 方法往里面添加统计信息（如地形等级、奖励等）
        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            # 检查是否启用随机地形spawn
            if getattr(self.cfg.scene, 'enable_random_terrain_spawn', False):
                terrain_extras = self.randomize_terrain_spawn(env_ids)
                self.extras["log"].update(terrain_extras)
            elif self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        # 在环境 reset 时，确保所有状态和目标都已同步到仿真引擎，仿真世界和 Python 侧的数据保持一致。
        self.scene.write_data_to_sim()
        self.sim.forward()
        
    # newly added    
    def _pick_no_repeat(self, cells_tensor: torch.Tensor, n: int, count: int) -> torch.Tensor:
        """从 cells_tensor[:n] 中无重复采样 count 个；count > n 时循环整轮再取余量。"""
        if count <= n:
            return cells_tensor[torch.randperm(n, device=self.device)[:count]]
        parts = []
        full_rounds, remainder = divmod(count, n)
        for _ in range(full_rounds):
            parts.append(cells_tensor[torch.randperm(n, device=self.device)])
        if remainder:
            parts.append(cells_tensor[torch.randperm(n, device=self.device)[:remainder]])
        return torch.cat(parts, dim=0)

    def _sample_terrain_cells_priority(self, free_cells: list, busy_cells: list, num_samples: int):
        """优先从 free_cells（空闲格子）无重复采样，不足时再从 busy_cells 补充。

        free_cells : 当前无机器人的有效地形格子
        busy_cells : 已有机器人的有效地形格子（退而求其次）
        """
        n_free = len(free_cells)
        n_busy = len(busy_cells)

        if n_free == 0 and n_busy == 0:
            raise ValueError("No valid terrain cells available.")

        # 全在 free 内可以搞定
        if n_free >= num_samples:
            free_t = torch.tensor(free_cells, device=self.device)
            chosen = self._pick_no_repeat(free_t, n_free, num_samples)
            return chosen[:, 0], chosen[:, 1]

        parts = []
        # 先把所有 free 格子（打乱顺序）全放进去
        if n_free > 0:
            free_t = torch.tensor(free_cells, device=self.device)
            parts.append(free_t[torch.randperm(n_free, device=self.device)])

        remaining = num_samples - n_free
        # 用 busy 补充；若 busy 也没有则循环 free
        if n_busy > 0:
            busy_t = torch.tensor(busy_cells, device=self.device)
            parts.append(self._pick_no_repeat(busy_t, n_busy, remaining))
        else:
            free_t = torch.tensor(free_cells, device=self.device)
            parts.append(self._pick_no_repeat(free_t, n_free, remaining))

        chosen = torch.cat(parts, dim=0)
        return chosen[:, 0], chosen[:, 1]

    def randomize_terrain_spawn(self, env_ids):
        """随机选择地形tile作为spawn点（优化版：利用查找表）
        
        核心优化：利用初始化时已计算好的env_origins，通过查找而非重新计算坐标
        
        地形网格布局说明：
            - rows (levels): 难度等级维度，从0（最简单）到num_rows-1（最难）
            - cols (types): 地形类型维度，从0到num_cols-1（不同形态的地形）
            - terrain[row=i, col=j] 表示第i个难度等级的第j种地形类型
        
        实现思路：
            1. 随机选择目标terrain[level, type]
            2. 查找哪个env_id已经在该terrain上（而不是计算坐标！）
            3. 直接复制该env_id的origin（保证与初始化100%一致）
        
        Args:
            env_ids: 需要重置的环境索引列表
        
        Returns:
            extras: 包含地形统计信息的字典，用于日志记录
        """
        num_resets = len(env_ids)
        terrain_gen_cfg = self.scene.terrain.cfg.terrain_generator
        num_rows = terrain_gen_cfg.num_rows
        num_cols = terrain_gen_cfg.num_cols

        # 如果 cfg 有 grid_layout + spawn_tile_keys，只从允许的格子里随机
        grid_layout = getattr(terrain_gen_cfg, 'grid_layout', None)
        spawn_tile_keys = getattr(terrain_gen_cfg, 'spawn_tile_keys', None)
        if grid_layout is not None and spawn_tile_keys is not None:
            valid_cells = [
                (idx // num_cols, idx % num_cols)
                for idx, key in enumerate(grid_layout)
                if key in spawn_tile_keys
            ]
            if not valid_cells:
                raise ValueError(f"spawn_tile_keys={spawn_tile_keys} 在 grid_layout 中没有匹配的格子。")
        else:
            valid_cells = [(r, c) for r in range(num_rows) for c in range(num_cols)]

        # 计算当前仍占据 terrain 的 env（不在本次 reset 列表中）所占用的格子
        staying_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        staying_mask[env_ids] = False
        staying_levels = self.scene.terrain.terrain_levels[staying_mask].cpu().tolist()
        staying_types  = self.scene.terrain.terrain_types[staying_mask].cpu().tolist()
        occupied_cells = set(zip(staying_levels, staying_types))

        free_cells = [c for c in valid_cells if c not in occupied_cells]
        busy_cells = [c for c in valid_cells if c in occupied_cells]
        random_levels, random_types = self._sample_terrain_cells_priority(free_cells, busy_cells, num_resets)
        
        # === 直接使用 terrain_origins 原始坐标表 ===
        # 
        # TerrainImporter 在初始化时保存了所有地形块的原点坐标：
        #   terrain_origins[level, type] = [x, y, z]  # shape: (num_rows, num_cols, 3)
        # 
        # 这是最准确的坐标来源，包含：
        #   - 正确的 x, y 坐标（考虑了 size 和 spacing）
        #   - 正确的 z 高度（如果地形有高度变化）
        #
        # 直接查表，不需要找其他 env 或手动计算！
        
        # 检查 terrain_origins 是否存在
        if self.scene.terrain.terrain_origins is not None:
            # 直接从原始地形坐标表查找
            for i in range(num_resets):
                # 需要从pytorch tensor转化成 Python int 才能索引
                target_level = random_levels[i].item()
                target_type = random_types[i].item()
                
                # 直接复制对应的origin坐标
                self.scene.env_origins[env_ids[i]] = self.scene.terrain.terrain_origins[target_level, target_type].clone()
                
                # 更新地形索引
                self.scene.terrain.terrain_levels[env_ids[i]] = target_level
                self.scene.terrain.terrain_types[env_ids[i]] = target_type
        else:
            # terrain_origins 不存在（不应该发生，但保险起见）
            print(f"[Warning] terrain_origins is None, using fallback calculation")
            for i in range(num_resets):
                target_level = random_levels[i]
                target_type = random_types[i]
                
                self.scene.terrain.terrain_levels[env_ids[i]] = target_level
                self.scene.terrain.terrain_types[env_ids[i]] = target_type
                self._update_env_origins_from_terrain_indices(env_ids[i:i+1])
        
        # 调试输出：显示地形切换信息
        if len(env_ids) > 0:
            origins_info = [self.scene.env_origins[env_ids[i]].cpu().tolist() for i in range(min(3, num_resets))]
            print(f"[Random Spawn] Reset {len(env_ids)} envs → terrains: {list(zip(random_levels.cpu().tolist()[:3], random_types.cpu().tolist()[:3]))}, origins: {origins_info}")
        
        extras = {
            "Terrain/random_spawn_enabled": 1.0,
            "Terrain/avg_level": torch.mean(random_levels.float()),
            "Terrain/avg_type": torch.mean(random_types.float())
        }
        return extras
    
    def _update_env_origins_from_terrain_indices(self, env_ids):
        """根据 terrain_levels 和 terrain_types 计算新的 env_origins
        
        坐标系统说明：
            地形网格在世界坐标系中的布局：
            - x轴方向：对应 cols (地形类型)
            - y轴方向：对应 rows (难度等级)
            - z轴方向：地形高度
            
            terrain[row=i, col=j] 的世界坐标原点：
                x = j * (tile_width + spacing)
                y = i * (tile_height + spacing)
                z = 地形起始高度（从高度图采样或默认为0）
        
        Args:
            env_ids: 需要更新origin的环境索引列表
        """
        terrain_cfg = self.scene.terrain.cfg.terrain_generator
        size = terrain_cfg.size  # 单个terrain tile的尺寸，例如 (8.0, 8.0)
        spacing = self.cfg.scene.env_spacing  # env之间的间距，例如 2.5
        
        # 获取对应env的terrain索引
        levels = self.scene.terrain.terrain_levels[env_ids]  # 行索引（难度）
        types = self.scene.terrain.terrain_types[env_ids]    # 列索引（类型）
        
        # 计算新的origin坐标（世界坐标系）
        new_origins = torch.zeros(len(env_ids), 3, device=self.device)
        new_origins[:,0] = types * (size[0] + spacing)
        new_origins[:,1] = levels * (size[1] + spacing)
        new_origins[:,2] = 0.0
        
        self.scene.env_origins[env_ids] = new_origins
    
    
    # 环境步 对应 step_dt
    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)
        # delayed_actions = torch.zeros_like(actions)  
        

        # actions完全作用于joints上
        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos
        # print("body_names:", self.robot.data.body_names)
        #print(f"current base height: {self.robot.data.root_pos_w[0,2].item():.4f}", end='\r')

        # 每个环境步执行 decimation 次物理仿真步，每次用同一个动作。
        # step_dt = self.cfg.sim.dt * self.cfg.sim.decimation
        # where self.cfg.sim.dt 是物理仿真的时间步长(self.physics_dt)
        has_camera = hasattr(self.cfg.scene, 'camera') and getattr(self.cfg.scene.camera, 'enable_camera', False)

        for i in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            # 物理步始终用 render=False，避免 sim.step(render=True) 在 decimation 中间
            # 触发 app.update()，后者可能通过 omni.physx 回调额外推进一次物理步，
            # 导致有效物理步数为 decimation+1，破坏控制频率（训练 50Hz → 实际 ~40Hz）。
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # 物理循环结束后统一渲染：
        # - 有相机时必须渲染以更新 Replicator annotators（相机数据比物理晚一个控制步，但对 WM 训练可接受）
        # - 非 headless 时渲染 GUI viewport
        if has_camera or not self.headless:
            self.sim.render()

        # 一个episode一般包含多个环境步，每执行一次环境步，当前episode的步数+1
        # command的更新频率也是dt
        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)

        # 与 reset 区分，reset 事件只在环境重置时触发，interval 事件则在每个 step 都可以触发，
        # 适合需要持续/定期处理的逻辑。

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # reset_buf和time_out_buf都是bool类型的张量，表示哪些环境需要重置
        self.reset_buf, self.time_out_buf = self.check_reset()
        # 必须在 reset() 之前更新 extras["time_outs"]，因为 reset() 在 env_ids 为空时
        # 会 early-return（不执行 self.extras["time_outs"] = ...），导致 extras 里残留
        # 上一步的 time_out_buf 引用（指向旧 tensor），使脚本误判已 reset 的 env 仍在超时。
        self.extras["time_outs"] = self.time_out_buf
        # 计算奖励
        reward_buf = self.reward_manager.compute(self.step_dt)
        # [False, True, False, True]
        # nonzero() 返回所有非零元素的索引，每个索引一行。
        # tensor([[1], [3]])
        # 用 .flatten() 后变成 shape [N] 的一维向量，里面是所有非零元素的索引
        # tensor([1, 3])
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)
        
        # actor_obs, critic_obs = self.compute_observations()
        # self.extras["observations"] = {"critic": critic_obs}

        obs_dict = self.get_observations()

        return obs_dict, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # 判定是否重置的条件：
        # 注意，这里的力并没有做过归一化，直接与1.0比较
        # 某个 termination_contact_cfg.body_ids 上的受力超过 1.0 就重置 -> 终止(或者说重置)条件
        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        # reset_buf和time_out_buf都是bool类型的张量，表示哪些环境需要重置
        # 此时reset_buf只表示受力过大的重置
        # 对于每个环境，只要满足任一重置条件，就会被标记为需要重置
        # 经过or运算后，reset_buf表示所有重置条件
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # 0-2 body angular velocity
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            # 3-5 body projected gravity
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            # command
            noise_vec[6:9] = 0
            # joint positions
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            # joint velocities
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            # latest action
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        # xy平面上，机器人当前位置到当前地形中心点的距离
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        # 这里的"升高/降低"不是指 z 方向的物理高度，而是指 terrain curriculum（地形课程学习）中，
        # 机器人被分配到更难/更易的地形区域
        # 行走超过地形宽度的一半，就提升地形难度等级
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        # 小于半个episode可以走的距离就降低地形难度等级
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        # 非move_up同时move_down才move_down
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        
        # 保持向后兼容
        self.extras["observations"] = {"critic": critic_obs}
        
        # Create a TensorDict which supports both dict access and .to() method
        # 包含 policy (actor) 和 critic 观测
        obs_dict = TensorDict({
            "policy": actor_obs,
            "critic": critic_obs
        }, batch_size=torch.Size([self.num_envs]))
        
        return obs_dict
    
    # def get_observations(self):
    #     actor_obs, critic_obs = self.compute_observations()
    #     self.extras["observations"] = {"critic": critic_obs}
    #     return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
