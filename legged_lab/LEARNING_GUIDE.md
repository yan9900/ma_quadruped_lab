# Legged Lab 技术深度学习计划

## 🎯 技术掌握目标

### 面试中可能遇到的问题类型

#### 1. 强化学习基础问题
**面试官问："为什么选择PPO算法？"**
**你的回答应该包含：**
- PPO是策略梯度方法，适合连续动作空间
- 具有重要性采样和剪切机制，训练稳定
- 在机器人控制任务中表现优秀
- 相比TRPO计算更高效

#### 2. 奖励设计问题
**面试官问："如何设计奖励函数让机器人学会走路？"**
**你的回答应该展示：**
```python
# 奖励函数设计原则
reward_components = {
    "tracking": "跟踪目标速度",
    "stability": "保持身体稳定",
    "energy": "最小化能耗",
    "safety": "避免危险行为",
    "shaping": "引导学习过程"
}

# 具体实现例子
def track_lin_vel_xy_exp(env, std=0.5):
    """速度跟踪奖励 - 指数形式更smooth"""
    vel_error = torch.norm(cmd_vel - actual_vel, dim=1)
    return torch.exp(-vel_error / std**2)
```

#### 3. 技术实现问题
**面试官问："如何处理机器人的动作延迟？"**
**你的回答应该提到：**
```python
# 动作延迟缓冲器
self.action_buffer = DelayBuffer(
    max_delay=self.cfg.domain_rand.action_delay.params["max_delay"],
    num_envs=self.num_envs,
    device=self.device
)
# 模拟真实机器人的通信延迟
delayed_action = self.action_buffer.compute(action)
```

## 📖 深度学习路线

### Week 1-2: 强化学习理论
**必读材料：**
1. PPO论文：Proximal Policy Optimization Algorithms
2. Isaac Lab文档：Environment Design Principles
3. RSL-RL库：Policy Implementation

**实践任务：**
```bash
# 1. 跑通基础训练流程
python scripts/train.py --task go2_flat --num_envs 512

# 2. 分析训练日志
# 查看 wandb 或 tensorboard 输出
# 理解各项奖励的变化曲线

# 3. 修改简单配置
# 改变环境数量、学习率等参数
# 观察对训练的影响
```

### Week 3: 环境设计深入
**核心文件分析：**
```python
# base_env.py - 环境主逻辑
class BaseEnv(VecEnv):
    def step(self, actions):
        # 1. 动作处理和延迟模拟
        # 2. 物理仿真步进
        # 3. 观察获取
        # 4. 奖励计算
        # 5. 终止条件检查
        pass
    
    def get_observations(self):
        # 传感器数据融合
        # 历史信息缓存
        # 噪声添加
        pass
```

**深入理解点：**
1. **观察空间设计**：关节角度、速度、IMU、地形扫描
2. **动作空间设计**：关节位置目标、PD控制参数
3. **物理仿真**：PhysX引擎、接触检测、碰撞处理

### Week 4: 算法与优化
**RSL-RL库深入：**
```python
# 策略网络结构
class ActorCritic:
    def __init__(self):
        self.actor = MLP([obs_dim, 256, 256, action_dim])
        self.critic = MLP([obs_dim, 256, 256, 1])
    
    def act(self, obs):
        # 策略输出动作分布
        return self.actor(obs)
    
    def evaluate(self, obs):
        # 价值函数估计
        return self.critic(obs)
```

## 🔍 代码走读清单

### 必须理解的核心函数

#### 1. 环境核心循环
```python
# base_env.py:step()
def step(self, actions: torch.Tensor):
    # 动作处理
    delayed_actions = self.action_buffer.compute(actions)
    
    # 应用动作到机器人
    self.robot.set_joint_position_target(delayed_actions)
    
    # 物理仿真步进
    for _ in range(self.cfg.sim.decimation):
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=self.physics_dt)
    
    # 获取观察
    obs = self.get_observations()
    
    # 计算奖励
    rewards = self.reward_manager.compute()
    
    # 检查终止条件
    dones = self.get_dones()
    
    return obs, rewards, dones, self.extras
```

#### 2. 奖励计算机制
```python
# rewards.py 中的关键函数
def track_lin_vel_xy_yaw_frame_exp(env, std: float):
    """速度跟踪奖励"""
    asset = env.scene["robot"]
    
    # 获取机器人当前速度（yaw坐标系）
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), 
        asset.data.root_lin_vel_w[:, :3]
    )
    
    # 计算与命令速度的误差
    vel_error = torch.sum(
        torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), 
        dim=1
    )
    
    # 指数形式的奖励（更平滑）
    return torch.exp(-vel_error / std**2)
```

#### 3. 地形生成系统
```python
# terrain_generator_cfg.py
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,  # 启用课程学习
    size=(8.0, 8.0),  # 地形大小
    sub_terrains={
        "pyramid_stairs_28": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,  # 占总地形的比例
            step_height_range=(0.0, 0.23),  # 台阶高度范围
            step_width=0.28,  # 台阶宽度
        ),
        # ... 其他7种地形类型
    }
)
```

## 🎪 实战练习清单

### 练习1：奖励函数调试
```python
# 任务：修改GO2的奖励权重，让它学会更好的步态
# 文件：envs/go2/go2_config.py

class Go2RewardCfg(RewardCfg):
    # 实验：调整这些权重
    track_lin_vel_xy_exp = RewTerm(weight=2.0)  # 原来3.0
    feet_air_time = RewTerm(weight=0.2)         # 原来0.1
    upward = RewTerm(weight=2.0)                # 原来1.0
```

### 练习2：添加新的奖励项
```python
# 任务：在rewards.py中添加能耗最小化奖励
def energy_minimization(env, asset_cfg=SceneEntityCfg("robot")):
    """最小化关节功率消耗"""
    asset = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    joint_vel = asset.data.joint_vel
    power = torch.abs(torques * joint_vel)
    return -torch.sum(power, dim=1)
```

### 练习3：地形自定义
```python
# 任务：创建新的地形类型
"custom_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
    proportion=0.1,
    grid_width=0.3,
    grid_height_range=(0.05, 0.2),
    platform_width=1.5
)
```

## 📝 面试准备清单

### 技术问题准备

#### 强化学习类
1. "解释PPO算法的核心思想"
2. "为什么在机器人控制中使用连续动作空间？"
3. "如何设计稀疏奖励vs密集奖励？"
4. "什么是值函数？它在训练中的作用？"

#### 机器人控制类  
1. "如何处理机器人动力学？"
2. "PD控制在这个项目中的作用？"
3. "如何保证训练的安全性？"
4. "域随机化的必要性？"

#### 系统设计类
1. "如何扩展到新的机器人？"
2. "并行环境的好处和挑战？"
3. "仿真到真实的差距如何处理？"
4. "性能优化的关键点？"

### 项目展示准备
```python
# 准备demo脚本
python scripts/play.py --task go2_rough --checkpoint best_model.pth --num_envs 64
```

## 🚀 进阶拓展

### 高级话题
1. **课程学习**：地形难度自动调节机制
2. **域随机化**：提高sim-to-real转换
3. **多模态控制**：结合视觉和本体感觉
4. **分层强化学习**：高层规划+低层控制

### 相关论文阅读
1. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
2. Learning Agile Robotic Locomotion Skills by Imitating Animals  
3. RMA: Rapid Motor Adaptation for Legged Robots
4. Learning Robust, Real-Time, Reactive Robotic Movement
