# Legged Lab 项目架构分析

## 🏗️ 项目整体架构

```
legged_lab/
├── envs/                    # 环境定义
│   ├── base/               # 基础环境类
│   │   ├── base_env.py     # 核心环境逻辑
│   │   └── base_config.py  # 配置类定义
│   ├── go2/               # GO2机器人配置
│   ├── g1/                # G1机器人配置
│   ├── h1/                # H1机器人配置
│   └── gr2/               # GR2机器人配置
├── assets/                 # 机器人资产文件
│   ├── unitree/           # Unitree机器人模型
│   └── fftai/             # FFTAI机器人模型
├── mdp/                   # 马尔可夫决策过程组件
│   └── rewards.py         # 奖励函数实现
├── terrains/              # 地形生成
│   ├── terrain_generator_cfg.py  # 地形配置
│   └── ray_caster.py      # 地形感知
├── scripts/               # 训练和测试脚本
│   ├── train.py          # 训练脚本
│   └── play.py           # 测试脚本
└── utils/                 # 工具函数
    ├── task_registry.py   # 任务注册
    └── cli_args.py        # 命令行参数
```

## 🔄 数据流向

### 训练流程
```
1. 环境初始化
   ├── 机器人模型加载 (assets/)
   ├── 地形生成 (terrains/)
   └── 传感器配置

2. 强化学习循环
   ├── 观察获取 (BaseEnv.get_observations())
   ├── 动作选择 (PPO Policy)
   ├── 环境步进 (BaseEnv.step())
   ├── 奖励计算 (mdp/rewards.py)
   └── 策略更新 (RSL-RL)

3. 训练管理
   ├── 事件管理 (EventManager)
   ├── 域随机化 (DomainRandomization)
   └── 课程学习 (Curriculum Learning)
```

## 🎯 核心类关系

```python
BaseEnv (base_env.py)
├── 继承自 VecEnv (RSL-RL)
├── 包含 InteractiveScene (场景管理)
├── 包含 RewardManager (奖励管理)
├── 包含 EventManager (事件管理)
└── 包含 CommandGenerator (命令生成)

BaseEnvCfg (base_env_config.py)
├── scene: BaseSceneCfg (场景配置)
├── robot: RobotCfg (机器人配置)
├── reward: RewardCfg (奖励配置)
└── domain_rand: DomainRandCfg (域随机化)
```

## 📊 关键配置系统

### 环境配置层次
```
BaseEnvCfg (基础配置)
├── Go2FlatEnvCfg (GO2平坦地形)
│   └── Go2RoughEnvCfg (GO2复杂地形)
├── H1FlatEnvCfg (H1平坦地形)
│   └── H1RoughEnvCfg (H1复杂地形)
└── ...其他机器人配置
```

### 奖励系统架构
```python
RewardCfg:
├── track_lin_vel_xy_exp: 速度跟踪
├── base_height_l2: 高度控制
├── feet_air_time: 足端空中时间
├── upward: 向上方向奖励
└── ...20+个奖励项
```

## 🤖 支持的机器人

| 机器人 | 类型 | 自由度 | 特点 |
|--------|------|--------|------|
| GO2 | 四足 | 12DOF | 小型，适合研究 |
| G1 | 人形 | 37DOF | 双足行走 |
| H1 | 人形 | 19DOF | 轻量级人形 |
| GR2 | 四足 | 12DOF | 重载四足 |

## 🎮 使用方式

### 训练命令
```bash
python scripts/train.py --task go2_rough --num_envs 4096
```

### 测试命令
```bash
python scripts/play.py --task go2_rough --checkpoint path/to/model.pth
```

## 🔧 自定义扩展点

1. **新机器人**: 添加到 `assets/` 和 `envs/`
2. **新奖励**: 在 `mdp/rewards.py` 中实现
3. **新地形**: 在 `terrains/` 中配置
4. **新任务**: 在 `__init__.py` 中注册

## 📈 性能监控

- Wandb集成用于实验跟踪
- TensorBoard支持
- 实时奖励曲线监控
- 地形难度自动调节
