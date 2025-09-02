#!/usr/bin/env python3
"""
完整的机器人注册指南
以添加一个新机器人 "DogBot" 为例
"""

# =========================
# 步骤 1: 创建机器人资产配置
# 位置: legged_lab/assets/dogbot/dogbot.py
# =========================

DOGBOT_CFG_EXAMPLE = """
# legged_lab/assets/dogbot/dogbot.py

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

DOGBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/dogbot/dogbot.usd",  # 你的机器人USD文件
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),  # 初始位置
        joint_pos={
            ".*_hip_joint": 0.0,     # 髋关节初始角度
            ".*_thigh_joint": 0.8,   # 大腿关节初始角度  
            ".*_calf_joint": -1.6,   # 小腿关节初始角度
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,    # 扭矩限制
            velocity_limit=21.0,  # 速度限制
            stiffness=25.0,       # 刚度
            damping=0.5,          # 阻尼
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""

# =========================
# 步骤 2: 创建环境配置
# 位置: legged_lab/envs/dogbot/dogbot_config.py
# =========================

DOGBOT_CONFIG_EXAMPLE = """
# legged_lab/envs/dogbot/dogbot_config.py

from __future__ import annotations
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.assets.dogbot import DOGBOT_CFG

# 机器人常量
BASE_LINK_NAME = "base"
FOOT_REGEX = r".*_calf"

@configclass  
class DogBotRewardCfg:
    \"\"\"DogBot 奖励配置\"\"\"
    # 移动跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})
    
    # 姿态奖励
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # 基础惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

@configclass
class DogBotFlatEnvCfg(BaseEnvCfg):
    \"\"\"DogBot 平地环境配置\"\"\"
    
    def __post_init__(self):
        super().__post_init__()
        
        # 设置机器人和奖励
        self.scene.robot = DOGBOT_CFG
        self.reward = DogBotRewardCfg()
        
        # 环境设置
        self.scene.terrain_type = "plane"
        self.scene.max_episode_length_s = 20.0
        
        # 机器人特定设置
        self.robot.feet_body_names = [FOOT_REGEX]
        self.robot.base_link_name = BASE_LINK_NAME

@configclass
class DogBotFlatAgentCfg(BaseAgentCfg):
    \"\"\"DogBot 智能体配置\"\"\"
    experiment_name: str = "dogbot_flat"
    wandb_project: str = "dogbot_experiments"
"""

# =========================
# 步骤 3: 注册到系统
# 位置: legged_lab/envs/__init__.py
# =========================

REGISTRATION_EXAMPLE = """
# 在 legged_lab/envs/__init__.py 中添加:

# 导入新机器人配置
from legged_lab.envs.dogbot.dogbot_config import (
    DogBotFlatEnvCfg,
    DogBotFlatAgentCfg,
)

# 注册新任务
task_registry.register("dogbot_flat", BaseEnv, DogBotFlatEnvCfg(), DogBotFlatAgentCfg())
"""

def demonstrate_registration_process():
    """演示完整的注册过程"""
    
    print("=" * 80)
    print("🤖 新机器人注册完整指南")
    print("=" * 80)
    
    print("\n📁 文件结构:")
    print("""
legged_lab/
├── assets/
│   └── dogbot/                    # 新建：机器人资产目录
│       ├── __init__.py
│       ├── dogbot.py             # 机器人物理配置
│       └── dogbot.usd            # 机器人模型文件
├── envs/
│   ├── dogbot/                   # 新建：机器人环境目录  
│   │   ├── __init__.py
│   │   └── dogbot_config.py      # 环境和奖励配置
│   └── __init__.py               # 修改：添加注册代码
└── scripts/
    ├── train.py                  # 无需修改
    └── play.py                   # 无需修改
""")
    
    print("\n🔧 详细步骤:")
    steps = [
        ("1. 创建机器人资产配置", "定义物理属性、关节、执行器等"),
        ("2. 创建奖励配置类", "设计任务特定的奖励函数"),
        ("3. 创建环境配置类", "整合机器人、奖励、场景设置"),
        ("4. 创建智能体配置类", "定义训练相关的超参数"),
        ("5. 注册到系统", "一行代码注册新任务"),
        ("6. 测试运行", "python train.py --task dogbot_flat")
    ]
    
    for i, (step, description) in enumerate(steps, 1):
        print(f"{i}. {step}")
        print(f"   └─ {description}")
    
    print("\n" + "=" * 50)
    print("📋 实际的注册代码示例")
    print("=" * 50)
    
    print("\n1️⃣ 机器人资产配置 (assets/dogbot/dogbot.py):")
    print(DOGBOT_CFG_EXAMPLE[:500] + "...\n# (完整代码请参考GO2配置)")
    
    print("\n2️⃣ 环境配置 (envs/dogbot/dogbot_config.py):")
    print(DOGBOT_CONFIG_EXAMPLE[:500] + "...\n# (完整代码请参考GO2配置)")
    
    print("\n3️⃣ 注册代码 (envs/__init__.py):")
    print(REGISTRATION_EXAMPLE)
    
    print("\n" + "=" * 80)
    print("💡 关键要点:")
    print("=" * 80)
    
    key_points = [
        "🔗 所有配置类都继承自基类，确保兼容性",
        "📝 配置驱动：无需修改核心代码，只需配置",
        "🔄 自动发现：注册后立即可用 --task 参数调用",
        "⚡ 并行训练：自动支持多环境并行训练", 
        "🎯 模块化：每个机器人独立配置，便于维护",
        "🔧 可扩展：轻松添加新任务类型（rough, recovery等）"
    ]
    
    for point in key_points:
        print(point)
    
    print(f"\n" + "=" * 80)
    print("🚀 你的 go2_fall_recovery 就是这样注册的！")
    print("=" * 80)
    
    your_registration = """
# 你已经完成的注册:
task_registry.register(
    "go2_fall_recovery",           # 任务名称
    BaseEnv,                       # 环境类  
    Go2FallRecoveryFlatEnvCfg(),   # 环境配置 (你创建的)
    Go2FallRecoveryAgentCfg()      # 智能体配置 (你创建的)
)

# 结果: python train.py --task go2_fall_recovery 就能工作！
"""
    
    print(your_registration)

if __name__ == "__main__":
    demonstrate_registration_process()
