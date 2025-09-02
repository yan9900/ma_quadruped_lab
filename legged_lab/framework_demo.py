#!/usr/bin/env python3
"""
LeggedLab 框架核心概念演示
这个脚本展示了框架的关键设计模式
"""

import torch
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Dict, Any

# ===========================
# 1. 配置系统演示
# ===========================

@dataclass
class BaseConfig:
    """基础配置 - 所有任务共享"""
    device: str = "cuda:0"
    num_envs: int = 4
    max_episode_steps: int = 1000

@dataclass 
class FallRecoveryConfig(BaseConfig):
    """Fall Recovery 特定配置 - 继承并扩展"""
    max_episode_steps: int = 1600  # 重写：更长时间用于恢复
    recovery_reward_weight: float = 15.0  # 新增：恢复奖励权重
    
    def __post_init__(self):
        print(f"Fall Recovery Config: {self.max_episode_steps}s episodes, reward weight: {self.recovery_reward_weight}")

# ===========================
# 2. 观测系统演示
# ===========================

class MockRobot:
    """模拟机器人状态"""
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        # 模拟多个机器人的状态
        self.joint_pos = torch.randn(num_envs, 12)  # 12个关节
        self.root_orientation = torch.randn(num_envs, 4)  # 四元数
        self.angular_vel = torch.randn(num_envs, 3)
        
    def get_observations_old_way(self):
        """传统方式：返回简单张量"""
        obs = torch.cat([
            self.joint_pos,
            self.root_orientation, 
            self.angular_vel
        ], dim=-1)
        return obs  # Shape: [num_envs, 19]
    
    def get_observations_new_way(self):
        """新方式：返回 TensorDict"""
        actor_obs = torch.cat([
            self.joint_pos,
            self.root_orientation,
        ], dim=-1)
        
        critic_obs = torch.cat([
            actor_obs,
            self.angular_vel,  # Critic 需要额外信息
        ], dim=-1)
        
        return TensorDict({
            "policy": actor_obs,
            "critic": critic_obs
        }, batch_size=[self.num_envs])

# ===========================
# 3. 奖励设计演示
# ===========================

class RewardCalculator:
    """奖励计算器演示"""
    
    @staticmethod
    def upward_reward(orientations: torch.Tensor, weight: float = 15.0) -> torch.Tensor:
        """计算向上奖励"""
        # 假设 orientation 是四元数 [w, x, y, z]
        # 向上向量应该是 [0, 0, 1]
        gravity_direction = torch.tensor([0., 0., -1.]).to(orientations.device)
        
        # 简化版本：使用 z 分量作为"向上"指标
        upward_component = torch.abs(orientations[:, 3])  # z 分量
        reward = upward_component * weight
        
        return reward
    
    @staticmethod
    def joint_smoothness_penalty(joint_velocities: torch.Tensor, weight: float = -0.01) -> torch.Tensor:
        """关节平滑度惩罚"""
        return torch.sum(joint_velocities ** 2, dim=-1) * weight

# ===========================
# 4. 运行演示
# ===========================

def demonstrate_framework():
    """演示框架核心概念"""
    
    print("=" * 60)
    print("LeggedLab 框架核心概念演示")
    print("=" * 60)
    
    # 1. 配置系统演示
    print("\n1. 配置系统演示：")
    base_cfg = BaseConfig()
    fall_cfg = FallRecoveryConfig()
    print(f"基础配置：{base_cfg.max_episode_steps}s")
    print(f"Fall Recovery：{fall_cfg.max_episode_steps}s (继承并重写)")
    
    # 2. 观测系统演示
    print("\n2. 观测系统演示：")
    robot = MockRobot(num_envs=4)
    
    # 传统方式
    old_obs = robot.get_observations_old_way()
    print(f"传统观测格式：{old_obs.shape}")
    
    # 新方式 (TensorDict)
    new_obs = robot.get_observations_new_way()
    print(f"TensorDict 观测：")
    print(f"  - Policy观测：{new_obs['policy'].shape}")
    print(f"  - Critic观测：{new_obs['critic'].shape}")
    print(f"  - 支持设备转移：{type(new_obs.to('cpu'))}")
    
    # 3. 奖励计算演示
    print("\n3. 奖励计算演示：")
    calc = RewardCalculator()
    
    # 模拟不同姿态的机器人
    upright_robot = torch.tensor([[1.0, 0.0, 0.0, 0.1],   # 接近直立
                                  [1.0, 0.0, 0.0, 0.8],   # 很直立
                                  [0.0, 1.0, 0.0, 0.0],   # 完全倒立
                                  [0.7, 0.7, 0.0, 0.0]])  # 侧躺
    
    upward_rewards = calc.upward_reward(upright_robot, weight=15.0)
    print("不同姿态的upward奖励：")
    for i, reward in enumerate(upward_rewards):
        states = ["接近直立", "很直立", "完全倒立", "侧躺"]
        print(f"  机器人{i} ({states[i]}): {reward:.2f}")
    
    # 4. 批量处理优势演示
    print("\n4. 批量处理效率演示：")
    num_envs = 1000
    large_robot = MockRobot(num_envs)
    
    import time
    # TensorDict 方式
    start = time.time()
    obs_dict = large_robot.get_observations_new_way()
    actions = torch.randn(num_envs, 12)  # 模拟策略输出
    tensor_time = time.time() - start
    
    print(f"处理{num_envs}个环境：{tensor_time*1000:.2f}ms")
    print(f"内存使用：Policy obs {obs_dict['policy'].numel() * 4 / 1024 / 1024:.2f}MB")
    
    print("\n=" * 60)
    print("框架核心优势总结：")
    print("1. 配置继承减少重复代码")
    print("2. TensorDict 提高内存效率") 
    print("3. 批量处理提升计算性能")
    print("4. 模块化设计便于扩展")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_framework()
