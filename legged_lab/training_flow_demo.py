#!/usr/bin/env python3
"""
训练流程深度剖析演示
展示一个完整训练步骤的内部工作原理
"""

import torch
from tensordict import TensorDict

class TrainingStepDemo:
    """模拟一个完整的训练步骤"""
    
    def __init__(self, num_envs=4, num_actions=12):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.step_count = 0
        
    def simulate_training_step(self):
        """模拟完整的训练步骤"""
        
        print(f"\n===== Training Step {self.step_count} =====")
        
        # 1. 环境产生观测
        print("1. 环境状态 → 观测")
        robot_state = {
            "joint_positions": torch.randn(self.num_envs, 12),
            "orientation": torch.randn(self.num_envs, 4),  # 四元数
            "angular_velocity": torch.randn(self.num_envs, 3),
        }
        
        # 模拟不同的机器人状态
        orientations = robot_state["orientation"]
        for i in range(self.num_envs):
            if torch.norm(orientations[i]) > 0:
                orientations[i] = orientations[i] / torch.norm(orientations[i])  # 标准化四元数
        
        # 组合观测（类似 compute_current_observations）
        actor_obs = torch.cat([
            robot_state["joint_positions"],
            robot_state["orientation"]
        ], dim=-1)
        
        critic_obs = torch.cat([
            actor_obs,
            robot_state["angular_velocity"]
        ], dim=-1)
        
        obs_dict = TensorDict({
            "policy": actor_obs,
            "critic": critic_obs
        }, batch_size=[self.num_envs])
        
        print(f"   观测形状: Policy={obs_dict['policy'].shape}, Critic={obs_dict['critic'].shape}")
        
        # 2. 策略产生动作
        print("2. 策略网络 → 动作")
        # 模拟策略网络（简化的前向传播）
        with torch.no_grad():
            # 模拟 Actor 网络
            hidden = torch.relu(torch.matmul(obs_dict["policy"], torch.randn(16, 64)))
            hidden = torch.relu(torch.matmul(hidden, torch.randn(64, 32)))
            actions = torch.tanh(torch.matmul(hidden, torch.randn(32, self.num_actions)))  # 输出动作
            
        print(f"   动作形状: {actions.shape}, 范围: [{actions.min():.2f}, {actions.max():.2f}]")
        
        # 3. 环境执行动作并计算奖励
        print("3. 动作执行 → 奖励计算")
        
        # 模拟奖励计算（类似你的 Go2FallRecoveryRewardCfg）
        upward_reward = self.calculate_upward_reward(robot_state["orientation"]) * 15.0
        joint_penalty = -0.01 * torch.sum(robot_state["angular_velocity"] ** 2, dim=-1)
        total_reward = upward_reward + joint_penalty
        
        print(f"   奖励组成:")
        print(f"     - Upward奖励: {upward_reward.mean():.3f} (权重15.0)")
        print(f"     - 关节惩罚: {joint_penalty.mean():.3f} (权重-0.01)")
        print(f"     - 总奖励: {total_reward.mean():.3f}")
        
        # 4. 检查终止条件
        print("4. 终止条件检查")
        # 简化的终止逻辑
        robot_heights = torch.abs(robot_state["orientation"][:, 3])  # z分量
        terminated = robot_heights < 0.1  # 如果z太小认为机器人倒地严重
        
        print(f"   终止的环境: {terminated.sum().item()}/{self.num_envs}")
        
        # 5. 模拟重置
        if terminated.any():
            print("5. 环境重置 (域随机化)")
            reset_envs = terminated.nonzero().flatten()
            print(f"   重置环境: {reset_envs.tolist()}")
            
            # 模拟域随机化重置
            for env_id in reset_envs:
                # 随机初始姿态（类似你的 reset_base 事件）
                random_orientation = torch.randn(4)
                random_orientation = random_orientation / torch.norm(random_orientation)
                robot_state["orientation"][env_id] = random_orientation
                print(f"     环境{env_id}: 新随机姿态 {random_orientation[:2].tolist()}")
        
        self.step_count += 1
        return obs_dict, actions, total_reward, terminated
    
    def calculate_upward_reward(self, orientations: torch.Tensor) -> torch.Tensor:
        """计算机器人向上的奖励（简化版本）"""
        # 使用z分量作为"向上"程度的指标
        upward_component = torch.abs(orientations[:, 3])
        return upward_component

def demonstrate_training_flow():
    """演示完整的训练流程"""
    
    print("LeggedLab 训练流程深度解析")
    print("=" * 60)
    
    # 创建演示实例
    demo = TrainingStepDemo(num_envs=4)
    
    # 运行几个训练步骤
    for step in range(3):
        obs_dict, actions, rewards, terminated = demo.simulate_training_step()
        
        # 显示关键信息
        if step < 2:  # 避免输出太长
            print(f"\n>>> 步骤总结:")
            print(f"    平均奖励: {rewards.mean():.3f}")
            print(f"    动作幅度: {torch.norm(actions, dim=-1).mean():.3f}")
    
    print(f"\n" + "=" * 60)
    print("关键洞察:")
    print("1. 每个步骤都是 多环境并行 处理")
    print("2. TensorDict 让不同观测类型分离处理")  
    print("3. 奖励权重直接影响学习优先级")
    print("4. 域随机化在重置时自动生效")
    print("5. 整个流程高度向量化，GPU加速")

if __name__ == "__main__":
    demonstrate_training_flow()
