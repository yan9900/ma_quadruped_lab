#!/usr/bin/env python3

"""
验证收集的Go2数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_verify_data(data_file: str):
    """加载并验证数据"""
    print(f"📁 加载数据文件: {data_file}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print("\n📊 数据结构验证:")
    print(f"  数据类型: {type(data)}")
    print(f"  主要键: {list(data.keys())}")
    print(f"  观测键: {list(data['obs'].keys())}")
    
    # 验证数据形状
    print(f"\n📏 数据形状:")
    for key, value in data.items():
        if key == 'obs':
            for obs_key, obs_value in value.items():
                print(f"  {key}.{obs_key}: {obs_value.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # 数据统计
    print(f"\n📈 数据统计:")
    print(f"  总时间步: {len(data['actions'])}")
    print(f"  环境数量: {data['actions'].shape[1] if len(data['actions'].shape) > 1 else 1}")
    print(f"  图像分辨率: {data['obs']['image'].shape[-3:-1]}")
    print(f"  图像通道: {data['obs']['image'].shape[-1]}")
    
    # 数据范围
    print(f"\n📊 数据范围:")
    print(f"  动作范围: [{data['actions'].min():.3f}, {data['actions'].max():.3f}]")
    print(f"  图像范围: [{data['obs']['image'].min():.3f}, {data['obs']['image'].max():.3f}]")
    print(f"  状态范围: [{data['obs']['state'].min():.3f}, {data['obs']['state'].max():.3f}]")
    
    return data

def visualize_sample_data(data, sample_idx=0, env_idx=0):
    """可视化样本数据"""
    print(f"\n🖼️  可视化样本数据 (时间步={sample_idx}, 环境={env_idx})")
    
    # 创建图像显示
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. RGB图像
    if data['obs']['image'].shape[-1] >= 3:
        rgb_img = data['obs']['image'][sample_idx, env_idx, :, :, :3]
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis('off')
    
    # 2. 深度图像 (如果有)
    if data['obs']['image'].shape[-1] > 3:
        depth_img = data['obs']['image'][sample_idx, env_idx, :, :, 3]
        im = axes[0, 1].imshow(depth_img, cmap='plasma')
        axes[0, 1].set_title("Depth Image")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
    
    # 3. 关节位置
    joint_pos = data['obs']['state'][sample_idx, env_idx, :12]  # 前12维是关节位置
    axes[1, 0].bar(range(len(joint_pos)), joint_pos)
    axes[1, 0].set_title("Joint Positions")
    axes[1, 0].set_xlabel("Joint Index")
    axes[1, 0].set_ylabel("Position (rad)")
    
    # 4. 动作
    actions = data['actions'][sample_idx, env_idx]
    axes[1, 1].bar(range(len(actions)), actions)
    axes[1, 1].set_title("Actions")
    axes[1, 1].set_xlabel("Action Index")
    axes[1, 1].set_ylabel("Action Value")
    
    plt.tight_layout()
    plt.savefig(f"sample_visualization_{sample_idx}_{env_idx}.png", dpi=150)
    print(f"📊 可视化图像已保存: sample_visualization_{sample_idx}_{env_idx}.png")
    plt.show()

def analyze_trajectories(data, env_idx=0):
    """分析轨迹数据"""
    print(f"\n📈 分析环境 {env_idx} 的轨迹")
    
    # 提取位置轨迹 (私有状态的前3维是位置)
    positions = data['obs']['priv_state'][:, env_idx, :3]  # [timesteps, 3]
    
    # 绘制轨迹
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY轨迹
    axes[0, 0].plot(positions[:, 0], positions[:, 1])
    axes[0, 0].set_title("XY Trajectory")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].grid(True)
    
    # Z高度变化
    axes[0, 1].plot(positions[:, 2])
    axes[0, 1].set_title("Height over Time")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Z (m)")
    axes[0, 1].grid(True)
    
    # 速度分析 (私有状态的第7-9维是线速度)
    velocities = data['obs']['priv_state'][:, env_idx, 7:10]  # [timesteps, 3]
    speed = np.linalg.norm(velocities, axis=1)
    axes[1, 0].plot(speed)
    axes[1, 0].set_title("Speed over Time")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].grid(True)
    
    # 奖励变化
    rewards = data['rewards'][:, env_idx]
    axes[1, 1].plot(rewards)
    axes[1, 1].set_title("Rewards over Time")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"trajectory_analysis_env_{env_idx}.png", dpi=150)
    print(f"📊 轨迹分析图像已保存: trajectory_analysis_env_{env_idx}.png")
    plt.show()

def main():
    # 数据文件路径 (请根据实际路径修改)
    data_file = "./data/go2_demo/go2_demo_flat_5ep.pkl"
    
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行数据收集脚本:")
        print("python scripts/collect_go2_data.py --num_episodes 5 --max_steps 200")
        return
    
    # 加载和验证数据
    data = load_and_verify_data(data_file)
    
    # 可视化样本数据
    visualize_sample_data(data, sample_idx=10, env_idx=0)
    
    # 分析轨迹
    analyze_trajectories(data, env_idx=0)
    
    print("\n✅ 数据验证完成!")

if __name__ == "__main__":
    main()