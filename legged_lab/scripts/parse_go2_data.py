#!/usr/bin/env python3
"""
解析 collect_go2_data_v2.py 收集的 .pkl 数据文件
可视化图像和轨迹数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(pkl_path: str):
    """加载 pkl 数据"""
    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        demos = pickle.load(f)
    print(f"Loaded {len(demos)} trajectories")
    return demos


def print_data_structure(demos):
    """打印数据结构"""
    print("\n" + "="*60)
    print("数据结构分析")
    print("="*60)
    
    print(f"\n总轨迹数: {len(demos)}")
    
    if not demos:
        print("数据为空!")
        return
    
    sample = demos[0]
    print(f"\n单条轨迹的 keys: {list(sample.keys())}")
    
    # obs 结构
    print(f"\nobs 结构:")
    for key, value in sample['obs'].items():
        if isinstance(value, list) and len(value) > 0:
            first_item = value[0]
            if isinstance(first_item, np.ndarray):
                print(f"  - {key}: List[{len(value)}] of ndarray, shape={first_item.shape}, dtype={first_item.dtype}")
            else:
                print(f"  - {key}: List[{len(value)}] of {type(first_item)}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # actions
    actions = sample['actions']
    if isinstance(actions, list) and len(actions) > 0:
        print(f"\nactions: List[{len(actions)}] of ndarray, shape={actions[0].shape}")
    
    # dones
    dones = sample['dones']
    print(f"\ndones: List[{len(dones)}], 最后几个值: {dones[-5:]}")
    
    # env_id (如果有)
    if 'env_id' in sample:
        print(f"\nenv_id: {sample['env_id']}")
    
    # 统计轨迹长度
    lengths = [len(demo['actions']) for demo in demos]
    print(f"\n轨迹长度统计:")
    print(f"  - 最短: {min(lengths)}")
    print(f"  - 最长: {max(lengths)}")
    print(f"  - 平均: {np.mean(lengths):.1f}")
    print(f"  - 总步数: {sum(lengths)}")


def visualize_images(demos, traj_idx=0, num_frames=10, save_path=None):
    """可视化指定轨迹的图像"""
    if traj_idx >= len(demos):
        print(f"轨迹索引 {traj_idx} 超出范围 (共 {len(demos)} 条)")
        return
    
    traj = demos[traj_idx]
    images = traj['obs']['image']
    
    if not images:
        print("该轨迹没有图像数据!")
        return
    
    print(f"\n轨迹 {traj_idx} 的图像信息:")
    print(f"  - 帧数: {len(images)}")
    
    first_img = images[0]
    print(f"  - 图像类型: {type(first_img)}")
    
    if isinstance(first_img, np.ndarray):
        print(f"  - 图像 shape: {first_img.shape}")
        print(f"  - 图像 dtype: {first_img.dtype}")
        print(f"  - 数值范围: [{first_img.min()}, {first_img.max()}]")
    elif isinstance(first_img, dict):
        print(f"  - 图像是字典，keys: {list(first_img.keys())}")
        for key, val in first_img.items():
            if isinstance(val, np.ndarray):
                print(f"    - {key}: shape={val.shape}, dtype={val.dtype}, range=[{val.min():.2f}, {val.max():.2f}]")
    
    # 选择要显示的帧
    total_frames = len(images)
    step = max(1, total_frames // num_frames)
    frame_indices = list(range(0, total_frames, step))[:num_frames]
    
    # 创建图像网格
    n_cols = min(5, len(frame_indices))
    n_rows = (len(frame_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes = np.array(axes).flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        img = images[frame_idx]
        
        if isinstance(img, dict):
            # 如果是字典，优先显示 RGB，否则显示 depth
            if 'rgb' in img:
                img_to_show = img['rgb']
                title = f"Frame {frame_idx} (RGB)"
            elif 'rgba' in img:
                img_to_show = img['rgba'][..., :3]  # 只取 RGB
                title = f"Frame {frame_idx} (RGBA→RGB)"
            elif 'depth' in img:
                img_to_show = img['depth']
                title = f"Frame {frame_idx} (Depth)"
            else:
                # 取第一个 key
                key = list(img.keys())[0]
                img_to_show = img[key]
                title = f"Frame {frame_idx} ({key})"
        else:
            img_to_show = img
            title = f"Frame {frame_idx}"
        
        # 显示图像
        if len(img_to_show.shape) == 2:
            # 灰度图/深度图
            im = ax.imshow(img_to_show, cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif len(img_to_show.shape) == 3:
            if img_to_show.shape[-1] in [3, 4]:
                # RGB(A) 图像
                if img_to_show.dtype == np.float32 or img_to_show.dtype == np.float64:
                    # 归一化到 0-1
                    img_to_show = np.clip(img_to_show, 0, 1)
                ax.imshow(img_to_show)
            else:
                # 其他格式，尝试显示第一个通道
                ax.imshow(img_to_show[..., 0], cmap='viridis')
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(len(frame_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Trajectory {traj_idx} - {len(images)} frames total", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def visualize_states(demos, traj_idx=0, save_path=None):
    """可视化状态数据"""
    if traj_idx >= len(demos):
        print(f"轨迹索引 {traj_idx} 超出范围")
        return
    
    traj = demos[traj_idx]
    states = traj['obs']['state']
    priv_states = traj['obs']['priv_state']
    actions = traj['actions']
    
    if not states:
        print("没有状态数据!")
        return
    
    # 转换为 numpy array
    states_arr = np.array(states)
    priv_arr = np.array(priv_states)
    actions_arr = np.array(actions)
    
    print(f"\n轨迹 {traj_idx} 状态信息:")
    print(f"  - state shape: {states_arr.shape}")
    print(f"  - priv_state shape: {priv_arr.shape}")
    print(f"  - actions shape: {actions_arr.shape}")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 绘制部分 state
    ax = axes[0]
    num_dims = min(10, states_arr.shape[1])
    for i in range(num_dims):
        ax.plot(states_arr[:, i], label=f'dim_{i}', alpha=0.7)
    ax.set_title(f'State (前{num_dims}维)')
    ax.set_xlabel('Step')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 绘制 priv_state
    ax = axes[1]
    num_dims = min(10, priv_arr.shape[1])
    for i in range(num_dims):
        ax.plot(priv_arr[:, i], label=f'dim_{i}', alpha=0.7)
    ax.set_title(f'Privileged State (前{num_dims}维)')
    ax.set_xlabel('Step')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 绘制 actions
    ax = axes[2]
    for i in range(actions_arr.shape[1]):
        ax.plot(actions_arr[:, i], label=f'joint_{i}', alpha=0.7)
    ax.set_title('Actions (关节命令)')
    ax.set_xlabel('Step')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Trajectory {traj_idx} - State & Action Visualization', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def create_video(demos, traj_idx=0, output_path="trajectory.mp4", fps=30):
    """从轨迹图像创建视频"""
    try:
        import cv2
    except ImportError:
        print("需要安装 opencv: pip install opencv-python")
        return
    
    if traj_idx >= len(demos):
        print(f"轨迹索引 {traj_idx} 超出范围")
        return
    
    traj = demos[traj_idx]
    images = traj['obs']['image']
    
    if not images:
        print("没有图像数据!")
        return
    
    # 获取第一帧的尺寸
    first_img = images[0]
    if isinstance(first_img, dict):
        if 'rgb' in first_img:
            first_img = first_img['rgb']
        elif 'rgba' in first_img:
            first_img = first_img['rgba'][..., :3]
        else:
            first_img = list(first_img.values())[0]
    
    h, w = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in images:
        if isinstance(img, dict):
            if 'rgb' in img:
                img = img['rgb']
            elif 'rgba' in img:
                img = img['rgba'][..., :3]
            else:
                img = list(img.values())[0]
        
        # 确保是 uint8 格式
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        out.write(img)
    
    out.release()
    print(f"视频已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 Go2 数据")
    parser.add_argument("--pkl", type=str, default="./data/go2_demo/go2_demos_2000ep.pkl",
                       help="pkl 文件路径")
    parser.add_argument("--traj", type=int, default=0, help="要可视化的轨迹索引")
    parser.add_argument("--frames", type=int, default=10, help="显示的帧数")
    parser.add_argument("--save_dir", type=str, default="./data/go2_demo/vis",
                       help="保存可视化结果的目录")
    parser.add_argument("--video", action="store_true", help="生成视频")
    
    args = parser.parse_args()
    
    # 加载数据
    demos = load_data(args.pkl)
    
    # 打印数据结构
    print_data_structure(demos)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化图像
    print("\n" + "="*60)
    print("可视化图像")
    print("="*60)
    visualize_images(
        demos, 
        traj_idx=args.traj, 
        num_frames=args.frames,
        save_path=save_dir / f"traj_{args.traj}_images.png"
    )
    
    # 可视化状态
    print("\n" + "="*60)
    print("可视化状态")
    print("="*60)
    visualize_states(
        demos,
        traj_idx=args.traj,
        save_path=save_dir / f"traj_{args.traj}_states.png"
    )
    
    # 生成视频（可选）
    if args.video:
        print("\n" + "="*60)
        print("生成视频")
        print("="*60)
        create_video(
            demos,
            traj_idx=args.traj,
            output_path=str(save_dir / f"traj_{args.traj}.mp4")
        )
