#!/usr/bin/env python3
"""
提取 collect_go2_data_v2.py 收集的所有图像
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2


def load_data(pkl_path: str):
    """加载 pkl 数据"""
    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        demos = pickle.load(f)
    print(f"Loaded {len(demos)} trajectories")
    return demos


def normalize_depth_image(depth_img, min_val=0.5, max_val=5.0):
    """
    将深度图归一化到 0-255 用于可视化
    depth_img: 深度值，单位米
    min_val: 最小深度（米）
    max_val: 最大深度（米）
    """
    # 处理 inf 值
    depth_img = np.clip(depth_img, min_val, max_val)
    # 归一化到 0-1
    normalized = (depth_img - min_val) / (max_val - min_val)
    # 转换到 0-255
    img_uint8 = (normalized * 255).astype(np.uint8)
    return img_uint8


def extract_all_images(demos, output_dir: str, format='png', max_trajs=None):
    """
    提取所有轨迹的所有图像
    
    目录结构:
    output_dir/
        traj_0000/
            frame_0000.png
            frame_0001.png
            ...
        traj_0001/
            ...
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(demos) if max_trajs is None else min(max_trajs, len(demos))
    
    total_frames = 0
    
    print(f"\n提取 {num_trajs} 条轨迹的图像...")
    
    for traj_idx in tqdm(range(num_trajs), desc="轨迹"):
        traj = demos[traj_idx]
        images = traj['obs']['image']
        
        if not images:
            continue
        
        # 创建轨迹目录
        traj_dir = output_path / f"traj_{traj_idx:04d}"
        traj_dir.mkdir(exist_ok=True)
        
        for frame_idx, img in enumerate(images):
            # 处理图像格式
            if isinstance(img, dict):
                if 'rgb' in img:
                    img_to_save = img['rgb']
                elif 'rgba' in img:
                    img_to_save = img['rgba'][..., :3]
                elif 'depth' in img:
                    img_to_save = img['depth']
                else:
                    img_to_save = list(img.values())[0]
            else:
                img_to_save = img
            
            # 处理深度图 (shape: H, W, 1 或 H, W)
            if len(img_to_save.shape) == 3 and img_to_save.shape[-1] == 1:
                img_to_save = img_to_save.squeeze(-1)  # 去掉最后一维
            
            # 如果是深度图（float32 且值域较大），进行归一化
            if img_to_save.dtype == np.float32:
                if len(img_to_save.shape) == 2:  # 深度图
                    img_to_save = normalize_depth_image(img_to_save)
                    # 转为伪彩色便于可视化
                    img_to_save = cv2.applyColorMap(img_to_save, cv2.COLORMAP_VIRIDIS)
                else:
                    # RGB float 图像
                    img_to_save = np.clip(img_to_save * 255, 0, 255).astype(np.uint8)
            
            # 保存图像
            frame_path = traj_dir / f"frame_{frame_idx:04d}.{format}"
            
            if len(img_to_save.shape) == 3:
                # BGR for OpenCV
                cv2.imwrite(str(frame_path), img_to_save)
            else:
                cv2.imwrite(str(frame_path), img_to_save)
            
            total_frames += 1
    
    print(f"\n✓ 提取完成!")
    print(f"  - 轨迹数: {num_trajs}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 保存目录: {output_path}")


def extract_all_images_flat(demos, output_dir: str, format='png', max_trajs=None):
    """
    将所有图像提取到一个平面目录（用于训练）
    
    文件名格式: traj{轨迹号}_frame{帧号}.png
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(demos) if max_trajs is None else min(max_trajs, len(demos))
    
    total_frames = 0
    
    print(f"\n提取 {num_trajs} 条轨迹的图像 (平面模式)...")
    
    for traj_idx in tqdm(range(num_trajs), desc="轨迹"):
        traj = demos[traj_idx]
        images = traj['obs']['image']
        
        if not images:
            continue
        
        for frame_idx, img in enumerate(images):
            # 处理图像格式
            if isinstance(img, dict):
                if 'rgb' in img:
                    img_to_save = img['rgb']
                elif 'rgba' in img:
                    img_to_save = img['rgba'][..., :3]
                elif 'depth' in img:
                    img_to_save = img['depth']
                else:
                    img_to_save = list(img.values())[0]
            else:
                img_to_save = img
            
            # 处理深度图
            if len(img_to_save.shape) == 3 and img_to_save.shape[-1] == 1:
                img_to_save = img_to_save.squeeze(-1)
            
            if img_to_save.dtype == np.float32:
                if len(img_to_save.shape) == 2:
                    img_to_save = normalize_depth_image(img_to_save)
                    img_to_save = cv2.applyColorMap(img_to_save, cv2.COLORMAP_VIRIDIS)
                else:
                    img_to_save = np.clip(img_to_save * 255, 0, 255).astype(np.uint8)
            
            # 保存图像
            frame_path = output_path / f"traj{traj_idx:04d}_frame{frame_idx:04d}.{format}"
            cv2.imwrite(str(frame_path), img_to_save)
            
            total_frames += 1
    
    print(f"\n✓ 提取完成!")
    print(f"  - 轨迹数: {num_trajs}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 保存目录: {output_path}")


def extract_raw_depth(demos, output_dir: str, max_trajs=None):
    """
    提取原始深度数据（保存为 .npy 格式，保留原始数值）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(demos) if max_trajs is None else min(max_trajs, len(demos))
    
    total_frames = 0
    
    print(f"\n提取 {num_trajs} 条轨迹的原始深度数据...")
    
    for traj_idx in tqdm(range(num_trajs), desc="轨迹"):
        traj = demos[traj_idx]
        images = traj['obs']['image']
        
        if not images:
            continue
        
        traj_dir = output_path / f"traj_{traj_idx:04d}"
        traj_dir.mkdir(exist_ok=True)
        
        for frame_idx, img in enumerate(images):
            if isinstance(img, dict):
                img = list(img.values())[0]
            
            # 保存为 npy
            frame_path = traj_dir / f"depth_{frame_idx:04d}.npy"
            np.save(frame_path, img)
            
            total_frames += 1
    
    print(f"\n✓ 提取完成!")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 保存目录: {output_path}")


def create_video_from_traj(demos, traj_idx: int, output_path: str, fps=30):
    """为单条轨迹创建视频"""
    if traj_idx >= len(demos):
        print(f"轨迹索引 {traj_idx} 超出范围")
        return
    
    traj = demos[traj_idx]
    images = traj['obs']['image']
    
    if not images:
        print("没有图像数据!")
        return
    
    # 处理第一帧获取尺寸
    first_img = images[0]
    if isinstance(first_img, dict):
        first_img = list(first_img.values())[0]
    if len(first_img.shape) == 3 and first_img.shape[-1] == 1:
        first_img = first_img.squeeze(-1)
    
    h, w = first_img.shape[:2]
    
    # 应用 colormap 后是 3 通道
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in images:
        if isinstance(img, dict):
            img = list(img.values())[0]
        
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        if img.dtype == np.float32:
            img = normalize_depth_image(img)
            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        
        out.write(img)
    
    out.release()
    print(f"视频已保存: {output_path}")


def create_all_videos(demos, output_dir: str, fps=30, max_trajs=None):
    """为所有轨迹创建视频"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(demos) if max_trajs is None else min(max_trajs, len(demos))
    
    print(f"\n为 {num_trajs} 条轨迹创建视频...")
    
    for traj_idx in tqdm(range(num_trajs), desc="视频"):
        video_path = output_path / f"traj_{traj_idx:04d}.mp4"
        create_video_from_traj(demos, traj_idx, str(video_path), fps)
    
    print(f"\n✓ 视频创建完成!")
    print(f"  - 保存目录: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 Go2 数据中的所有图像")
    parser.add_argument("--pkl", type=str, default="./data/go2_demo/go2_demos_2000ep.pkl",
                       help="pkl 文件路径")
    parser.add_argument("--output_dir", type=str, default="./data/go2_demo/images",
                       help="输出目录")
    parser.add_argument("--format", type=str, default="png", choices=['png', 'jpg'],
                       help="图像格式")
    parser.add_argument("--flat", action="store_true",
                       help="平面模式（所有图像放一个目录）")
    parser.add_argument("--raw", action="store_true",
                       help="保存原始深度数据 (.npy)")
    parser.add_argument("--video", action="store_true",
                       help="为每条轨迹创建视频")
    parser.add_argument("--max_trajs", type=int, default=None,
                       help="最多处理多少条轨迹")
    parser.add_argument("--fps", type=int, default=30,
                       help="视频帧率")
    
    args = parser.parse_args()
    
    # 加载数据
    demos = load_data(args.pkl)
    
    # 打印基本信息
    total_frames = sum(len(demo['obs']['image']) for demo in demos)
    print(f"总帧数: {total_frames}")
    
    if args.raw:
        # 保存原始深度数据
        extract_raw_depth(demos, args.output_dir + "_raw", args.max_trajs)
    elif args.video:
        # 创建视频
        create_all_videos(demos, args.output_dir + "_videos", args.fps, args.max_trajs)
    elif args.flat:
        # 平面模式
        extract_all_images_flat(demos, args.output_dir, args.format, args.max_trajs)
    else:
        # 分层目录模式
        extract_all_images(demos, args.output_dir, args.format, args.max_trajs)
