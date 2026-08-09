#!/usr/bin/env python3
"""
分片合并脚本

用途：将 collect_go2_data_v2.py 崩溃后留下的 shards/ 目录合并为完整 .pkl 文件

用法：
    # 单个目录合并
    python scripts/merge_shards.py --shard_dir ./data/go2_demo/shards --output ./data/go2_demo/go2_recovered.pkl

    # 也可以只指定包含 shards/ 的父目录（自动查找）
    python scripts/merge_shards.py --shard_dir ./data/go2_demo --output ./data/go2_demo/go2_recovered.pkl

    # 合并多个目录并 shuffle
    python scripts/merge_shards.py \
        --shard_dir ./data/go2_0310hspeed_nopush_bigPlatform \
        --shard_dir ./data/go2_0310lspeed_nopush \
        --shard_dir ./data/go2_1028hspeed_nopush \
        --output ./data/go2_merged.pkl --shuffle

    # 合并后删除分片
    python scripts/merge_shards.py --shard_dir ./data/go2_demo --output ./data/go2_demo/go2_recovered.pkl --delete_shards
"""

import argparse
import glob
import pickle
import os
import random
from pathlib import Path


def find_shards(shard_dir: Path):
    """查找并排序所有分片文件"""
    # 如果传入的是父目录，自动定位到 shards/ 子目录
    if (shard_dir / "shards").exists():
        shard_dir = shard_dir / "shards"
    
    shards = sorted(shard_dir.glob("shard_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"在 {shard_dir} 中未找到 shard_*.pkl 文件")
    return shards, shard_dir


def merge_shards(shard_dirs: list, output_path: Path, shuffle: bool = False,
                 delete_shards: bool = False, num_shards_list: list = None):
    """
    合并多个分片目录
    
    Args:
        shard_dirs: 分片目录路径列表
        output_path: 输出文件路径
        shuffle: 是否打乱合并后的数据
        delete_shards: 合并后是否删除分片目录
        num_shards_list: 每个目录选取的分片数量（与 shard_dirs 一一对应），None 表示全取
    """
    all_demos = []
    all_shard_dirs = []
    
    if num_shards_list is None:
        num_shards_list = [None] * len(shard_dirs)
    
    # 处理多个目录
    for idx, shard_dir in enumerate(shard_dirs):
        shards, resolved_shard_dir = find_shards(Path(shard_dir))
        all_shard_dirs.append(resolved_shard_dir)
        max_n = num_shards_list[idx]
        total_shards = len(shards)
        
        if max_n is not None and max_n < total_shards:
            shards = shards[:max_n]
            print(f"\n[INFO] 处理目录: {shard_dir}  (选取前 {max_n}/{total_shards} 个分片)")
        else:
            print(f"\n[INFO] 处理目录: {shard_dir}")
        
        print(f"       找到 {len(shards)} 个分片:")
        total_size = 0
        for s in shards:
            size_mb = s.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"       {s.name}  ({size_mb:.1f} MB)")
        print(f"       总大小: {total_size:.1f} MB")
        
        # 读取分片
        for i, shard_file in enumerate(shards):
            with open(shard_file, 'rb') as f:
                chunk = pickle.load(f)
            all_demos.extend(chunk)
            print(f"       [{i+1}/{len(shards)}] {shard_file.name}: {len(chunk)} eps  (累计 {len(all_demos)} eps)")
    
    print(f"\n[INFO] 合并完成: {len(all_demos)} 个 episodes")
    
    # 打乱顺序
    if shuffle:
        print(f"[INFO] 正在打乱数据顺序...")
        random.shuffle(all_demos)
        print(f"[INFO] 打乱完成")
    
    # 原子写入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(all_demos, f, protocol=4)
    tmp_path.rename(output_path)
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[INFO] 已写入: {output_path}  ({size_mb:.1f} MB)")
    
    # 打印数据结构
    if all_demos:
        s = all_demos[0]
        print(f"\n[INFO] 数据结构验证:")
        print(f"   - episodes: {len(all_demos)}")
        print(f"   - 第1条轨迹长度: {len(s['actions'])} steps")
        print(f"   - state shape: {s['obs']['state'][0].shape}")
        print(f"   - priv_state shape: {s['obs']['priv_state'][0].shape}")
        print(f"   - action shape: {s['actions'][0].shape}")
        print(f"   - image shape: {s['obs']['image'][0].shape}")
        ok_count = sum(1 for d in all_demos if d['failure'][-1] == 0)
        fail_count = len(all_demos) - ok_count
        print(f"   - OK episodes: {ok_count} ({100*ok_count/len(all_demos):.1f}%)")
        print(f"   - FAIL episodes: {fail_count} ({100*fail_count/len(all_demos):.1f}%)")
    
    if delete_shards:
        import shutil
        for shard_dir in all_shard_dirs:
            shutil.rmtree(shard_dir)
            print(f"\n[INFO] 已删除分片目录: {shard_dir}")
    else:
        print(f"\n[HINT] 确认数据无误后可手动删除分片:")
        for shard_dir in all_shard_dirs:
            print(f"       rm -rf {shard_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 collect_go2_data_v2.py 生成的分片文件")
    parser.add_argument("--shard_dir", type=str, action="append", required=True,
                        help="分片目录路径（shards/ 目录或其父目录），可指定多个")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 .pkl 文件路径")
    parser.add_argument("--shuffle", action="store_true",
                        help="合并后是否打乱数据顺序")
    parser.add_argument("--delete_shards", action="store_true",
                        help="合并成功后删除分片目录")
    parser.add_argument("--num_shards", type=int, action="append", default=None,
                        help="每个 --shard_dir 选取的分片数量（与 --shard_dir 一一对应），不指定则全取")
    args = parser.parse_args()
    
    merge_shards(
        shard_dirs=args.shard_dir,
        output_path=Path(args.output),
        shuffle=args.shuffle,
        delete_shards=args.delete_shards,
        num_shards_list=args.num_shards,
    )
