#!/usr/bin/env python3
"""
测试 Safety Monitor 能否正确加载

运行方式:
    python scripts/test_safety_monitor.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_safety_monitor():
    print("=" * 60)
    print("Testing Safety Monitor")
    print("=" * 60)
    
    # 测试配置
    wm_path = "/home/lcy/latent-safety/logs/go2_world_model_v2/rssm_ckpt.pt"
    config_path = "/home/lcy/latent-safety/configs.yaml"
    
    print(f"\n1. Loading Safety Monitor...")
    print(f"   World model: {wm_path}")
    print(f"   Config: {config_path}")
    
    try:
        from safety_monitor import Go2SafetyMonitor
        
        monitor = Go2SafetyMonitor(
            world_model_path=wm_path,
            config_path=config_path,
            threshold=0.0,
            fallback_strategy="stop",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("   ✓ Safety Monitor loaded successfully!")
        print(f"   - Feature size: {monitor.feat_size}")
        print(f"   - Threshold: {monitor.threshold}")
        print(f"   - Device: {monitor.device}")
        
    except Exception as e:
        print(f"   ✗ Failed to load Safety Monitor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n2. Testing with fake observations...")
    
    try:
        # 创建假数据
        batch_size = 4
        fake_obs = {
            'policy': torch.randn(batch_size, 45),  # 状态向量
            'depth': torch.rand(batch_size, 64, 64) * 255,  # 深度图 (0-255)
        }
        
        # 测试安全值计算
        g_values, latent = monitor.compute_safety(fake_obs)
        print(f"   ✓ Safety values computed: {g_values}")
        print(f"   - Shape: {g_values.shape}")
        print(f"   - Range: [{g_values.min():.3f}, {g_values.max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ Failed to compute safety: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n3. Testing action filtering...")
    
    try:
        # 重置状态
        monitor.reset()
        
        # 创建假动作
        fake_action = torch.randn(batch_size, 12)
        
        # 测试过滤
        filtered_action, was_filtered, g_values = monitor.filter_action(
            fake_obs, fake_action
        )
        
        print(f"   ✓ Action filtering works!")
        print(f"   - Filtered count: {was_filtered.sum().item()}/{batch_size}")
        print(f"   - g(x) values: {g_values}")
        
    except Exception as e:
        print(f"   ✗ Failed to filter action: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n4. Testing statistics...")
    
    try:
        stats = monitor.get_stats_summary()
        print(f"   ✓ Statistics retrieved!")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"   ✗ Failed to get statistics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_safety_monitor()
    sys.exit(0 if success else 1)
