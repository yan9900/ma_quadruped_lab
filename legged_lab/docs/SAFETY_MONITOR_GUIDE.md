# Go2 Safety Monitor 集成指南

## 📖 概述

本指南说明如何在 LeggedLab 中集成世界模型作为安全过滤器，在检测到危险状态时自动过滤不安全的动作。

## 🎯 工作原理

```
┌─────────────────────────────────────────────────────┐
│  IsaacSim 环境 (LeggedLab)                           │
│    ↓                                                │
│  [观测] depth_image (64x64) + obs_state (45D)       │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  原始策略 (PPO)                                      │
│  输入: obs → 输出: action_raw (12D)                 │
└─────────────────────────────────────────────────────┘
         ↓
    action_raw (候选动作)
         ↓
┌─────────────────────────────────────────────────────┐
│  Safety Monitor (世界模型)                           │
│  1. Encoder: image, obs_state → embed               │
│  2. RSSM: embed → z (latent state)                 │
│  3. Margin Head: z → g(z) (安全边界)                │
└─────────────────────────────────────────────────────┘
         ↓
    if g(z) < threshold:
        action_final = STOP_ACTION  # ❌ 危险！停止
    else:
        action_final = action_raw   # ✅ 安全，放行
         ↓
┌─────────────────────────────────────────────────────┐
│  执行到 IsaacSim                                     │
└─────────────────────────────────────────────────────┘
```

## 📁 文件结构

```
LeggedLab/legged_lab/scripts/
├── safety_monitor.py       # Safety Monitor 核心模块
├── play_with_safety.py     # 带安全过滤的评估脚本
└── play.py                 # 原始评估脚本 (无安全过滤)

latent-safety/
├── logs/go2_world_model_v2/
│   └── rssm_ckpt.pt        # 训练好的世界模型
├── configs.yaml            # 世界模型配置
└── dreamerv3-torch/        # 世界模型实现
```

## 🚀 快速开始

### 1. 运行带安全过滤的评估

```bash
cd /home/lcy/LeggedLab

# 基本用法
python legged_lab/scripts/play_with_safety.py --task go2_flat --enable_cameras

# 调整安全阈值 (更保守)
python legged_lab/scripts/play_with_safety.py --task go2_flat --enable_cameras \
    --safety_threshold 0.1

# 仅监控模式 (不过滤，只观察)
python legged_lab/scripts/play_with_safety.py --task go2_flat --enable_cameras \
    --monitor_only

# 保存统计结果
python legged_lab/scripts/play_with_safety.py --task go2_flat --enable_cameras \
    --save_stats ./results/safety_eval.json
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wm_path` | `logs/go2_world_model_v2/rssm_ckpt.pt` | 世界模型checkpoint路径 |
| `--wm_config` | `configs.yaml` | 世界模型配置文件 |
| `--safety_threshold` | `0.0` | 安全阈值，g(x) < threshold 触发过滤 |
| `--fallback_strategy` | `stop` | 后备策略: stop/previous/zero_velocity |
| `--monitor_only` | `False` | 仅监控，不过滤 |
| `--save_stats` | `None` | 统计结果保存路径 |
| `--print_interval` | `100` | 打印间隔 (步数) |

### 3. 后备策略说明

- **stop**: 零动作，机器人停止（最安全，但可能卡住）
- **previous**: 使用上一个安全动作（较平滑，但可能延迟响应）
- **zero_velocity**: 小幅制动（尝试减速）

## 💻 在代码中集成

### 方式1: 直接在 play.py 中添加

```python
# 在 play.py 中添加以下代码

from safety_monitor import Go2SafetyMonitor

# 创建 Safety Monitor
safety_monitor = Go2SafetyMonitor(
    world_model_path="/home/lcy/latent-safety/logs/go2_world_model_v2/rssm_ckpt.pt",
    config_path="/home/lcy/latent-safety/configs.yaml",
    threshold=0.0,
    fallback_strategy="stop",
    device=str(env.device),
    env=env  # 传入环境对象以获取相机数据
)

# 在主循环中使用
while simulation_app.is_running():
    with torch.inference_mode():
        # 原始策略生成动作
        actions_raw = policy(obs_dict)
        
        # 安全过滤
        actions, was_filtered, g_values = safety_monitor.filter_action(
            obs_dict, actions_raw
        )
        
        # 执行过滤后的动作
        obs_dict, _, _, _ = env.step(actions)
        
        # (可选) 检查是否有环境触发过滤
        if was_filtered.any():
            print(f"Safety filter triggered! g(x)={g_values[was_filtered]}")
```

### 方式2: 仅获取安全值 (不过滤)

```python
# 仅监控安全状态，不改变动作
g_values = safety_monitor.get_safety_values(obs_dict)

# 可以用于:
# 1. 记录统计
# 2. 可视化
# 3. 触发报警
if (g_values < -0.5).any():
    print(f"WARNING: Very dangerous state detected!")
```

### 方式3: 自定义过滤逻辑

```python
# 获取安全值
g_values = safety_monitor.get_safety_values(obs_dict)

# 自定义过滤逻辑
dangerous_mask = g_values < args.threshold
if dangerous_mask.any():
    # 对危险环境使用自定义策略
    safe_action = custom_recovery_policy(obs_dict)
    actions_raw[dangerous_mask] = safe_action[dangerous_mask]

# 执行
obs_dict, _, _, _ = env.step(actions_raw)
```

## 📊 评估指标

运行结束后，Safety Monitor 会输出以下统计信息：

```
============================================================
Safety Monitor Statistics
============================================================
Total steps:           10000
Total episodes:        50
Filter trigger rate:   8.50%      # 过滤触发比例
Fall rate (with filter): 2.00%   # 使用过滤后的摔倒率
g(x) mean:             0.2345    # 平均安全边界值
g(x) std:              0.1234    # 安全边界标准差
g(x) min:              -0.5678   # 最小安全边界值
g(x) max:              0.8901    # 最大安全边界值
g(x) < 0 rate:         15.20%    # 负值比例（危险状态）
============================================================
```

## 🔧 调参指南

### 安全阈值 (threshold)

| 阈值 | 特点 | 适用场景 |
|------|------|----------|
| `0.3` | 非常保守，频繁过滤 | 测试/验证 |
| `0.0` | 平衡，g(x)<0时过滤 | **推荐** |
| `-0.3` | 宽松，只过滤高危 | 任务优先 |

### 如何选择阈值?

1. 先用 `--monitor_only` 运行，观察 g(x) 分布
2. 查看 `g(x) < 0 rate`，如果 > 30%，阈值可能太高
3. 查看 `g(x) min`，如果经常低于 -0.5，说明有危险状态
4. 根据任务需求调整：安全优先选高阈值，任务优先选低阈值

## ❓ 常见问题

### Q1: 找不到相机数据?

确保：
1. 任务配置启用了相机：`env_cfg.scene.camera.enable_camera = True`
2. 使用 `--enable_cameras` 参数运行
3. 相机数据类型正确：`data_types = ["distance_to_image_plane"]`

### Q2: 世界模型加载失败?

检查：
1. 路径是否正确
2. PyTorch 版本兼容性
3. 配置文件与checkpoint匹配

### Q3: 过滤太频繁导致卡住?

尝试：
1. 降低阈值：`--safety_threshold -0.2`
2. 使用 `previous` 策略：`--fallback_strategy previous`
3. 检查世界模型是否训练充分

### Q4: 机器人仍然摔倒?

可能原因：
1. 阈值太低，未能及时检测
2. 世界模型预测不准确
3. 后备策略不合适

解决：
1. 提高阈值
2. 使用更好的世界模型
3. 尝试预测式安全检查（未来版本）

## 📈 性能参考

基于 Dubins Car 实验的预期：

| 指标 | Baseline | 带过滤 | 改进 |
|------|----------|--------|------|
| 摔倒率 | ~15% | ~3% | ↓80% |
| 任务成功率 | ~85% | ~75% | ↓10% |
| 平均步数 | ~850 | ~820 | 略降 |

**注意**: 方案1（过滤模式）会牺牲一些任务完成率换取安全性。

## 🔗 相关文档

- [部署方案对比](../../../latent-safety/docs/deployment_comparison_guide.md)
- [世界模型训练](../../../latent-safety/docs/dreamer_offline_training_pipeline.md)
- [Margin Head 实现](../../../latent-safety/docs/margin_head_implementation.md)

---

**如有问题，请查阅相关文档或联系开发者。** 🚀
