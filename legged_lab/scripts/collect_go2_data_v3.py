#!/usr/bin/env python3
"""
Go2数据收集脚本 v3 - 速度跟随失效场景（V1: 摩擦力变化 / V2: 持续阻力）

基于 v2 扩展，新增：
  - V1 (--fault_type friction): T_fault 步骤后将足部摩擦系数降至 OOD 范围
  - V2 (--fault_type force_x): T_fault 步骤后对 base 施加持续 -x 方向外力
  - collection_mode=braking: 保持原数据结构，采集正常制动 command transition
  - T_fault 在 [t_fault_min, t_fault_max] 内随机化（防止 RSSM 学到时序伪相关）
  - 3D 速度跟随失效标签（vx/vy/yaw_rate 三分量分别检测）
  - t_fault / fault_type 写入 demo 元数据，供训练时分层采样使用
  - priv_state 扩展至 R^10：在原 R^7（base_lin_vel_b + feet_contact）基础上追加
      foot_static_friction (1): 当前 env 的足部静摩擦系数（正常轨迹为 DR 采样值；V1 触发后为 OOD 值）
      foot_dynamic_friction(1): 当前 env 的足部动摩擦系数
      applied_force_x      (1): 当前 env 受到的 base x 向外力（N；V2 触发后非零，安全帧为 0）

demo 的核心 key 与 v2 / generate_data_traj_cont.py 兼容，tools.py 通过
  demo.get('fault_type', 'fall') 判断是否跳过 gz/n_prefall 后处理；但 priv_state
维度从 7/9 升至 10，WM 的 priv_recon 头和旧 checkpoint 必须相应更新/重训。
"""

import argparse
import copy
import json
import math
import pickle
import random
import torch
import numpy as np
from typing import Dict, List, Any
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from isaaclab.app import AppLauncher

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description="Go2 data collection script v3 - velocity tracking failure")
parser.add_argument("--task", type=str, default="go2_data_collection", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--num_episodes", type=int, default=25, help="Episodes per environment")
parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
parser.add_argument("--output_dir", type=str, default="./data/vel_failure", help="Output directory")
parser.add_argument("--save_interval", type=int, default=50, help="每收集多少个 episode 保存一次分片")
parser.add_argument("--no_final_merge", action="store_true", help="跳过末尾合并步骤")
parser.add_argument("--output_name", type=str, default=None, help="最终输出文件名（不含 .pkl）")

# ── V3: 扰动实验参数 ──────────────────────────────────────────────────────────
parser.add_argument("--fault_type", type=str, default="none",
    choices=["none", "friction", "force_x"],
    help="扰动类型: none=纯安全数据, friction=V1摩擦力变化, force_x=V2持续阻力")
parser.add_argument("--t_fault_min", type=int, default=30,
    help="扰动生效最早步数（含），建议 ≥ 20")
parser.add_argument("--t_fault_max", type=int, default=60,
    help="扰动生效最晚步数（含），建议 ≤ max_steps * 0.4")
parser.add_argument("--safe_ratio", type=float, default=0.2,
    help="每 episode 以此概率不施加扰动（纯安全轨迹比例）")
# V1: 摩擦力变化（每 episode 在范围内随机采样，增加扰动多样性）
parser.add_argument("--ood_static_friction_range", type=float, nargs=2, default=[0.08, 0.12],
    metavar=('LOW', 'HIGH'),
    help="V1: OOD 静摩擦系数范围 [LOW HIGH]，每 episode 均匀采样（训练正常范围 0.6-1.0）")
parser.add_argument("--ood_dynamic_friction_range", type=float, nargs=2, default=[0.04, 0.06],
    metavar=('LOW', 'HIGH'),
    help="V1: OOD 动摩擦系数范围 [LOW HIGH]，每 episode 均匀采样（训练正常范围 0.4-0.8）")
parser.add_argument("--ood_dynamic_ratio_range", type=float, nargs=2, default=None,
    metavar=('LOW', 'HIGH'),
    help="V1 friction-adaptive: 动摩擦/静摩擦比值范围；设置后 dynamic_mu=static_mu×ratio，"
         "优先于 --ood_dynamic_friction_range。建议 0.6 0.8。")
# V2: 持续阻力
parser.add_argument("--force_x_magnitude_min", type=float, default=-20.0,
    help="V2: base -x 方向持续力（牛顿，负值=阻力），建议 -30 ~ -80 N")
parser.add_argument("--force_x_magnitude_max", type=float, default=-80.0,
    help="V2: base -x 方向持续力（牛顿，负值=阻力），建议 -30 ~ -80 N")
parser.add_argument("--force_x_ramp_steps", type=int, default=0,
    help="V2: 从 0 线性斜坡到目标力值所需步数（0=瞬间施加）")
parser.add_argument("--min_fail_consecutive", type=int, default=5,
    help="V1/V2: 连续 failure 帧数达到此阈值才保留标签，消除速度噪声误标（0=不平滑）")
# Braking: 正常制动采集，不作为 fault_type，不改变 pkl demo 结构
parser.add_argument("--collection_mode", type=str, default="normal",
    choices=["normal", "braking"],
    help="数据采集模式: normal=原逻辑, braking=制动轨迹")
parser.add_argument("--brake_start_range", type=int, nargs=2, default=[50, 80],
    metavar=('LOW', 'HIGH'),
    help="braking: 制动开始步数范围 [LOW HIGH]，每 episode 均匀采样")
parser.add_argument("--brake_complete_ratio", type=float, default=0.5,
    help="braking: 完全制动轨迹比例；完全制动 target_vx=0，不完全制动 target_vx 从范围采样")
parser.add_argument("--brake_target_ratio_range", type=float, nargs=2, default=[0.2, 0.6],
    metavar=('LOW', 'HIGH'),
    help="braking: 不完全制动保留原始 vx command 的比例范围 [LOW HIGH]，例如 0.2 0.6")
# Stage-2 low-friction-brake: braking + fault_type=friction 组合模式
parser.add_argument("--post_fault_brake_delay_range", type=int, nargs=2, default=[15, 40],
    metavar=('LOW', 'HIGH'),
    help="low_friction_brake: t_brake = t_fault + delay，delay 从 [LOW HIGH] 均匀采样。"
         "保证先在低摩擦上走一段(检测延迟窗口)再制动。仅 collection_mode=braking & fault_type=friction 生效。")
parser.add_argument("--brake_trigger", choices=["frame", "edge"], default="frame",
    help="braking 触发方式：frame=旧的固定帧；edge=沿朝向到平台边缘的距离触发。")
parser.add_argument("--brake_margin_ratio_range", type=float, nargs=2, default=[0.4, 1.6],
    metavar=('LOW', 'HIGH'),
    help="edge brake: 每 episode 的 r 范围，d*=r×d_hat_ref(v)。")
parser.add_argument("--fault_trigger", choices=["frame", "window"], default="frame",
    help="friction fault 触发方式：frame=旧的固定帧；window=在 brake 前固定时间窗口触发。")
parser.add_argument("--fault_window_s", type=float, default=0.60,
    help="window fault: fault 到 edge brake 的目标时间窗口（秒）。")
parser.add_argument("--spawn_mode", choices=["range", "solve"], default="range",
    help="range=沿用环境 reset spawn；solve=由 (v,r,theta) 反解 episode spawn。")
parser.add_argument("--platform_half_width", type=float, default=4.0,
    help="edge/solve 使用的正方形平台半宽（米）；8m platform 应设为 4.0。")
parser.add_argument("--terrain_profile", choices=["task", "cliff_brake", "flat_mesh", "robotlab_train", "gx_pilot", "gx_drop05", "gx_main_small", "gx_main_big", "gx_e4"], default="task",
    help="采集时覆盖 task 的地形：cliff_brake=CLIFF_DETECTION_TERRAINS_CFG（10x10=100 格，"
         "platform/pit/stairs/slope 按 proportion 随机采样，实际配比在启动日志里打印）；"
         "flat_mesh=无边缘平地；robotlab_train=ROBOTLAB_TRAIN_STAIRS_SLOPE_CFG（RobotLab 训练用的"
         "上楼梯 0.05–0.257 m / 上坡 0.1–0.568，平台半宽 1.5 m，配 --platform_half_width 1.5）；task=不覆盖。")
parser.add_argument("--spawn_accel", type=float, default=2.6,
    help="solve spawn 使用的实测加速度（m/s^2）。")
parser.add_argument("--spawn_settle_s", type=float, default=0.25,
    help="solve spawn 中加速结束后的稳定余量（秒）。")
parser.add_argument("--tracking_error_ratio", type=float, default=0.15,
    help="window/edge trigger: |v_x-v_cmd|/v_cmd 的稳定判据（与绝对阈值取较大者）。")
parser.add_argument("--tracking_error_abs", type=float, default=0.10,
    help="window/edge trigger: 速度跟随稳定判据的绝对误差下限（m/s）。")
parser.add_argument("--tracking_hold_s", type=float, default=0.20,
    help="window/edge trigger: 速度误差连续满足判据至少多久后，才允许触发低摩擦。")
parser.add_argument("--spawn_heading_range", type=float, nargs=2, default=[-0.785398, 0.785398],
    metavar=('LOW', 'HIGH'),
    help="solve spawn: 相对目标 +x 边法线的初始 yaw 范围（弧度）。")
parser.add_argument("--spawn_lateral_margin", type=float, default=0.4,
    help="solve spawn: 射线交点与平台侧边至少保留的距离（米）。")
parser.add_argument("--depth_uint8", action="store_true", default=False,
    help="深度图以 uint8 存储（内存/磁盘 4 倍压缩）。深度传感器裁剪范围恰为 "
         "[--depth_clip_min, --depth_clip_max]，量化分辨率约 1.2 cm、最大误差 0.6 cm，"
         "而本任务最小相关尺度是 d_hold≈0.5 m，余量 80 倍。255 保留给无回波(inf)，"
         "顺带消除了原始数据里 24-27% 的非有限像素。")
parser.add_argument("--depth_clip_min", type=float, default=0.30,
    help="uint8 量化的近端裁剪（米）。")
parser.add_argument("--depth_clip_max", type=float, default=3.00,
    help="uint8 量化的远端裁剪（米）。")
parser.add_argument("--early_stop_hold_steps", type=int, default=0,
    help="刹停确认后再记录多少帧就结束 episode（0=关闭，跑满预算）。"
         "刹停后的站立占全部帧的 29%%，是最大的一块浪费；但 drift_off 的"
         "刹停->翻覆延迟中位 66 帧、90 分位 132 帧，留 90 帧可捕捉约 67%%。")
parser.add_argument("--fault_window_range", type=float, nargs=2, default=None,
    metavar=('LOW', 'HIGH'),
    help="window fault: 逐 episode 采样 fault->brake 的时间窗口（秒），覆盖 --fault_window_s。"
         "更长的窗口只是让 spawn 后退（D0 里的 v*(settle+window) 项），不改变 r，"
         "所以这是在刹车块内部拿到不同时长低摩擦行走的免费办法。上限受跑道约束："
         "v=2.5 时 1.2 s 要 3.0 m，再长会在高速端不可行从而制造 v-window 相关。")
parser.add_argument("--spawn_extra_runway", type=float, nargs=2, default=[0.0, 0.0],
    metavar=('LOW', 'HIGH'),
    help="solve spawn: 解算跑道之外额外增加的巡航距离（米），每 episode 独立于 v 采样。"
         "触发是闭环的，额外跑道不改变 r，只用于解除 (d_edge, v) 共线。")
parser.add_argument("--terrain_seed", type=int, default=None,
    help="固定地形生成的随机种子并启用 mesh cache。不设时地形随全局 seed 变，"
         "各块（B1/B2/B3...）拿到的地形配比与每格 difficulty 都不同；设成同一个值则"
         "七条命令共用完全相同的地形，跨块可比性最好，且第二次起从 cache 加载、"
         "省掉整个 mesh 生成。cache 按 (子地形 cfg + difficulty) 哈希，逐格独立，"
         "不会把不同 difficulty 的格子合并。")
parser.add_argument("--spawn_max_retries", type=int, default=50,
    help="solve spawn: 不可行时重采 (v, r, theta, extra) 的最大次数；耗尽才报错。")
parser.add_argument("--dynamic_max_steps", action="store_true", default=False,
    help="按 (v, D0) 逐 episode 解算步数上限；--max_steps 退化为硬上限。")
parser.add_argument("--post_brake_budget_s", type=float, default=3.0,
    help="dynamic_max_steps: t_brake 之后预留的刹停+驻留时间（秒）。")
parser.add_argument("--stop_speed_thresh", type=float, default=0.20,
    help="outcome 分类: 判定刹停的平面速度阈值（m/s）。")
parser.add_argument("--stop_sustain_steps", type=int, default=5,
    help="outcome 分类: 速度需连续低于阈值的帧数。")
parser.add_argument("--brake_ref_mode", choices=["mu_blind", "mu_aware"], default="mu_blind",
    help="起刹调度用哪把尺子。mu_blind=旧行为，d*=r*d_hat_ref(v)，与摩擦无关；"
         "mu_aware=d*=r*(d_brake(mu_dyn,v) + --brake_ref_hold)。"
         "为什么要改：mu_blind 下三档摩擦的锚点落在同一个绝对距离(实测起刹中位 1.03-1.15 m)，"
         "而各自真实需求是 1.34/0.95/0.64/0.35 m —— 高摩擦侧 margin<0 只占 5.8%，"
         "模型学到低摩擦那条边界然后套给所有摩擦(docs/68 §6.1)。"
         "默认保持 mu_blind，不写这一行时行为逐字节不变。")
parser.add_argument("--brake_ref_dyn_a", type=float, default=0.0388,
    help="mu_aware: d_brake(mu,v)=a*v^2/mu + b*v/mu + c*v 的 a。"
         "拟合自 go2_fricsweep_2700ep 的 1476 条非摔倒制动 episode(录制 t_brake->t_stop 的速度积分)，"
         "R^2=0.927 MAE=0.069m，20 个 mu_dyn x v 格残差<=0.060m。见 latent-safety/results/d_brake_fit_muv.json。")
parser.add_argument("--brake_ref_dyn_b", type=float, default=0.0848,
    help="mu_aware: b。注意 v/mu 项在常用速度段是主导项(mu=0.17,v=1.0 时 0.50m vs v^2/mu 的 0.23m)——"
         "制动不是恒定最大减速度，指令斜坡下降那段距离也随 1/mu 放大。")
parser.add_argument("--brake_ref_dyn_c", type=float, default=-0.0109,
    help="mu_aware: c（与 mu 无关的一次项，实测为小负值）。")
parser.add_argument("--brake_ref_hold", type=float, default=0.35,
    help="mu_aware: d_hold，刹停后站稳所需的离边余量(米)。**加在 d_brake 上，不是取 max**：\n"
         "  d* = r * (d_brake(mu_dyn, v) + d_hold)\n"
         "d_brake 只拟合了 t_brake->t_stop 的滑行距离，不含停稳余量，所以真实存活边界\n"
         "不在 margin=0 而在 margin≈+0.21m —— 实测 go2_fricsweep_2700ep 的 2334 条：\n"
         "  margin [-0.2,0) 摔 66.1% | [0,0.2) 62.0% | [0.2,0.4) 37.3% | [0.4,0.6) 21.3%\n"
         "  | [0.6,0.8) 13.0% | [1.5,inf) 15.9%(与边缘无关的本底)\n"
         "把 d_hold 加进去之后 margin=(r-1)*(d_brake+d_hold)，以存活边界为中心。\n"
         "取值(用实测 p_fall(margin) 曲线模拟, r 0.4-1.6, 高 mu 档):\n"
         "  0.21 -> 摔 46%, IQR[33,61] | 0.35 -> 摔 37%, IQR[19,55] | 0.50 -> 摔 31%, IQR[15,45]\n"
         "  对照 旧 mu_blind -> 摔 23%, IQR[14,25]（只有 11 点对比度）\n"
         "取 0.35：三档摔倒率都在 37-39%，且档内展幅最宽(36 点)——边界被两侧密集探测。\n"
         "**别用取 max 的写法**：那会把分布中心压到 margin=0，实测 89% 全摔，同样没有对比。")
parser.add_argument("--brake_ref_fallback_mu_dyn", type=float, default=0.60,
    help="mu_aware 且本 episode 没有 friction fault(fault_type!=friction 或 safe 轨迹)时，"
         "调度用的名义动摩擦。地形本身的 DR 动摩擦范围是 0.4-0.8。")
parser.add_argument("--brake_ref_linear", type=float, default=0.59,
    help="参考制动曲线一次项 a1：d_hat_ref(v)=a1*v+a2*v^2。")
parser.add_argument("--brake_ref_quadratic", type=float, default=0.144,
    help="参考制动曲线二次项 a2：d_hat_ref(v)=a1*v+a2*v^2+c。")
parser.add_argument("--brake_ref_offset", type=float, default=0.0,
    help="参考制动曲线常数项 c（米）= d_hold，刹停后站稳所需的最小离边距离。"
         "实测 ~0.50 m 且不随速度缩放：低速时若不加这一项，触发阈值会短于机身长度，"
         "低速段会同时爆出 off_before_brake 和 drift_off。")
parser.add_argument("--command_vx_range", type=float, nargs=2, default=None,
    metavar=('LOW', 'HIGH'), help="可选：覆盖 task 的前向速度指令范围。")
parser.add_argument("--command_vy_range", type=float, nargs=2, default=None,
    metavar=('LOW', 'HIGH'), help="可选：覆盖 task 的横向速度指令范围。")
parser.add_argument("--command_yaw_rate_range", type=float, nargs=2, default=None,
    metavar=('LOW', 'HIGH'), help="可选：覆盖 task 的 yaw-rate 指令范围。")
parser.add_argument("--lowfric_walk", action="store_true", default=False,
    help="Stage-2 type-5 低摩擦 walk(z0/WM 表征用): normal+friction 下改用 fall/collision 语义"
         "(只有真摔=failure，打滑但没摔≠failure)，并写 fault_type='low_friction'/event_type='low_friction_walk'"
         "→ 训练侧走 fall 分支，保住 margin 的分级(否则低摩擦走路会被全标 unsafe)。")
parser.add_argument("--robotlab_policy", type=str, default=None,
    help="RobotLab MoE-CTS 导出的 TorchScript student（exported/policy.pt）。给定时跳过 OnPolicyRunner，"
         "用 utils/robotlab_policy.py 做关节置换 + 历史重排后批量推理。需配合 --task go2_data_collection_robotlab；"
         "见 docs/ROBOTLAB_POLICY_COLLECTION.md。")
parser.add_argument("--diag_termination", action="store_true", default=False,
    help="诊断：每次接触类提前终止时，记录触发的 body、头部受力（机身坐标系）以及与腿部受力的"
         "牛顿第三定律配对（区分头撞地形 vs 被自己的腿碰到），写入 <output_dir>/termination_diag.jsonl。")
parser.add_argument("--store_geo", action="store_true", default=False,
    help="docs/82 §6.4：逐帧存 root_pos_w/root_quat_w/foot_pos_w/foot_contact/env_origin 与"
         "（终止前快照的）base/Head 受力、foot_force_w 到 demo['geo']；启动时把地形 mesh 导出到"
         "<output_dir>/terrain_mesh.npz，运行配置写到 <output_dir>/run_config.json。")
parser.add_argument("--terminate_base_only", action="store_true", default=False,
    help="本次运行只用 base 接触终止（与 RobotLab 训练一致），头部接触不终止；不改任务配置。")
parser.add_argument("--log_head_contact", action="store_true", default=False,
    help="全程逐步记录每个 env 的头部受力（Head_upper/Head_lower，取最近接触历史最大值）、"
         "机身位置与 episode 步数，定期写入 <output_dir>/head_contact_log.npz。")
parser.add_argument("--camera_mount", choices=["offset", "d435"], default="offset",
    help="深度相机挂载。offset = 任务默认的非实体偏置 (0.33, 0, 0.08)；d435 = 实体 D435 USD（旧 go2_data_collection / "
         "E4 fricsweep 数据的挂载方式，sensor 在 base/d435/front_cam、自身 offset 为单位变换）。"
         "两者图像分布不同（docs/current/00 速查卡『v1 数据 vs 旧数据 obs』），同一 WM 的数据必须统一。")
parser.add_argument("--actuator", choices=["dcmotor", "go2hv"], default="dcmotor",
    help="dcmotor=任务配置自带的执行器（LeggedLab DCMotor，23.5 Nm，扭矩从 0 rad/s 起线性下降）；"
         "go2hv=RobotLab v1 训练用的 UnitreeActuatorCfg_Go2HV（20.2/23.4 Nm，13.5 rad/s 内满扭矩，"
         "关节摩擦 0.01，0–4 物理步延迟），见 assets/unitree/unitree_actuator.py。")
parser.add_argument("--base_mass_range", type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
    help="可选：覆盖 add_base_mass 的附加质量范围（kg，startup 采样、每 env 固定）。"
         "默认沿用 task 配置（数据采集 EventCfg 为 ±5）；RobotLab 训练为 ±1。")
# ─────────────────────────────────────────────────────────────────────────────

import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets.articulation import Articulation

import re as _re
def get_checkpoint_path(log_path, run_dir=".*", checkpoint=".*", other_dirs=None, sort_alpha=True):
    try:
        import os as _os
        runs = [_os.path.join(log_path, r) for r in _os.scandir(log_path) if r.is_dir() and _re.match(run_dir, r.name)]
        runs.sort() if sort_alpha else runs.sort(key=_os.path.getmtime)
        run_path = _os.path.join(runs[-1], *other_dirs) if other_dirs else runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in '{log_path}' matching '{run_dir}'.")
    model_checkpoints = [f for f in _os.listdir(run_path) if _re.match(checkpoint, f)]
    if not model_checkpoints:
        raise ValueError(f"No checkpoints in '{run_path}' matching '{checkpoint}'.")
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    return _os.path.join(run_path, model_checkpoints[-1])

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


# ========== 数据缓冲区 ==========

class TrajectoryBuffer:
    """单条轨迹缓冲区，格式与 v2 / generate_data_traj_cont.py 一致"""
    """
    单条轨迹的缓冲区，对应 generate_data_traj_cont.py 的一条 demo
    
    数据格式：
    demo = {
        'obs': {
            'image': [img_t0, img_t1, ...],        # List[np.ndarray]
            'state': [state_t0, state_t1, ...],    # List[np.ndarray] (42,): ang_vel, proj_grav, jpos, jvel, last_action
            'priv_state': [priv_t0, priv_t1, ...]  # List[np.ndarray] (10,): base_lin_vel_b(3), feet_contact(4), static_mu(1), dynamic_mu(1), applied_force_x(1)
        },
        'rewards':[r_t0, r_t1, ...],               # List[float]
        'actions': [cmd_t0, cmd_t1, ...],          # List[np.ndarray] (3,): 速度指令 (vx, vy, yaw_rate)，WM 的控制输入
        'dones': [0, 0, ..., 1]                    # List[int]
        'failure': [0, 0, ..., 0] or [0, 0, ..., 1]
    }
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_list: List[np.ndarray] = []
        self.state_list: List[np.ndarray] = []
        self.priv_state_list: List[np.ndarray] = []
        self.reward_list: List[float] = []
        self.action_list: List[np.ndarray] = []
        self.done_list: List[int] = []
        self.failure_list: List[int] = []
        self.d_edge_list: List[float] = []
        self.d_edge_ray_list: List[float] = []
        self.base_z_list: List[float] = []
        # 平地参考曲线用：d_brake = |p(t_stop) - p(t_brake)|，避免速度积分的累积
        # 误差，也避免 |v_xy| 积分把路径长度当成位移。local frame = 相对 env origin。
        self.root_x_list: List[float] = []
        self.root_y_list: List[float] = []
        self.yaw_list: List[float] = []
        self.step_count = 0
        self.geo_lists: Dict[str, List[np.ndarray]] = {}
        # 临时存储（step 内跨阶段传递）
        self._pending_obs = None
        self._pending_command = None

    def add_step(self, image, state, priv_state, reward, action, done, failure,
                 d_edge=float("nan"), d_edge_ray=float("nan"), base_z=float("nan"),
                 root_x=float("nan"), root_y=float("nan"), yaw=float("nan")):
        self.image_list.append(image)
        self.state_list.append(state)
        self.priv_state_list.append(priv_state)
        self.reward_list.append(reward)
        self.action_list.append(action)
        self.done_list.append(done)
        self.failure_list.append(failure)
        self.d_edge_list.append(float(d_edge))
        self.d_edge_ray_list.append(float(d_edge_ray))
        self.base_z_list.append(float(base_z))
        self.root_x_list.append(float(root_x))
        self.root_y_list.append(float(root_y))
        self.yaw_list.append(float(yaw))
        self.step_count += 1

    def add_geo(self, geo: Dict[str, np.ndarray]):
        """--store_geo 的逐帧几何/接触字段；与 add_step 同帧调用。"""
        for k, v in geo.items():
            self.geo_lists.setdefault(k, []).append(np.asarray(v, dtype=np.float32))

    def to_demo(self, env_id=None):
        demo = {
            'obs': {
                'image': self.image_list.copy(),
                'state': self.state_list.copy(),
                'priv_state': self.priv_state_list.copy()
            },
            'rewards': self.reward_list.copy(),
            'actions': self.action_list.copy(),
            'dones': self.done_list.copy(),
            'failure': self.failure_list.copy(),
            'd_edge': self.d_edge_list.copy(),
            'd_edge_ray': self.d_edge_ray_list.copy(),
            'base_z': self.base_z_list.copy(),
            'root_x': self.root_x_list.copy(),
            'root_y': self.root_y_list.copy(),
            'yaw': self.yaw_list.copy(),
        }
        if env_id is not None:
            demo['env_id'] = env_id
        if self.geo_lists:
            demo['geo'] = {k: np.stack(v) for k, v in self.geo_lists.items()}
        return demo

    def __len__(self):
        return self.step_count


class BatchedEnvData:
    """批量预提取所有环境的数据（单次 CUDA→CPU，避免 per-env sync）"""
    def __init__(self):
        self.ang_vel = None
        self.projected_gravity = None
        self.joint_pos = None
        self.joint_vel = None
        self.last_action = None
        self.base_lin_vel_w = None
        self.base_lin_vel_b = None
        self.root_pos_w = None
        self.root_quat_w = None
        self.feet_contact = None
        self.commands = None

    def fetch(self, env, robot: Articulation):
        self.ang_vel = robot.data.root_ang_vel_b.cpu().numpy()
        self.projected_gravity = robot.data.projected_gravity_b.cpu().numpy()
        self.joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos).cpu().numpy()
        self.joint_vel = robot.data.joint_vel.cpu().numpy()
        self.last_action = env.action_buffer._circular_buffer.buffer[:, -1, :].cpu().numpy()
        self.base_lin_vel_w = robot.data.root_lin_vel_w.cpu().numpy()
        self.base_lin_vel_b = robot.data.root_lin_vel_b.cpu().numpy()
        self.root_pos_w = robot.data.root_pos_w.cpu().numpy()
        self.root_quat_w = robot.data.root_quat_w.cpu().numpy()
        contact_forces = env.contact_sensor.data.net_forces_w_history
        feet_body_ids = env.feet_cfg.body_ids if hasattr(env, "feet_cfg") else slice(0, 4)
        feet_contact_t = (torch.max(torch.norm(contact_forces[:, :, feet_body_ids], dim=-1), dim=1)[0] > 0.5)
        self.feet_contact = feet_contact_t.cpu().numpy().astype(np.float32)
        self.commands = env.command_generator.command.cpu().numpy()


def extract_state_vector(batch: BatchedEnvData, env_idx: int) -> np.ndarray:
    return np.concatenate([
        batch.ang_vel[env_idx],
        batch.projected_gravity[env_idx],
        batch.joint_pos[env_idx],
        batch.joint_vel[env_idx],
        batch.last_action[env_idx]
    ])


def extract_priv_state_vector(batch: BatchedEnvData, env_idx: int) -> np.ndarray:
    """v2 兼容格式：R^7 (base_lin_vel_b + feet_contact)"""
    return np.concatenate([
        batch.base_lin_vel_b[env_idx],
        batch.feet_contact[env_idx]
    ])


def extract_priv_state_vector_v3(batch: BatchedEnvData, env_idx: int,
                                  foot_static_friction: float,
                                  foot_dynamic_friction: float,
                                  applied_force_x: float) -> np.ndarray:
    """
    V3 friction-adaptive 格式：R^10
      base_lin_vel_b       (3): 基座坐标系下 base 线速度
      feet_contact         (4): 四足接触状态（二值 0/1）
      foot_static_friction (1): 足部静摩擦系数（DR 采样值或 OOD 值；multiply 模式下等于有效摩擦）
      foot_dynamic_friction(1): 足部动摩擦系数
      applied_force_x      (1): base x 向外力（N；安全帧为 0，V2 触发后为负值阻力）
    """
    return np.concatenate([
        batch.base_lin_vel_b[env_idx],
        batch.feet_contact[env_idx],
        np.array([foot_static_friction], dtype=np.float32),
        np.array([foot_dynamic_friction], dtype=np.float32),
        np.array([applied_force_x], dtype=np.float32)
    ])


def quantize_depth(img: np.ndarray) -> np.ndarray:
    """Depth -> uint8 with 255 reserved for "no return".

    The sensor already clips to [clip_min, clip_max] and returns inf beyond it,
    so a quarter of every frame is non-finite in the float32 data.  Mapping the
    finite band onto 1..254 and inf onto 255 keeps the encoding monotone in
    distance (near = small, far = large, no-return = largest) and removes the
    non-finite values that every downstream consumer would otherwise have to
    special-case.
    """
    lo, hi = float(args.depth_clip_min), float(args.depth_clip_max)
    a = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(a)
    q = np.empty(a.shape, dtype=np.uint8)
    q[~finite] = 255
    if finite.any():
        v = np.clip(a[finite], lo, hi)
        q[finite] = np.round((v - lo) / max(hi - lo, 1e-9) * 254.0).astype(np.uint8)
    return q


def extract_image(env, env_idx: int) -> np.ndarray:
    try:
        if hasattr(env.scene, 'sensors') and 'front_camera' in env.scene.sensors:
            camera = env.scene.sensors['front_camera']
            if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                output = camera.data.output
                if 'rgb' in output:
                    return output['rgb'][env_idx].cpu().numpy()
                elif 'rgba' in output:
                    return output['rgba'][env_idx].cpu().numpy()[..., :3]
                elif 'distance_to_image_plane' in output:
                    depth = output['distance_to_image_plane'][env_idx].cpu().numpy()
                    return quantize_depth(depth) if args.depth_uint8 else depth
    except Exception as e:
        print(f"[WARN] extract_image env{env_idx}: {e}")
    h = env.cfg.scene.camera.height
    w = env.cfg.scene.camera.width
    return np.zeros((h, w, 1), dtype=np.uint8 if args.depth_uint8 else np.float32)


def atomic_save_shard(demos: List[Dict], shard_path: Path):
    tmp_path = shard_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(demos, f, protocol=4)
    tmp_path.rename(shard_path)
    print(f"   [分片保存] → {shard_path.name}  ({len(demos)} eps, "
          f"{shard_path.stat().st_size / 1024 / 1024:.1f} MB)")


def count_outcome_stats(demos: List[Dict]) -> Dict[str, int]:
    """Tally the 3-way braking outcome so a run reports why it failed, not just that."""
    tally: Dict[str, int] = {}
    for d in demos:
        key = d.get('outcome')
        if key:
            tally[key] = tally.get(key, 0) + 1
    return tally


def _report_outcomes(tally: Dict[str, int], retries: List[int]) -> None:
    """Print the outcome split and the spawn rejection rate.

    A high rejection rate is not an error, it means the requested (v, r) box sticks
    out of the platform's feasible region -- the r marginal then depends on v, which
    is exactly the confound the controlled schedule exists to remove.  Surface it.
    """
    total = sum(tally.values())
    if total:
        parts = ", ".join(f"{k}={v} ({v / total:.1%})"
                          for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
        print(f"   outcome 分布: {parts}")
    if retries:
        rejected = sum(1 for r in retries if r > 0)
        print(f"   spawn 重采: {rejected}/{len(retries)} episode 需要重采 "
              f"({rejected / len(retries):.1%}), 平均 {sum(retries) / len(retries):.2f} 次/episode")
        if rejected / len(retries) > 0.10:
            print("   [WARN] 重采率 >10%: (v, r) 区间超出平台可行域，r 的边缘分布会依赖 v。"
                  "收窄 r 上沿或降低速度上限。")


def count_braking_stats(demos: List[Dict]):
    """返回 braking safe / fail reset 计数。"""
    n_total = len(demos)
    n_failed = sum(1 for d in demos if d.get('failure', [0]) and bool(d['failure'][-1]))
    n_safe = n_total - n_failed
    return n_safe, n_failed


def print_braking_stats(demos: List[Dict]):
    """打印 braking safe / fail reset 比例。"""
    n_safe, n_failed = count_braking_stats(demos)
    n_total = n_safe + n_failed
    denom = max(n_total, 1)
    print(f"   braking 统计: BRAKE_SAFE={n_safe}/{n_total} ({n_safe / denom:.1%}), "
          f"FAIL_RESET={n_failed}/{n_total} ({n_failed / denom:.1%})")


# ========== failure 标签后处理 ==========

def smooth_failure_labels(failure_list: List[int], min_consecutive: int) -> List[int]:
    """
    对 per-step 原始 failure 信号做连续帧平滑：
    只有连续 >= min_consecutive 帧的 failure run 才保留，孤立帧清零。
    episode 结束后对完整序列调用，避免速度噪声导致误标。

    示例 (min_consecutive=3):
      输入: [0,0,1,0,1,1,1,1,0,1,0]
      输出: [0,0,0,0,1,1,1,1,0,0,0]  # 长度>=3的run保留，短run清零
    """
    if min_consecutive <= 1:
        return list(failure_list)
    n = len(failure_list)
    result = [0] * n
    i = 0
    while i < n:
        if failure_list[i] == 1:
            j = i
            while j < n and failure_list[j] == 1:
                j += 1
            if (j - i) >= min_consecutive:
                for k in range(i, j):
                    result[k] = 1
            i = j
        else:
            i += 1
    return result


# ========== V2 兼容：原始 failure 判断（fault_type='none' 时使用）==========

def extract_failure(env, env_idx: int, is_done: bool, is_timeout: bool) -> int:
    """v2 原有逻辑：提前 reset（非 timeout）视为 failure"""
    if is_done and not is_timeout:
        return 1
    return 0


# ========== V3 新增：扰动相关函数 ==========

def apply_friction_fault(robot: Articulation, env_idx: int,
                          static_mu: float, dynamic_mu: float):
    """V1: 将指定 env 的所有材质摩擦系数修改为 OOD 值"""
    mats = robot.root_physx_view.get_material_properties()  # numpy (N, num_shapes, 3)
    mats[env_idx, :, 0] = static_mu
    mats[env_idx, :, 1] = dynamic_mu
    robot.root_physx_view.set_material_properties(
        mats, torch.tensor([env_idx], dtype=torch.int32)
    )


def restore_friction(robot: Articulation, env_idx: int, original_materials: np.ndarray):
    """V1: 将指定 env 的材质恢复为原始摩擦系数"""
    mats = robot.root_physx_view.get_material_properties()
    mats[env_idx] = original_materials[env_idx]
    robot.root_physx_view.set_material_properties(
        mats, torch.tensor([env_idx], dtype=torch.int32)
    )


def compute_vel_failure(batch: BatchedEnvData, env_idx: int,
                         current_step: int, t_fault: int,
                         is_safe: bool, v_cmd: np.ndarray,
                         applied_force_x: float = 0.0) -> int:
    """
    V1/V2 失效标签：基于 3D 速度跟随误差（任意分量超标 → failure=1）

    Grace period: t < t_fault + 5 时返回 0，避免过渡帧被误标为 failure。

    分量阈值：
      ex  = |v_cmd_x - v_x| / max(v_cmd_x, 0.5) > 0.3  （相对阈值，兜底 0.5m/s）
      ey  = |v_y| > 0.25 m/s                             （绝对阈值，避免正常步态侧摆误报）
      eyaw = |ang_vel_z| > 0.3 rad/s                     （绝对阈值）
    """
    if is_safe or current_step < t_fault + 5:
        return 0
    v = batch.base_lin_vel_b[env_idx]  # base frame velocity
    ex = abs(float(v_cmd[0]) - float(v[0])) / max(float(v_cmd[0]), 0.5) > 0.3
    ey = abs(float(v[1])) > 0.25
    eyaw = abs(float(batch.ang_vel[env_idx][2])) > 0.3
    print(f"   [Failure Check] env{env_idx} step={current_step} force_x={applied_force_x:.1f}N v_cmd={v_cmd} v={v} ex={ex} ey={ey} eyaw={eyaw}")
    return 1 if (ex or ey or eyaw) else 0


# ========== braking：command schedule ==========

def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Return world yaw from an Isaac Lab quaternion in (w, x, y, z) order."""
    w, x, y, z = (float(q) for q in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# 碰撞型地形（撞墙失效）：先碰到的是机头，不是 base。起刹调度与 spawn solve 统一用
# 「机头到墙」的距离 d_ray − HEAD_REACH_M，使 r 在方箱（base 到边）和坑（机头到墙）上含义一致。
# 0.35 = latent-safety docs/78 拟合的 GX_BODY_HALF_LEN（pit 失效位置 R² 0.916）。
COLLIDE_TERRAIN_KEYS = ("step_pit", "deep_pit", "stairs")
HEAD_REACH_M = 0.35
# 坠落型地形（平台边）：停住时前脚落在 base 前方 0.19–0.32 m（gx_main 试运行实测）。base 离边 0.14–0.16 m
# 停住时最靠边的脚已在边外 0.03–0.11 m，约 50 帧后掉下（drift_off）。起刹调度与 spawn solve 统一用
# 「前脚到边」的距离 d_ray − FOOT_REACH_M，使 r = 1 对应前脚恰好停在边上。
FALL_TERRAIN_KEYS = ("drop_box", "box_low", "box_high", "dangerous_platform")
FOOT_REACH_M = 0.25


def leading_contact_reach(terrain_key: str) -> float:
    """起刹调度用的「最先越界的身体部位」相对 base 的前伸：墙 = 机头，平台边 = 前脚。"""
    if terrain_key in COLLIDE_TERRAIN_KEYS:
        return HEAD_REACH_M
    if terrain_key in FALL_TERRAIN_KEYS:
        return FOOT_REACH_M
    return 0.0

# 子地形的危险方向：box 是向下掉落，pit / inverted-pyramid stairs 的 origin z 为负，
# 机器人生成在凹陷底部，越过边界是向上撞台阶。二者的余量几何相同（都是 platform_width
# 的矩形边界），失败机制不同 —— 记录下来供分层，不要混在一个标签里分析。
HAZARD_DIRECTION = {
    "dangerous_platform": -1,   # 悬崖：越界 = 坠落
    "deep_pit": +1,             # 凹坑底：越界 = 撞上升台阶
    "stairs": +1,               # 倒金字塔阶梯底：越界 = 撞上升台阶
    "slope": +1,                # 倒金字塔斜坡：inverted=True 平台在最低处，越界 = 上坡
    "flat": 0,
    # 静态高程图 g_x 采集（GX_PILOT / GX_MAIN_*）
    "drop_box": -1, "box_low": -1, "box_high": -1,
    "step_pit": +1,
    "slope_up": +1, "slope_down": -1,
}


def classify_braking_outcome(priv_states, d_edge_list, failure_list, t_brake,
                             stop_thresh: float, sustain: int):
    """Split a braking episode into stop-timing and outcome.

    ``FAIL_RESET`` conflates two different mechanisms: running past the edge while
    still moving (the margin was too small) and coming to rest inside the platform
    then creeping over it afterwards (the margin was fine, the hold failed).  They
    need opposite fixes, so they get separate labels.
    """
    T = len(failure_list)
    out = {"outcome": "unknown", "t_stop": -1,
           "d_edge_at_stop": float("nan"), "t_first_failure": -1}
    fails = [i for i, f in enumerate(failure_list) if f]
    out["t_first_failure"] = fails[0] if fails else -1
    braked = t_brake is not None and 0 <= t_brake < T
    # 顺序要紧：触发从未点火而机器人已经出界，那是 off_before_brake，不是 no_brake。
    # 反过来先判 t_brake 会把真正的余量失败记成"没刹车"，低速段整段被误分类。
    if fails and (not braked or fails[0] < t_brake):
        out["outcome"] = "off_before_brake"
        return out
    if not braked:
        out["outcome"] = "no_brake"
        return out
    speed = [float(np.linalg.norm(np.asarray(p)[:2])) for p in priv_states]
    t_stop = -1
    for t in range(t_brake, max(t_brake, T - sustain)):
        if all(speed[k] < stop_thresh for k in range(t, t + sustain)):
            t_stop = t
            break
    out["t_stop"] = t_stop
    if t_stop >= 0:
        out["d_edge_at_stop"] = float(d_edge_list[t_stop])
    if t_stop < 0:
        # 从未刹停：越界了就是余量不够，没越界就是被帧预算截断
        out["outcome"] = "overrun" if fails else "no_stop"
    else:
        # 刹停之后才失败 = 停住了又蹭出去，与余量无关
        out["outcome"] = "drift_off" if (fails and fails[0] >= t_stop) else "safe_stop"
    return out


def edge_distances(position_xy: np.ndarray, yaw: float, half_width: float):
    """Return (Chebyshev margin, forward-ray range) for a square platform.

    The Chebyshev margin is signed and is the labelling quantity.  The ray
    range follows the robot heading and is the edge trigger quantity.
    """
    x, y = float(position_xy[0]), float(position_xy[1])
    d_edge = min(half_width - abs(x), half_width - abs(y))
    c, s = math.cos(yaw), math.sin(yaw)
    candidates = []
    if abs(c) > 1e-9:
        tx = ((half_width if c > 0.0 else -half_width) - x) / c
        if tx >= 0.0:
            candidates.append(tx)
    if abs(s) > 1e-9:
        ty = ((half_width if s > 0.0 else -half_width) - y) / s
        if ty >= 0.0:
            candidates.append(ty)
    d_edge_ray = min(candidates) if candidates else float("inf")
    return float(d_edge), float(d_edge_ray)


def brake_distance_ref(speed: float) -> float:
    """Shared mu-independent scheduling ruler d_hat_ref(v), in metres."""
    v = max(float(speed), 0.0)
    return (args.brake_ref_linear * v + args.brake_ref_quadratic * v * v
            + args.brake_ref_offset)


def brake_distance_dyn(speed: float, mu_dyn: float) -> float:
    """Measured braking distance d_brake(mu_dyn, v), in metres.

    The friction that matters is DYNAMIC: braking is sliding, not sticking.
    Fitting the same form on static mu costs 35% of the accuracy (MAE 0.106 vs
    0.069 m) and its residual still regresses on mu_dyn/mu_static with R^2 0.24.
    """
    v = max(float(speed), 0.0)
    m = max(float(mu_dyn), 1e-3)
    return (args.brake_ref_dyn_a * v * v / m
            + args.brake_ref_dyn_b * v / m
            + args.brake_ref_dyn_c * v)


def brake_trigger_distance(speed: float, ratio: float, mu_dyn: float = None) -> float:
    """The scheduled trigger distance d*.  Single source of truth.

    Every consumer -- the spawn solver, the fault trigger and the brake trigger --
    must use this one function: if the spawn runway is solved with one ruler and
    the trigger fires on another, D0 and d* disagree and the recorded margin is
    not the one that was scheduled.
    """
    if args.brake_ref_mode == "mu_blind":
        return ratio * brake_distance_ref(speed)
    m = args.brake_ref_fallback_mu_dyn if (mu_dyn is None or mu_dyn <= 0.0) else mu_dyn
    # Survival distance, not braking distance: d_brake covers only t_brake -> t_stop,
    # and a robot that stops exactly at the edge still drifts off.  See --brake_ref_hold.
    return ratio * (brake_distance_dyn(speed, m) + float(args.brake_ref_hold))


def solved_runway(speed: float, ratio: float, window: float = None,
                  mu_dyn: float = None) -> float:
    """Initial spawn-to-edge runway D0 for the controlled schedule."""
    v = max(float(speed), 0.0)
    w = args.fault_window_s if window is None else float(window)
    return (v * v / (2.0 * args.spawn_accel)
            + v * (args.spawn_settle_s + w)
            + brake_trigger_distance(v, ratio, mu_dyn))

def compute_braking_command(current_step: int,
                            brake_start_step: int,
                            original_cmd: np.ndarray,
                            target_vx: float) -> np.ndarray:
    """生成单步制动 command；制动后完整覆盖 [vx, vy, yaw]。"""
    if brake_start_step < 0 or current_step < brake_start_step:
        return np.array(original_cmd, dtype=np.float32)
    return np.array([target_vx, 0.0, 0.0], dtype=np.float32)


def apply_command_override_to_obs(env, obs_dict, commands_t: torch.Tensor):
    """
    覆盖 command generator 并同步当前 policy/critic obs 中最新一帧的 command slice.

    BaseEnv 的单帧 actor obs 布局为:
      ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), action(12)
    因此 command slice 是 [6:9]。这里同时 patch obs buffer 和本轮 policy 输入，
    避免 policy 看到随机 command，而 dataset 记录 braking command。
    """
    env.command_generator.command[:, :] = commands_t
    scaled_commands = commands_t * env.obs_scales.commands
    cmd_start = 6
    cmd_end = 9

    actor_buf = env.actor_obs_buffer.buffer
    actor_buf[:, -1, cmd_start:cmd_end] = scaled_commands
    actor_frame_dim = actor_buf.shape[-1]
    actor_hist_len = actor_buf.shape[1]
    actor_flat_start = (actor_hist_len - 1) * actor_frame_dim + cmd_start
    if "policy" in obs_dict and obs_dict["policy"].shape[-1] >= actor_flat_start + 3:
        obs_dict["policy"][:, actor_flat_start:actor_flat_start + 3] = scaled_commands

    if hasattr(env, "critic_obs_buffer") and "critic" in obs_dict:
        critic_buf = env.critic_obs_buffer.buffer
        if critic_buf.shape[-1] >= cmd_end:
            critic_buf[:, -1, cmd_start:cmd_end] = scaled_commands
            critic_frame_dim = critic_buf.shape[-1]
            critic_hist_len = critic_buf.shape[1]
            critic_flat_start = (critic_hist_len - 1) * critic_frame_dim + cmd_start
            if obs_dict["critic"].shape[-1] >= critic_flat_start + 3:
                obs_dict["critic"][:, critic_flat_start:critic_flat_start + 3] = scaled_commands
    return obs_dict


# ========== 主采集函数 ==========

def collect_data():
    if args.collection_mode == 'braking' and args.fault_type == 'force_x':
        raise ValueError("--collection_mode braking 暂不支持 fault_type=force_x；仅支持 none 或 friction(=stage-2 低摩擦制动)。")
    if args.collection_mode == 'braking' and args.fault_type == 'friction':
        print("[INFO] Stage-2 low_friction_brake 组合模式: 先在 t_fault 掉低摩擦，再 t_brake=t_fault+delay 制动。")
    if args.lowfric_walk and not (args.collection_mode == 'normal' and args.fault_type == 'friction'):
        raise ValueError("--lowfric_walk 仅支持 --collection_mode normal --fault_type friction(低摩擦走路,不制动)。")
    if args.lowfric_walk:
        print("[INFO] Stage-2 type-5 低摩擦 walk: fall 语义 + low_friction 元数据(不制动，专供 terrain×low 的 z0/表征)。")
    if args.spawn_accel <= 0.0 or args.platform_half_width <= 0.0:
        raise ValueError("--spawn_accel 与 --platform_half_width 必须为正数。")
    if args.fault_window_s < 0.0 or args.spawn_settle_s < 0.0:
        raise ValueError("--fault_window_s 与 --spawn_settle_s 不能为负数。")
    if min(args.tracking_error_ratio, args.tracking_error_abs, args.tracking_hold_s) < 0.0:
        raise ValueError("--tracking_error_ratio/abs 与 --tracking_hold_s 不能为负数。")
    if args.spawn_mode == "solve" and args.collection_mode != "braking":
        raise ValueError("--spawn_mode solve 只支持 --collection_mode braking。")
    if args.brake_trigger == "edge" and args.collection_mode != "braking":
        raise ValueError("--brake_trigger edge 只支持 --collection_mode braking。")
    if args.fault_trigger == "window" and args.brake_trigger != "edge":
        raise ValueError("--fault_trigger window 必须与 --brake_trigger edge 配合。")
    if args.spawn_mode == "solve" and args.brake_trigger != "edge":
        raise ValueError("--spawn_mode solve 必须与 --brake_trigger edge 配合。")
    if args.brake_trigger == "edge" and args.brake_complete_ratio < 1.0:
        print("[WARN] edge schedule 的参考曲线是完全刹停距离，建议 --brake_complete_ratio 1.0。")

    # ── 环境配置 ──────────────────────────────────────────────────────────────
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    if args.terrain_profile != "task":
        from legged_lab.terrains.terrain_generator_cfg import (
            CLIFF_DETECTION_TERRAINS_CFG,
            FLAT_MESH_TERRAINS_CFG,
            ROBOTLAB_TRAIN_STAIRS_SLOPE_CFG,
            GX_PILOT_TERRAINS_CFG,
            GX_DROP05_TERRAINS_CFG,
            GX_MAIN_SMALL_TERRAINS_CFG,
            GX_MAIN_BIG_TERRAINS_CFG,
            GX_E4_TERRAINS_CFG,
        )
        selected_terrain = {
            "gx_e4": GX_E4_TERRAINS_CFG,
            "gx_main_big": GX_MAIN_BIG_TERRAINS_CFG,
            "gx_main_small": GX_MAIN_SMALL_TERRAINS_CFG,
            "gx_drop05": GX_DROP05_TERRAINS_CFG,
            "gx_pilot": GX_PILOT_TERRAINS_CFG,
            "cliff_brake": CLIFF_DETECTION_TERRAINS_CFG,
            "flat_mesh": FLAT_MESH_TERRAINS_CFG,
            "robotlab_train": ROBOTLAB_TRAIN_STAIRS_SLOPE_CFG,
        }[args.terrain_profile]
        env_cfg.scene.terrain_type = "generator"
        env_cfg.scene.terrain_generator = copy.deepcopy(selected_terrain)
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.enable_random_terrain_spawn = True
        if args.terrain_seed is not None:
            env_cfg.scene.terrain_generator.seed = int(args.terrain_seed)
            env_cfg.scene.terrain_generator.use_cache = True
            print(f"[INFO] 地形 seed 固定为 {args.terrain_seed}，已启用 mesh cache "
                  f"(dir={env_cfg.scene.terrain_generator.cache_dir})")

        # spawn 隔离检查。randomize_terrain_spawn 只在存在空闲格子时才保证独占；
        # num_envs >= 可 spawn 格数时必然有两台机器人同格，而同一 8m 平台上的
        # 两台 Go2 会进入彼此的 3m 深度相机——margin head 本该从图像里读地形，
        # 却读到了另一台机器人。这是静默的观测污染，必须拦在采集之前。
        _tg = env_cfg.scene.terrain_generator
        _layout = getattr(_tg, "grid_layout", None)
        _keys = getattr(_tg, "spawn_tile_keys", None)
        if _layout is not None and _keys is not None:
            _n_spawn = sum(1 for k in _layout if k in _keys)
        else:
            _n_spawn = _tg.num_rows * _tg.num_cols
        _n_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
        if _n_envs >= _n_spawn:
            raise ValueError(
                f"--num_envs {_n_envs} >= 可 spawn 地形格数 {_n_spawn}"
                f"（profile={args.terrain_profile}）。每次 reset 至少有一台机器人被迫"
                f"与另一台同格，两者会互相出现在深度图里。请用 --num_envs <= {_n_spawn - 1}，"
                f"或换用格数更多的 terrain profile。"
            )
        if _n_envs > _n_spawn - 3:
            print(f"[WARN] num_envs={_n_envs} 接近可 spawn 格数 {_n_spawn}，"
                  f"空闲格子只剩 {_n_spawn - _n_envs + 1} 个，地形随机化会退化成固定轮转。")
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.terminate_base_only:
        env_cfg.robot.terminate_contacts_body_names = [r".*base.*"]
        print("[INFO] 本次运行只用 base 接触终止（头部接触不终止）")
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False
    if args.command_vx_range is not None:
        env_cfg.commands.ranges.lin_vel_x = tuple(sorted(float(x) for x in args.command_vx_range))
    if args.command_vy_range is not None:
        env_cfg.commands.ranges.lin_vel_y = tuple(sorted(float(x) for x in args.command_vy_range))
    if args.command_yaw_rate_range is not None:
        env_cfg.commands.ranges.ang_vel_z = tuple(sorted(float(x) for x in args.command_yaw_rate_range))
    if args.actuator == "go2hv":
        from legged_lab.assets.unitree.unitree_actuator import UnitreeActuatorCfg_Go2HV
        # 与 go2_rl_robotlab GO2_CFG_UNITREE（v1 训练）一致
        env_cfg.scene.robot.actuators = {
            "legs": UnitreeActuatorCfg_Go2HV(
                joint_names_expr=[".*"], stiffness=25.0, damping=0.5, friction=0.01, min_delay=0, max_delay=4,
            )
        }
        print("[INFO] 执行器：GO2HV（RobotLab v1 训练配置）")
    if args.base_mass_range is not None:
        env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = tuple(
            sorted(float(x) for x in args.base_mass_range))
        print(f"[INFO] add_base_mass 覆盖为 {env_cfg.domain_rand.events.add_base_mass.params['mass_distribution_params']} kg")

    if args.camera_mount == "d435":
        from isaaclab.sensors import CameraCfg as _CameraCfg
        env_cfg.scene.camera.use_physical_asset = True
        # 位姿由 D435 资产的 init_state 给出（legged_lab/sensors/camera/camera_asset_cfg.py），
        # sensor 挂在 base/d435/front_cam 上，自身 offset 必须是单位变换，否则会叠两次。
        env_cfg.scene.camera.offset = _CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros")
        print("[INFO] 深度相机：实体 D435 挂载（base/d435/front_cam）")
    if not getattr(args, 'enable_cameras', False):
        env_cfg.scene.camera.enable_camera = False
        env_cfg.scene.camera.use_physical_asset = False
        print("[INFO] 相机已禁用（未传 --enable_cameras）")

    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt

    # 创建环境
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    num_envs = env.num_envs
    print(f"\n[INFO] 环境创建完成: {num_envs} envs, {env.num_actions} actions")

    # ── 加载策略 ──────────────────────────────────────────────────────────────
    if args.robotlab_policy is not None:
        # RobotLab MoE-CTS student：只用 student（450 维本体历史），不需要 teacher/priv。
        # env 保持 LeggedLab 原生布局（按帧展开、按类型排关节），置换与重排都在 adapter 内完成，
        # 因此 apply_command_override_to_obs 的按帧 command patch 不受影响。
        from legged_lab.utils.robotlab_policy import RobotLabStudentPolicy, check_env_compat
        compat_problems = check_env_compat(env)
        if compat_problems:
            raise ValueError("env 与 RobotLab student 的输入/动作约定不一致（应使用 "
                             "--task go2_data_collection_robotlab）:\n  " + "\n  ".join(compat_problems))
        policy = RobotLabStudentPolicy(
            os.path.abspath(args.robotlab_policy),
            env_joint_names=env.robot.data.joint_names,
            history_len=env.cfg.robot.actor_obs_history_length,
            device=env.device,
        )
        print(f"[INFO] 加载 RobotLab student: {args.robotlab_policy}")
        print(f"[INFO]   env joint order: {policy.env_joint_names}")
        print(f"[INFO]   robotlab_from_env: {policy.robotlab_from_env.tolist()}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] 加载策略: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
    print("[INFO] 策略加载成功")

    # 打印每个 env 的 base 质量（add_base_mass 为 startup 采样，整个运行期间固定），
    # 用于把行为差异（如基座高度）对应到采样的负载上。
    _base_ids = [i for i, n in enumerate(env.robot.body_names) if "base" in n]
    _masses = env.robot.root_physx_view.get_masses()
    print(f"[INFO] base 质量 (kg, body={[env.robot.body_names[i] for i in _base_ids]}): "
          f"{[round(float(m), 2) for m in _masses[:, _base_ids].sum(dim=1).tolist()]}")

    if args.store_geo:
        import hashlib
        import omni.usd
        from pxr import UsdGeom
        _g_names = list(env.contact_sensor.body_names)
        _g_base = [i for i, n in enumerate(_g_names) if n == "base"]
        _g_hu, _g_hl = _g_names.index("Head_upper"), _g_names.index("Head_lower")
        _g_feet_s = [i for i, n in enumerate(_g_names) if n.endswith("_foot")]
        _g_feet_names = [_g_names[i] for i in _g_feet_s]
        _g_feet_r = [list(env.robot.body_names).index(n) for n in _g_feet_names]
        _g_base_mass = _masses[:, _base_ids].sum(dim=1).cpu().numpy()
        _g_snap = {}
        _orig_check_reset_geo = env.check_reset

        def _check_reset_geo():
            # env.step 在 check_reset 之后才 reset（reset 会清空接触历史），所以在这里快照当步受力。
            reset_buf, time_out_buf = _orig_check_reset_geo()
            Fh = env.contact_sensor.data.net_forces_w_history          # [N, T, B, 3]，T=0 最新
            Fn = Fh.norm(dim=-1).amax(dim=1)                            # [N, B]，最近 T 个物理步的最大值
            _g_snap["base_force"] = Fn[:, _g_base].amax(dim=1).cpu().numpy()
            _g_snap["head_upper_force"] = Fn[:, _g_hu].cpu().numpy()
            _g_snap["head_lower_force"] = Fn[:, _g_hl].cpu().numpy()
            _g_snap["foot_force_w"] = Fh[:, 0][:, _g_feet_s].cpu().numpy()
            return reset_buf, time_out_buf

        env.check_reset = _check_reset_geo

        # 地形 mesh：直接从仿真中的 USD 读，保证与实际碰撞几何一致（不依赖 seed 重建，docs/82 §6.4）
        _out_dir = Path(args.output_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        _stage = omni.usd.get_context().get_stage()
        _root = env.scene.terrain.cfg.prim_path
        _xf = UsdGeom.XformCache()
        _V, _F, _off = [], [], 0
        for _prim in _stage.Traverse():
            if not _prim.GetPath().pathString.startswith(_root) or not _prim.IsA(UsdGeom.Mesh):
                continue
            _m = UsdGeom.Mesh(_prim)
            _pts = np.asarray(_m.GetPointsAttr().Get(), dtype=np.float64)
            _M = np.asarray(_xf.GetLocalToWorldTransform(_prim), dtype=np.float64)   # USD 行向量约定
            _pts_w = (np.c_[_pts, np.ones(len(_pts))] @ _M)[:, :3]
            _idx = np.asarray(_m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            _cnt = np.asarray(_m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            _tris, _p = [], 0
            for _c in _cnt:                                                         # 扇形三角化
                _tris += [[_idx[_p], _idx[_p + k], _idx[_p + k + 1]] for k in range(1, _c - 1)]
                _p += _c
            _V.append(_pts_w)
            _F.append(np.asarray(_tris, dtype=np.int64) + _off)
            _off += len(_pts_w)
        _V = np.concatenate(_V).astype(np.float32)
        _F = np.concatenate(_F).astype(np.int32)
        _t_origins = getattr(env.scene.terrain, "terrain_origins", None)
        np.savez_compressed(
            _out_dir / "terrain_mesh.npz", vertices=_V, faces=_F,
            env_origins=env.scene.env_origins.cpu().numpy(),
            terrain_origins=(_t_origins.cpu().numpy() if _t_origins is not None else np.zeros(0)),
        )
        print(f"[GEO] 地形 mesh 导出：{len(_V)} 顶点、{len(_F)} 三角形，"
              f"范围 x[{_V[:, 0].min():.1f},{_V[:, 0].max():.1f}] y[{_V[:, 1].min():.1f},{_V[:, 1].max():.1f}] "
              f"z[{_V[:, 2].min():.2f},{_V[:, 2].max():.2f}] → {_out_dir / 'terrain_mesh.npz'}")

        # 运行配置：增量合并前核对"同一物理/同一策略/同一终止语义"（docs/82 数据集口径）
        _pol_path = args.robotlab_policy or locals().get("resume_path")
        _pol_sha = hashlib.sha256(open(_pol_path, "rb").read()).hexdigest() if _pol_path and os.path.isfile(_pol_path) else None
        _run_cfg = {
            "collector_args": vars(args),
            "policy_path": _pol_path, "policy_sha256": _pol_sha,
            "actuator": args.actuator,
            "actuator_cfg": repr(env_cfg.scene.robot.actuators),
            "base_mass_range": env_cfg.domain_rand.events.add_base_mass.params.get("mass_distribution_params"),
            "base_mass_per_env": [round(float(m), 4) for m in _g_base_mass],
            "friction_combine_mode": env_cfg.scene.friction_combine_mode,
            "ground_static_friction": env_cfg.scene.static_friction,
            "ground_dynamic_friction": env_cfg.scene.dynamic_friction,
            "robot_material_event": repr(getattr(env_cfg.domain_rand.events, "physics_material", None)),
            "terminate_contacts_body_names": list(env_cfg.robot.terminate_contacts_body_names),
            "command_ranges": repr(env_cfg.commands.ranges),
            "terrain_generator": repr(env_cfg.scene.terrain_generator),
            "camera": {"mount": args.camera_mount,
                       "use_physical_asset": bool(getattr(env_cfg.scene.camera, "use_physical_asset", False)),
                       "offset": repr(env_cfg.scene.camera.offset)},
            "step_dt": step_dt,
            "contact_sensor_bodies": _g_names,
            "feet": _g_feet_names,
            "geo_fields": {
                "pre_step": ["root_pos_w(3)", "root_quat_w(4,wxyz)", "foot_pos_w(4,3)", "foot_contact(4)", "env_origin(3)"],
                "post_step_snapshot": ["base_force", "head_upper_force", "head_lower_force", "foot_force_w(4,3)"],
                "note": "受力为 check_reset 时刻（reset 之前）最近接触历史的最大模长；foot_force_w 为最新物理步矢量",
            },
        }
        with open(_out_dir / "run_config.json", "w") as _fh:
            json.dump(_run_cfg, _fh, indent=2, default=str, ensure_ascii=False)
        print(f"[GEO] 运行配置 → {_out_dir / 'run_config.json'}")

    if args.log_head_contact:
        _hl_names = list(env.contact_sensor.body_names)
        _hl_head = [i for i, n in enumerate(_hl_names) if n.startswith("Head")]
        _hl_up = [_hl_names[i] for i in _hl_head].index("Head_upper")
        _hl_lo = [_hl_names[i] for i in _hl_head].index("Head_lower")
        _hl_path = Path(args.output_dir) / "head_contact_log.npz"
        _hl_path.parent.mkdir(parents=True, exist_ok=True)
        _hl = {k: [] for k in ("head_force", "head_upper", "head_lower", "base_z", "x", "y", "ep_step", "done")}
        print(f"[INFO] 头部接触全程记录：{[_hl_names[i] for i in _hl_head]} → {_hl_path}")

        def _hl_record(dones_t):
            Fh = env.contact_sensor.data.net_forces_w_history[:, :, _hl_head].norm(dim=-1).amax(dim=1)  # [N, n_head]
            rel = env.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
            _hl["head_force"].append(Fh.amax(dim=1).cpu().numpy())
            _hl["head_upper"].append(Fh[:, _hl_up].cpu().numpy())
            _hl["head_lower"].append(Fh[:, _hl_lo].cpu().numpy())
            _hl["base_z"].append(env.robot.data.root_pos_w[:, 2].cpu().numpy())
            _hl["x"].append(rel[:, 0].cpu().numpy())
            _hl["y"].append(rel[:, 1].cpu().numpy())
            _hl["ep_step"].append(env.episode_length_buf.cpu().numpy())
            _hl["done"].append(dones_t.cpu().numpy())
            if len(_hl["done"]) % 250 == 0:
                np.savez_compressed(_hl_path, names=np.array([_hl_names[i] for i in _hl_head]),
                                    **{k: np.stack(v) for k, v in _hl.items()})

    if args.diag_termination:
        # 包一层 check_reset：在 env 内部 reset 清空接触历史之前，记录每次接触类终止的细节。
        # 自碰撞判据（牛顿第三定律）：头被自己的腿碰到时，某条腿 body 上有一个与头部受力
        # 大小相近、方向相反的力；撞地形时找不到这样的配对。
        from isaaclab.utils.math import quat_apply_inverse
        _cs_names = list(env.contact_sensor.body_names)
        _term_ids = list(env.termination_contact_cfg.body_ids)
        _head_ids = [i for i, n in enumerate(_cs_names) if n.startswith("Head")]
        _leg_ids = [i for i, n in enumerate(_cs_names) if any(k in n for k in ("hip", "thigh", "calf", "foot"))]
        _diag_path = Path(args.output_dir) / "termination_diag.jsonl"
        _diag_path.parent.mkdir(parents=True, exist_ok=True)
        _orig_check_reset = env.check_reset
        print(f"[DIAG] 终止诊断开启：终止 body={[_cs_names[i] for i in _term_ids]}，写入 {_diag_path}")

        def _check_reset_with_diag():
            reset_buf, time_out_buf = _orig_check_reset()
            failed = (reset_buf & ~time_out_buf).nonzero().flatten().tolist()
            if failed:
                F = env.contact_sensor.data.net_forces_w_history  # [N, T, B, 3]，T=0 为最新
                Fn = F.norm(dim=-1)
                with open(_diag_path, "a") as fh:
                    for e in failed:
                        trig = [i for i in _term_ids if Fn[e, :, i].max() > 1.0]
                        # 取触发 body 受力最大的那一帧，所有 body 在同一帧比较
                        i0 = max(trig, key=lambda i: Fn[e, :, i].max()) if trig else _term_ids[0]
                        t = int(Fn[e, :, i0].argmax())
                        quat = env.robot.data.root_quat_w[e:e + 1]
                        rec = {
                            "env_id": e,
                            "episode_step": int(env.episode_length_buf[e]),
                            "root_pos_w": [round(float(v), 3) for v in env.robot.data.root_pos_w[e]],
                            "triggered": {_cs_names[i]: round(float(Fn[e, :, i].max()), 1) for i in trig},
                        }
                        for h in _head_ids:
                            fh_w = F[e, t, h]
                            fmag = float(fh_w.norm())
                            entry = {"force_N": round(fmag, 1),
                                     "force_body": [round(float(v), 1) for v in quat_apply_inverse(quat, fh_w[None])[0]]}
                            if fmag > 1.0:
                                best = None
                                for j in _leg_ids:
                                    fl = F[e, t, j]
                                    if fl.norm() < 1e-6:
                                        continue
                                    cos = float(torch.dot(fh_w, -fl) / (fh_w.norm() * fl.norm()))
                                    ratio = float(fl.norm() / fh_w.norm())
                                    if best is None or cos > best[1]:
                                        best = (_cs_names[j], cos, ratio)
                                if best is not None:
                                    entry["best_leg_pair"] = {"body": best[0], "cos_opposite": round(best[1], 3), "force_ratio": round(best[2], 2)}
                            rec[_cs_names[h]] = entry
                        fh.write(json.dumps(rec) + "\n")
            return reset_buf, time_out_buf

        env.check_reset = _check_reset_with_diag

    # ── 机器人引用 & V3 扰动初始化 ────────────────────────────────────────────
    robot: Articulation = env.scene["robot"]
    batch_data = BatchedEnvData()

    # V1: original_materials 在预热后读取（需等待 DR reset 后得到真实 per-env 摩擦值）
    original_materials = None
    if args.fault_type == 'friction':
        print(f"[INFO] V1 摩擦力实验")
        print(f"       OOD static range:  [{args.ood_static_friction_range[0]}, {args.ood_static_friction_range[1]}]")
        if args.ood_dynamic_ratio_range is not None:
            print(f"       OOD dynamic/static ratio: {args.ood_dynamic_ratio_range}")
        else:
            print(f"       OOD dynamic range: [{args.ood_dynamic_friction_range[0]}, {args.ood_dynamic_friction_range[1]}]")

    # V2: 预分配持续力缓冲区（每步更新后调用 apply_forces_and_torques_at_position）
    persistent_forces: torch.Tensor = None
    persistent_torques: torch.Tensor = None
    persistent_positions: torch.Tensor = None
    all_env_ids_tensor: torch.Tensor = None
    if args.fault_type == 'force_x':
        n_bodies = robot.num_bodies
        persistent_forces = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        persistent_torques = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        persistent_positions = torch.zeros((num_envs, n_bodies, 3), dtype=torch.float32, device=env.device)
        all_env_ids_tensor = torch.arange(num_envs, dtype=torch.int32, device=env.device)
        print(f"[INFO] V2 持续阻力实验")
        print(f"       force_x range=[{args.force_x_magnitude_min}, {args.force_x_magnitude_max}] N, num_bodies={n_bodies}")

    # V3: per-env 扰动状态
    env_t_fault = [0] * num_envs              # 本 episode 的实际/计划扰动触发步数
    env_is_safe = [True] * num_envs           # 本 episode 是否为安全轨迹
    env_fault_applied = [False] * num_envs    # V1/V2: 是否已触发扰动
    env_force_active = [False] * num_envs     # V2: 是否正在施加持续力
    env_sampled_static_mu = [0.0] * num_envs  # V1: 本 episode 采样的 OOD 静摩擦系数
    env_sampled_dynamic_mu = [0.0] * num_envs # V1: 本 episode 采样的 OOD 动摩擦系数
    env_sampled_force_x = [0.0] * num_envs    # V2: 本 episode 采样的 force_x 大小（负值）
    env_force_steps = [0] * num_envs           # V2: fault 触发后已经过的步数（用于斜坡计算）
    env_brake_start_step = [0] * num_envs
    env_brake_original_cmd = [np.zeros(3, dtype=np.float32) for _ in range(num_envs)]
    env_brake_target_vx = [0.0] * num_envs
    env_brake_is_complete = [True] * num_envs
    env_brake_margin_ratio = [1.0] * num_envs
    env_spawn_d0 = [float("nan")] * num_envs
    env_spawn_theta = [float("nan")] * num_envs
    env_spawn_y_cross = [float("nan")] * num_envs
    env_spawn_extra = [0.0] * num_envs
    env_fault_window = [float(args.fault_window_s)] * num_envs
    env_stop_streak = [0] * num_envs
    env_stop_step = [-1] * num_envs
    env_max_steps = [int(args.max_steps)] * num_envs
    spawn_retry_counts: List[int] = []
    env_terrain_type = [-1] * num_envs
    env_terrain_level = [-1] * num_envs
    env_terrain_key = ["unknown"] * num_envs
    env_tracking_streak = [0] * num_envs
    env_tracking_settled_step = [-1] * num_envs

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    def sample_braking_base_command() -> np.ndarray:
        """Sample a fresh non-braked command from the task command ranges."""
        ranges = env.cfg.commands.ranges
        vx = random.uniform(float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1]))
        vy = random.uniform(float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1]))
        yaw = random.uniform(float(ranges.ang_vel_z[0]), float(ranges.ang_vel_z[1]))
        return np.array([vx, vy, yaw], dtype=np.float32)

    def resample_episode_params(env_idx: int):
        """episode 开始时为该 env 重新采样扰动参数"""
        if args.fault_type == 'none':
            env_is_safe[env_idx] = True
        else:
            env_is_safe[env_idx] = (random.random() < args.safe_ratio)
        env_t_fault[env_idx] = (-1 if args.fault_trigger == "window" else
                                random.randint(args.t_fault_min, args.t_fault_max))
        env_fault_applied[env_idx] = False
        env_force_active[env_idx] = False
        env_force_steps[env_idx] = 0
        env_tracking_streak[env_idx] = 0
        env_tracking_settled_step[env_idx] = -1
        env_stop_streak[env_idx] = 0
        env_stop_step[env_idx] = -1
        # r 由 configure_braking_episode 的可行性重采循环采样：它必须和 (v, theta,
        # extra) 一起被接受或一起被拒，在这里单独采会让被拒的 r 泄漏进统计。
        # V1: 每 episode 独立采样 OOD 摩擦系数，增加扰动多样性
        if args.fault_type == 'friction':
            env_sampled_static_mu[env_idx] = random.uniform(
                args.ood_static_friction_range[0], args.ood_static_friction_range[1]
            )
            if args.ood_dynamic_ratio_range is not None:
                dyn_lo, dyn_hi = sorted(float(x) for x in args.ood_dynamic_ratio_range)
                env_sampled_dynamic_mu[env_idx] = (
                    env_sampled_static_mu[env_idx] * random.uniform(dyn_lo, dyn_hi)
                )
            else:
                env_sampled_dynamic_mu[env_idx] = random.uniform(
                    args.ood_dynamic_friction_range[0], args.ood_dynamic_friction_range[1]
                )
        # V2: 每 episode 独立采样 force_x 大小，增加扰动多样性
        if args.fault_type == 'force_x':
            env_sampled_force_x[env_idx] = random.uniform(
                args.force_x_magnitude_min, args.force_x_magnitude_max
            )

    def configure_braking_episode(env_idx: int):
        """reset 后采样本 episode 原始 command，并采样制动开始时间/目标速度。"""
        if getattr(env.scene, 'terrain', None) is not None \
                and hasattr(env.scene.terrain, 'terrain_types'):
            env_terrain_type[env_idx] = int(env.scene.terrain.terrain_types[env_idx].item())
            env_terrain_level[env_idx] = int(env.scene.terrain.terrain_levels[env_idx].item())
            terrain_cfg = env.cfg.scene.terrain_generator
            layout = getattr(terrain_cfg, "grid_layout", None)
            if layout is None:
                # cfg 对象在 scene 构建链路里可能被复制，回写没落到这一份上；
                # 退到生成器模块记录的那一份（采集是单进程单地形，可靠）。
                from legged_lab.terrains import grid_terrain_generator as _gtg
                layout = _gtg.LAST_REALIZED_LAYOUT
            if layout is not None:
                flat_idx = env_terrain_level[env_idx] * terrain_cfg.num_cols + env_terrain_type[env_idx]
                env_terrain_key[env_idx] = str(layout[flat_idx])
            else:
                env_terrain_key[env_idx] = "unknown"
        else:
            env_terrain_type[env_idx] = -1
            env_terrain_level[env_idx] = -1
            env_terrain_key[env_idx] = "unknown"
        lo, hi = args.brake_start_range
        lo = max(0, min(lo, hi))
        hi = min(max(lo, hi), max(args.max_steps - 1, 0))
        # Stage-2 组合模式：非 safe 的 fault episode，制动锚定在 t_fault 之后 delay 步，
        # 保证"先正常走 → t_fault 掉低摩擦 → 走 delay 步(检测窗口) → t_brake 制动"。
        # safe(control) episode 或纯 braking 用原有 brake_start_range。
        if args.brake_trigger == "edge":
            env_brake_start_step[env_idx] = -1
        elif args.fault_type == 'friction' and not env_is_safe[env_idx]:
            d_lo, d_hi = args.post_fault_brake_delay_range
            d_lo = max(1, int(min(d_lo, d_hi)))
            d_hi = max(d_lo, int(d_hi))
            delay = random.randint(d_lo, d_hi)
            brake_start = env_t_fault[env_idx] + delay
            env_brake_start_step[env_idx] = int(min(brake_start, max(args.max_steps - 1, 0)))
        else:
            env_brake_start_step[env_idx] = random.randint(lo, hi)
        # Do not read the current command here: after a braking reset it may still
        # be the previous low target. Sample a fresh base command so ratios do not
        # compound across episodes.
        # (v, r, theta, extra) 联合决定 spawn 是否放得下。任一组合不可行时重采，
        # 而不是中断整个 job —— 高 v 配大 r 必然超出平台，这是设计的可行域边界，
        # 不是错误。重采率本身就是"参数区间超出可行域"的信号，采完打印。
        solved = None
        attempts = 0
        for attempts in range(1, max(1, int(args.spawn_max_retries)) + 1):
            cand_cmd = sample_braking_base_command()
            ratio_lo, ratio_hi = sorted(float(x) for x in args.brake_margin_ratio_range)
            cand_ratio = random.uniform(ratio_lo, ratio_hi)
            if args.spawn_mode != "solve":
                solved = (cand_cmd, cand_ratio, None)
                env_fault_window[env_idx] = float(args.fault_window_s)
                break
            ex_lo, ex_hi = sorted(max(0.0, float(x)) for x in args.spawn_extra_runway)
            extra = random.uniform(ex_lo, ex_hi)
            if args.fault_window_range is not None:
                w_lo, w_hi = sorted(max(0.0, float(x)) for x in args.fault_window_range)
                cand_window = random.uniform(w_lo, w_hi)
            else:
                cand_window = float(args.fault_window_s)
            v = max(float(cand_cmd[0]), 0.0)
            # mu is sampled earlier in this same function, so the runway is solved
            # against the friction this episode will actually brake on.
            d0 = solved_runway(v, cand_ratio, cand_window,
                               env_sampled_dynamic_mu[env_idx]) + extra
            d0 += leading_contact_reach(env_terrain_key[env_idx])
            heading_lo, heading_hi = sorted(float(x) for x in args.spawn_heading_range)
            theta = random.uniform(heading_lo, heading_hi)
            half = float(args.platform_half_width)
            lateral = max(0.0, float(args.spawn_lateral_margin))
            dx, dy = d0 * math.cos(theta), d0 * math.sin(theta)
            x0 = half - dx
            band_lo, band_hi = -half + lateral, half - lateral
            cross_lo = max(band_lo, band_lo + dy)
            cross_hi = min(band_hi, band_hi + dy)
            # ``lateral`` protects the side edges and the far-side spawn limit.
            # Near the target +x edge, low-v/low-r episodes legitimately start
            # closer than this margin (D0 itself is the controlled clearance).
            if x0 < band_lo or x0 >= half or cross_lo > cross_hi:
                continue
            y_cross = random.uniform(cross_lo, cross_hi)
            solved = (cand_cmd, cand_ratio,
                      (d0, theta, x0, y_cross - dy, y_cross, extra, cand_window))
            break
        if solved is None:
            raise ValueError(
                f"solve spawn 连续 {args.spawn_max_retries} 次不可行: env={env_idx}。"
                f"当前 v∈{tuple(env.cfg.commands.ranges.lin_vel_x)}, "
                f"r∈{tuple(args.brake_margin_ratio_range)}, extra∈{tuple(args.spawn_extra_runway)}。"
                "可行域上界 r_max(v)=(2*half-lateral - v^2/(2a) - v*(settle+window))/d_hat_ref(v)；"
                "请收窄 r 上沿或降低速度上限。"
            )
        spawn_retry_counts.append(attempts - 1)
        original_cmd, cand_ratio, geom = solved
        env_brake_margin_ratio[env_idx] = cand_ratio
        env.command_generator.command[env_idx] = torch.as_tensor(
            original_cmd, dtype=torch.float32, device=env.device
        )
        env_brake_original_cmd[env_idx] = original_cmd

        is_complete = random.random() < args.brake_complete_ratio
        if is_complete:
            target_vx = 0.0
        else:
            v0 = max(float(original_cmd[0]), 0.0)
            ratio_lo, ratio_hi = args.brake_target_ratio_range
            ratio_lo = max(0.0, min(float(ratio_lo), 1.0))
            ratio_hi = max(ratio_lo, min(float(ratio_hi), 1.0))
            ratio = random.uniform(ratio_lo, ratio_hi)
            target_vx = v0 * ratio
        env_brake_target_vx[env_idx] = target_vx
        env_brake_is_complete[env_idx] = is_complete

        if geom is not None:
            d0, theta, x0, y0, y_cross, extra, window = geom
            env_fault_window[env_idx] = float(window)
            origin = env.scene.env_origins[env_idx]
            pose = robot.data.root_pose_w[env_idx:env_idx + 1].clone()
            pose[0, 0] = origin[0] + x0
            pose[0, 1] = origin[1] + y0
            pose[0, 3] = math.cos(theta * 0.5)
            pose[0, 4] = 0.0
            pose[0, 5] = 0.0
            pose[0, 6] = math.sin(theta * 0.5)
            env_id_t = torch.tensor([env_idx], dtype=torch.long, device=env.device)
            robot.write_root_pose_to_sim(pose, env_ids=env_id_t)
            robot.write_root_velocity_to_sim(
                torch.zeros((1, 6), dtype=pose.dtype, device=env.device), env_ids=env_id_t
            )
            env_spawn_d0[env_idx] = d0
            env_spawn_theta[env_idx] = theta
            env_spawn_y_cross[env_idx] = y_cross
            env_spawn_extra[env_idx] = extra

        # ── 逐 episode 步数预算 ────────────────────────────────────────────────
        # 固定 max_steps 会在最需要减速尾段的格子（低 mu / 高速 / 长跑道）制造删失，
        # 而删失是信息性的：刹得越远越容易被截断。按实际跑道长度给预算。
        if args.dynamic_max_steps and geom is not None:
            v = max(float(original_cmd[0]), 1e-3)
            d0 = geom[0]
            t_ramp = v / max(float(args.spawn_accel), 1e-6)
            d_ramp = 0.5 * v * t_ramp
            t_travel = t_ramp + max(0.0, d0 - d_ramp) / v
            budget = t_travel + float(args.post_brake_budget_s)
            env_max_steps[env_idx] = int(min(int(args.max_steps),
                                             math.ceil(budget / step_dt) + 5))
        else:
            env_max_steps[env_idx] = int(args.max_steps)

    def apply_braking_commands_for_current_buffers(obs_dict):
        commands = torch.zeros((num_envs, 3), dtype=torch.float32, device=env.device)
        for env_idx in range(num_envs):
            if episodes_completed[env_idx] >= args.num_episodes:
                commands[env_idx] = env.command_generator.command[env_idx]
                continue
            current_step = traj_buffers[env_idx].step_count
            commands[env_idx] = torch.as_tensor(
                compute_braking_command(
                    current_step,
                    env_brake_start_step[env_idx],
                    env_brake_original_cmd[env_idx],
                    env_brake_target_vx[env_idx],
                ),
                dtype=torch.float32,
                device=env.device,
            )
        return apply_command_override_to_obs(env, obs_dict, commands)

    def current_edge_geometry(env_idx: int):
        """Current local pose and both edge distances for one square platform tile."""
        pos_w = robot.data.root_pos_w[env_idx, :2].detach().cpu().numpy()
        origin = env.scene.env_origins[env_idx, :2].detach().cpu().numpy()
        quat = robot.data.root_quat_w[env_idx].detach().cpu().numpy()
        yaw = yaw_from_quat_wxyz(quat)
        local_xy = pos_w - origin
        d_edge, d_ray = edge_distances(local_xy, yaw, args.platform_half_width)
        return local_xy, yaw, d_edge, d_ray

    def update_fault_and_brake_triggers():
        """Apply frame/window faults and arm edge-triggered braking before policy inference."""
        for env_idx in range(num_envs):
            if episodes_completed[env_idx] >= args.num_episodes:
                continue
            current_step = traj_buffers[env_idx].step_count
            _, _, _, d_ray = current_edge_geometry(env_idx)
            d_ray = d_ray - leading_contact_reach(env_terrain_key[env_idx])   # 机头到墙 / 前脚到边
            speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[env_idx, :2]).item())
            v_cmd = max(float(env_brake_original_cmd[env_idx][0]), 0.0)
            # Trigger eligibility must come from measured tracking, not merely an
            # acceleration-time estimate.  Otherwise a slowly accelerating robot
            # can receive the low-friction fault before it ever reaches v_cmd.
            vx_body = float(robot.data.root_lin_vel_b[env_idx, 0].item())
            tracking_tol = max(float(args.tracking_error_abs),
                               float(args.tracking_error_ratio) * max(v_cmd, 0.0))
            if abs(vx_body - v_cmd) <= tracking_tol:
                env_tracking_streak[env_idx] += 1
            else:
                env_tracking_streak[env_idx] = 0
            hold_steps = max(1, int(math.ceil(float(args.tracking_hold_s) / step_dt)))
            settled = env_tracking_streak[env_idx] >= hold_steps
            if settled and env_tracking_settled_step[env_idx] < 0:
                env_tracking_settled_step[env_idx] = current_step
            # 刹停检测：与 outcome 分类用同一判据，供 --early_stop_hold_steps 使用。
            # 刹停后的站立占全部帧的 29%，是最大的一块可回收浪费。
            if env_brake_start_step[env_idx] >= 0 and current_step >= env_brake_start_step[env_idx]:
                if speed < float(args.stop_speed_thresh):
                    env_stop_streak[env_idx] += 1
                else:
                    env_stop_streak[env_idx] = 0
                if (env_stop_step[env_idx] < 0
                        and env_stop_streak[env_idx] >= max(1, int(args.stop_sustain_steps))):
                    env_stop_step[env_idx] = current_step

            ratio = env_brake_margin_ratio[env_idx]

            should_fault = False
            if not env_is_safe[env_idx] and not env_fault_applied[env_idx]:
                if args.fault_trigger == "frame":
                    should_fault = current_step >= env_t_fault[env_idx]
                elif settled:
                    should_fault = (
                        d_ray <= brake_trigger_distance(
                            speed, ratio, env_sampled_dynamic_mu[env_idx])
                        + speed * env_fault_window[env_idx]
                    )
            if should_fault:
                if args.fault_type == 'friction':
                    apply_friction_fault(
                        robot, env_idx,
                        env_sampled_static_mu[env_idx], env_sampled_dynamic_mu[env_idx]
                    )
                    env_fault_applied[env_idx] = True
                    env_t_fault[env_idx] = current_step
                    env_current_static_friction[env_idx] = env_sampled_static_mu[env_idx]
                    env_current_dynamic_friction[env_idx] = env_sampled_dynamic_mu[env_idx]
                    print(f"   [V1] env{env_idx} step{current_step}: "
                          f"static={env_sampled_static_mu[env_idx]:.3f} "
                          f"dynamic={env_sampled_dynamic_mu[env_idx]:.3f} d_ray={d_ray:.3f}")
                elif args.fault_type == 'force_x':
                    env_force_active[env_idx] = True
                    env_fault_applied[env_idx] = True
                    env_t_fault[env_idx] = current_step
                    env_force_steps[env_idx] = 0

            fault_ready = (
                args.fault_type == 'none' or env_is_safe[env_idx] or env_fault_applied[env_idx]
            )
            if (args.collection_mode == 'braking' and args.brake_trigger == "edge"
                    and env_brake_start_step[env_idx] < 0 and settled and fault_ready
                    and d_ray <= brake_trigger_distance(
                        speed, ratio, env_sampled_dynamic_mu[env_idx])):
                env_brake_start_step[env_idx] = current_step
                _dstar = brake_trigger_distance(
                    speed, ratio, env_sampled_dynamic_mu[env_idx])
                print(f"   [BRAKE] env{env_idx} step{current_step}: v={speed:.3f} "
                      f"r={ratio:.3f} d_ray={d_ray:.3f} d*={_dstar:.3f} "
                      f"mu_d={env_sampled_dynamic_mu[env_idx]:.3f} "
                      f"need={brake_distance_dyn(speed, env_sampled_dynamic_mu[env_idx] or args.brake_ref_fallback_mu_dyn):.3f}")

    for i in range(num_envs):
        resample_episode_params(i)

    # ── 轨迹缓冲区 & 计数器 ───────────────────────────────────────────────────
    traj_buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    episodes_completed = [0] * num_envs
    total_target = num_envs * args.num_episodes

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    if args.collection_mode == 'braking':
        braking_cfg_path = output_path / "braking_config.json"
        with open(braking_cfg_path, 'w') as f:
            json.dump({
                "collection_mode": args.collection_mode,
                "max_steps": args.max_steps,
                "brake_start_range": args.brake_start_range,
                "brake_complete_ratio": args.brake_complete_ratio,
                "brake_target_ratio_range": args.brake_target_ratio_range,
                "command_ranges_from_task": args.task,
                "fault_type": args.fault_type,
                "post_fault_brake_delay_range": args.post_fault_brake_delay_range if args.fault_type == 'friction' else None,
                "ood_static_friction_range": args.ood_static_friction_range if args.fault_type == 'friction' else None,
                "ood_dynamic_friction_range": args.ood_dynamic_friction_range if args.fault_type == 'friction' else None,
                "ood_dynamic_ratio_range": args.ood_dynamic_ratio_range if args.fault_type == 'friction' else None,
                "t_fault_range": [args.t_fault_min, args.t_fault_max] if args.fault_type == 'friction' else None,
                "safe_ratio": args.safe_ratio,
                "terrain_type_names": list(env.cfg.scene.terrain_generator.sub_terrains.keys())
                    if env.cfg.scene.terrain_generator is not None else [],
                "full_args": vars(args),
                "notes": "priv_state now 10D (base_lin_vel_b3 + feet_contact4 + static_mu1 + dynamic_mu1 + applied_force_x1). "
                         "braking writes t_brake; fault_type=friction adds low_friction_brake with t_fault. "
                         "failure uses fall/collision (early-reset) semantics.",
            }, f, indent=2, default=str)
        print(f"[INFO] braking 配置已保存: {braking_cfg_path}")
    shard_idx = 0
    total_collected_saved = 0
    total_brake_safe_saved = 0
    outcome_tally_saved: Dict[str, int] = {}
    total_brake_failed_saved = 0
    pending_demos: List[Dict] = []

    print(f"\n[INFO] 开始数据收集")
    print(f"   collection_mode={args.collection_mode}, fault_type={args.fault_type}, safe_ratio={args.safe_ratio}")
    if args.fault_type != 'none':
        print(f"   t_fault 范围: [{args.t_fault_min}, {args.t_fault_max}]")
    if args.collection_mode == 'braking':
        print(f"   braking start_range={args.brake_start_range}, complete_ratio={args.brake_complete_ratio}, "
              f"target_ratio_range={args.brake_target_ratio_range}")
    print(f"   每环境 episodes: {args.num_episodes}, 总目标: {total_target}")
    print(f"   每 episode 最大步数: {args.max_steps}")

    # ── 初始化 & 预热 ──────────────────────────────────────────────────────────
    all_env_ids = torch.arange(num_envs, device=env.device)
    env.reset(all_env_ids)
    obs_dict = env.get_observations()
    if args.collection_mode == 'braking':
        for env_idx in range(num_envs):
            configure_braking_episode(env_idx)
        obs_dict = apply_braking_commands_for_current_buffers(obs_dict)

    if getattr(args, 'enable_cameras', False):
        print("[INFO] 预热相机渲染器（5帧）...")
        for _ in range(5):
            simulation_app.update()

    WARMUP_STEPS = 50
    print(f"[INFO] 物理预热（{WARMUP_STEPS} 步）...")
    dummy_actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    for _ in range(WARMUP_STEPS):
        obs_dict, _, _, _ = env.step(dummy_actions)
    env.reset(all_env_ids)
    obs_dict = env.get_observations()
    if args.collection_mode == 'braking':
        for env_idx in range(num_envs):
            configure_braking_episode(env_idx)
        obs_dict = apply_braking_commands_for_current_buffers(obs_dict)
    print("[INFO] 预热完成，开始收集")

    # 读取 DR 初始化后的 per-env 摩擦系数（预热 reset 后才得到真实 DR 值）
    # DR event 是 mode="reset"，episode 内固定；V1 fault 触发后会被替换为 OOD 值
    _init_mats = robot.root_physx_view.get_material_properties()
    env_current_static_friction = [float(_init_mats[i, 0, 0]) for i in range(num_envs)]
    env_current_dynamic_friction = [float(_init_mats[i, 0, 1]) for i in range(num_envs)]
    env_current_force_x = [0.0] * num_envs  # V2 fault 触发前为 0
    if args.fault_type == 'friction':
        original_materials = _init_mats.clone() if hasattr(_init_mats, 'clone') else _init_mats.copy()
        print(f"[INFO] 初始静摩擦（各 env，DR 随机化后）: {[f'{v:.3f}' for v in env_current_static_friction]}")

    step_count = 0

    # ── 主循环 ─────────────────────────────────────────────────────────────────
    while simulation_app.is_running():
        with torch.inference_mode():
            if sum(episodes_completed) >= total_target:
                print(f"\n[INFO] 数据收集完成! 共 {total_collected_saved + len(pending_demos)} 条轨迹")
                break

            # ===== Step 1: 在 policy 推理前更新 fault/brake 与 command =====
            update_fault_and_brake_triggers()
            if args.collection_mode == 'braking':
                obs_dict = apply_braking_commands_for_current_buffers(obs_dict)

            # ===== Step 1b: 获取动作 =====
            actions = policy(obs_dict)

            # ===== Step 2: 批量提取观测 & 检查 fault 触发 =====
            batch_data.fetch(env, robot)
            if args.store_geo:
                _g_foot_pos = env.robot.data.body_pos_w[:, _g_feet_r].cpu().numpy()
                _g_origins = env.scene.env_origins.cpu().numpy()

            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue

                buffer = traj_buffers[env_idx]
                current_step = buffer.step_count  # 即将执行的步骤编号（0-indexed）

                # V2: 每步更新斜坡力（fault 触发后逐步增大到目标值）
                if env_force_active[env_idx]:
                    ramp = args.force_x_ramp_steps
                    if ramp <= 0:
                        ratio = 1.0
                    else:
                        ratio = min(env_force_steps[env_idx] / ramp, 1.0)
                    env_current_force_x[env_idx] = env_sampled_force_x[env_idx] * ratio
                    env_force_steps[env_idx] += 1

                # 提取当前帧观测（存入 pending，done 信号在 env.step 后才有）
                # priv_state (R^10)：本帧 static/dynamic friction 与 applied_force_x。
                # fault 在本轮 fetch 前触发，因此触发帧已记录新的摩擦标签。
                current_command = batch_data.commands[env_idx].copy()
                image = extract_image(env, env_idx)
                state = extract_state_vector(batch_data, env_idx)
                # 统一记录 10D priv_state，braking 也记真实 DR/OOD 摩擦，
                # 供 stage-2 低摩擦制动的 friction 门控使用。
                priv_state = extract_priv_state_vector_v3(
                    batch_data, env_idx,
                    env_current_static_friction[env_idx],
                    env_current_dynamic_friction[env_idx],
                    env_current_force_x[env_idx]
                )
                origin_xy = env.scene.env_origins[env_idx, :2].detach().cpu().numpy()
                local_xy = batch_data.root_pos_w[env_idx, :2] - origin_xy
                yaw = yaw_from_quat_wxyz(batch_data.root_quat_w[env_idx])
                d_edge, d_edge_ray = edge_distances(local_xy, yaw, args.platform_half_width)
                base_z = float(batch_data.root_pos_w[env_idx, 2])
                buffer._pending_obs = (
                    image.copy(), state.copy(), priv_state.copy(), d_edge, d_edge_ray, base_z,
                    float(local_xy[0]), float(local_xy[1]), float(yaw)
                )
                buffer._pending_command = current_command
                if args.store_geo:
                    buffer._pending_geo = {
                        "root_pos_w": batch_data.root_pos_w[env_idx],
                        "root_quat_w": batch_data.root_quat_w[env_idx],
                        "foot_pos_w": _g_foot_pos[env_idx],
                        "foot_contact": batch_data.feet_contact[env_idx],
                        "env_origin": _g_origins[env_idx],
                    }

            # ===== Step 2b: V2 持续力（在 env.step 前每步刷新）=====
            if args.fault_type == 'force_x':
                persistent_forces.zero_()
                for env_idx in range(num_envs):
                    if env_force_active[env_idx]:
                        # body 0 = root/base, 施加 -x 方向（opposing forward motion），已含斜坡系数
                        persistent_forces[env_idx, 0, 0] = env_current_force_x[env_idx]
                robot.root_physx_view.apply_forces_and_torques_at_position(
                    persistent_forces,
                    persistent_torques,
                    persistent_positions,
                    all_env_ids_tensor,
                    False               # is_global=True → world frame
                )

            # ===== Step 3: 执行物理步骤 =====
            next_obs_dict, rewards, dones, extras = env.step(actions)
            if args.log_head_contact:
                _hl_record(dones)

            # ===== Step 4: 记录数据 & 处理 episode 结束 =====
            for env_idx in range(num_envs):
                if episodes_completed[env_idx] >= args.num_episodes:
                    continue

                buffer: TrajectoryBuffer = traj_buffers[env_idx]
                (image, state, priv_state, d_edge, d_edge_ray, base_z,
                 root_x, root_y, root_yaw) = buffer._pending_obs
                command = buffer._pending_command
                current_step = buffer.step_count  # 当前步骤编号（add_step 前）

                # done 判断
                script_timeout = buffer.step_count >= env_max_steps[env_idx] - 1
                if (int(args.early_stop_hold_steps) > 0 and env_stop_step[env_idx] >= 0
                        and buffer.step_count >= env_stop_step[env_idx]
                        + int(args.early_stop_hold_steps)):
                    script_timeout = True
                isaac_time_outs = extras.get("time_outs", torch.zeros(num_envs, dtype=torch.bool, device=env.device))
                isaac_timeout = bool(isaac_time_outs[env_idx].item())
                is_timeout = script_timeout or isaac_timeout
                is_env_done = bool(dones[env_idx].item() if isinstance(dones[env_idx], torch.Tensor) else dones[env_idx])
                is_done = is_timeout or is_env_done

                # ── failure 标签 ──────────────────────────────────────────────
                # braking(含 stage-2 低摩擦制动)始终用 fall/collision 语义(提前 reset=failure)，
                # 不用 vel-tracking OOD 判据——低摩擦但没摔的轨迹不应被误标为 failure。
                if args.collection_mode != 'braking' and args.fault_type in ('friction', 'force_x') \
                        and not args.lowfric_walk:
                    # V1/V2: 基于 3D 速度跟随误差
                    failure = compute_vel_failure(
                        batch_data, env_idx,
                        current_step,
                        env_t_fault[env_idx],
                        env_is_safe[env_idx],
                        command,
                        applied_force_x=env_current_force_x[env_idx]
                    )
                else:
                    # none/braking: 提前 reset = failure。制动碰撞 reset 是有意义的失败轨迹。
                    failure = extract_failure(env, env_idx, is_done, is_timeout)

                reward = rewards[env_idx].item()

                buffer.add_step(
                    image=image,
                    state=state,
                    priv_state=priv_state,
                    reward=reward,
                    action=command,
                    done=1 if is_done else 0,
                    failure=failure,
                    d_edge=d_edge,
                    d_edge_ray=d_edge_ray,
                    base_z=base_z,
                    root_x=root_x,
                    root_y=root_y,
                    yaw=root_yaw,
                )
                if args.store_geo:
                    _geo = dict(buffer._pending_geo)
                    for _k in ("base_force", "head_upper_force", "head_lower_force", "foot_force_w"):
                        _geo[_k] = _g_snap[_k][env_idx]
                    buffer.add_geo(_geo)

                if is_done:
                    # ── 组装 demo。写 t_brake / t_fault / fault_type / event_type 供分层采样。 ─────
                    demo = buffer.to_demo(env_id=env_idx)
                    if args.store_geo:
                        demo['base_mass'] = float(_g_base_mass[env_idx])
                    if args.collection_mode == 'braking':
                        # 始终写 t_brake；fault episode 额外写 t_fault，标 low_friction_brake。
                        # fault_type 写 'low_friction_brake'/'brake'(均非 friction/force_x) →
                        # 训练侧 loader 走 fall 语义(n_prefall hindsight + gz onset + failure-based terminal)。
                        demo['t_brake'] = int(env_brake_start_step[env_idx])
                        is_lowfric = (args.fault_type == 'friction' and not env_is_safe[env_idx])
                        demo['t_fault'] = int(env_t_fault[env_idx]) if is_lowfric else -1
                        demo['fault_type'] = 'low_friction_brake' if is_lowfric else 'brake'
                        demo['event_type'] = demo['fault_type']
                        demo['brake_margin_ratio'] = float(env_brake_margin_ratio[env_idx])
                        demo['d_hat_ref_coeffs'] = [
                            float(args.brake_ref_linear), float(args.brake_ref_quadratic),
                            float(args.brake_ref_offset)
                        ]
                        demo['sampled_static_mu'] = float(env_sampled_static_mu[env_idx]) \
                            if args.fault_type == 'friction' else float("nan")
                        demo['sampled_dynamic_mu'] = float(env_sampled_dynamic_mu[env_idx]) \
                            if args.fault_type == 'friction' else float("nan")
                        demo['spawn_d0'] = float(env_spawn_d0[env_idx])
                        demo['spawn_theta'] = float(env_spawn_theta[env_idx])
                        demo['spawn_y_cross'] = float(env_spawn_y_cross[env_idx])
                        demo['t_tracking_settled'] = int(env_tracking_settled_step[env_idx])
                        demo['terrain_type'] = int(env_terrain_type[env_idx])
                        demo['terrain_level'] = int(env_terrain_level[env_idx])
                        demo['terrain_key'] = env_terrain_key[env_idx]
                        demo['hazard_dir'] = int(
                            HAZARD_DIRECTION.get(env_terrain_key[env_idx], 0)
                        )
                        demo['spawn_extra_runway'] = float(env_spawn_extra[env_idx])
                        demo['fault_window_s'] = float(env_fault_window[env_idx])
                        demo['depth_encoding'] = (
                            {'dtype': 'uint8', 'clip_min': float(args.depth_clip_min),
                             'clip_max': float(args.depth_clip_max), 'no_return': 255}
                            if args.depth_uint8 else {'dtype': 'float32'}
                        )
                        demo['brake_original_cmd'] = [
                            float(c) for c in env_brake_original_cmd[env_idx]
                        ]
                        demo['brake_target_vx'] = float(env_brake_target_vx[env_idx])
                        demo['brake_is_complete'] = bool(env_brake_is_complete[env_idx])
                        demo['max_steps_budget'] = int(env_max_steps[env_idx])
                        demo.update(classify_braking_outcome(
                            demo['obs']['priv_state'], demo['d_edge'], demo['failure'],
                            demo['t_brake'], float(args.stop_speed_thresh),
                            max(1, int(args.stop_sustain_steps)),
                        ))
                    elif args.lowfric_walk:
                        # type-5 低摩擦 walk：fall 语义 + low_friction 元数据(非 'friction' → loader 走 fall 分支)。
                        is_lowfric = not env_is_safe[env_idx]
                        demo['t_fault'] = int(env_t_fault[env_idx]) if is_lowfric else -1
                        demo['fault_type'] = 'low_friction' if is_lowfric else 'none'
                        demo['event_type'] = 'low_friction_walk' if is_lowfric else 'normal'
                    elif args.fault_type in ('friction', 'force_x'):
                        demo['t_fault'] = env_t_fault[env_idx] if not env_is_safe[env_idx] else -1
                        demo['fault_type'] = (args.fault_type if not env_is_safe[env_idx] else 'none')

                        # ── V1/V2: 连续帧平滑（episode 结束后对完整序列处理）────
                        if args.min_fail_consecutive > 1:
                            demo['failure'] = smooth_failure_labels(
                                demo['failure'], args.min_fail_consecutive
                            )

                    pending_demos.append(demo)
                    episodes_completed[env_idx] += 1

                    n_fail = sum(demo['failure'])
                    if args.collection_mode == 'braking':
                        status = "FAIL_RESET" if n_fail > 0 else "BRAKE_SAFE"
                    elif args.fault_type == 'none':
                        status = "FAIL" if n_fail > 0 else "SAFE"
                    else:
                        status = "FAIL" if env_fault_applied[env_idx] and n_fail > 0 else \
                                 "SAFE" if env_is_safe[env_idx] else "NO_FAIL"
                    t_f = env_t_fault[env_idx] if not env_is_safe[env_idx] else -1
                    if args.collection_mode == 'braking':
                        is_lowfric = (args.fault_type == 'friction' and not env_is_safe[env_idx])
                        tf_str = f" t_fault={env_t_fault[env_idx]}" if is_lowfric else ""
                        step_str = f"brake_start={env_brake_start_step[env_idx]}{tf_str}"
                    else:
                        step_str = f"t_fault={t_f}"
                    print(f"   env{env_idx} | ep{episodes_completed[env_idx]}/{args.num_episodes} | "
                          f"步数={len(buffer)} {step_str} fail_frames={n_fail} | {status} | "
                          f"总进度={sum(episodes_completed)}/{total_target}")

                    # ── V1: 每条 episode 结束都恢复摩擦 ───────────────────────
                    # BaseEnv.step 会自动 reset 提前终止的 env，但本任务的材质 DR
                    # 是 startup 事件，reset 不会撤销 set_material_properties 写入的
                    # OOD 摩擦。因此 is_env_done 时也必须显式恢复，否则下一条轨迹
                    # 会从上一条的低摩擦出生。
                    if args.fault_type == 'friction' and env_fault_applied[env_idx]:
                        restore_friction(robot, env_idx, original_materials)

                    # ── V2: 停止持续力 ────────────────────────────────────────
                    # （下次循环 persistent_forces_np[:]=0.0 会自动清零该 env）
                    env_force_active[env_idx] = False

                    # 重置缓冲区 & 采样新 episode 参数
                    buffer.reset()
                    resample_episode_params(env_idx)

                    # 分片保存
                    if len(pending_demos) >= args.save_interval:
                        if args.collection_mode == 'braking':
                            n_safe_chunk, n_failed_chunk = count_braking_stats(pending_demos)
                            total_brake_safe_saved += n_safe_chunk
                            total_brake_failed_saved += n_failed_chunk
                            for k, v in count_outcome_stats(pending_demos).items():
                                outcome_tally_saved[k] = outcome_tally_saved.get(k, 0) + v
                        shard_file = shard_dir / f"shard_{shard_idx:04d}.pkl"
                        atomic_save_shard(pending_demos, shard_file)
                        shard_idx += 1
                        total_collected_saved += len(pending_demos)
                        pending_demos.clear()

                    # 脚本 timeout 时主动 reset（Isaac Lab 不会自动 reset）
                    if not is_env_done:
                        env.reset(torch.tensor([env_idx], device=env.device))
                    if args.collection_mode == 'braking':
                        configure_braking_episode(env_idx)

                    # 读取新 episode 经 DR 重新采样的摩擦系数（reset 已发生，DR 已更新）
                    # 同时更新 original_materials[env_idx]，供下次 restore_friction 使用
                    _post_mats = robot.root_physx_view.get_material_properties()
                    env_current_static_friction[env_idx] = float(_post_mats[env_idx, 0, 0])
                    env_current_dynamic_friction[env_idx] = float(_post_mats[env_idx, 0, 1])
                    env_current_force_x[env_idx] = 0.0
                    if args.fault_type == 'friction':
                        original_materials[env_idx] = _post_mats[env_idx]

            obs_dict = next_obs_dict
            if args.collection_mode == 'braking':
                obs_dict = apply_braking_commands_for_current_buffers(obs_dict)
            step_count += 1

            if step_count % 100 == 0:
                print(f"   Step {step_count} | 已收集: {sum(episodes_completed)}/{total_target} episodes")

    # ── 保存剩余分片 ──────────────────────────────────────────────────────────
    if pending_demos:
        if args.collection_mode == 'braking':
            n_safe_chunk, n_failed_chunk = count_braking_stats(pending_demos)
            total_brake_safe_saved += n_safe_chunk
            total_brake_failed_saved += n_failed_chunk
            for k, v in count_outcome_stats(pending_demos).items():
                outcome_tally_saved[k] = outcome_tally_saved.get(k, 0) + v
        shard_file = shard_dir / f"shard_{shard_idx:04d}.pkl"
        atomic_save_shard(pending_demos, shard_file)
        shard_idx += 1
        total_collected_saved += len(pending_demos)
        pending_demos.clear()

    if args.no_final_merge:
        print(f"\n[INFO] --no_final_merge 已设置，跳过合并")
        print(f"   分片目录: {shard_dir}  ({shard_idx} 个分片, {total_collected_saved} eps)")
        if args.collection_mode == 'braking':
            total = total_brake_safe_saved + total_brake_failed_saved
            denom = max(total, 1)
            _report_outcomes(outcome_tally_saved, spawn_retry_counts)
            print(f"   braking 统计: BRAKE_SAFE={total_brake_safe_saved}/{total} "
                  f"({total_brake_safe_saved / denom:.1%}), "
                  f"FAIL_RESET={total_brake_failed_saved}/{total} "
                  f"({total_brake_failed_saved / denom:.1%})")
        return

    # ── 合并所有分片 ──────────────────────────────────────────────────────────
    print(f"\n[INFO] 合并 {shard_idx} 个分片（共 {total_collected_saved} eps）...")
    if args.output_name:
        output_file = output_path / f"{args.output_name}.pkl"
    elif args.collection_mode == 'braking':
        output_file = output_path / f"go2_braking_{total_collected_saved}ep.pkl"
    else:
        output_file = output_path / f"go2_{args.fault_type}_{total_collected_saved}ep.pkl"

    tmp_output = output_file.with_suffix('.tmp')
    all_merged: List[Dict] = []
    with open(tmp_output, 'wb') as fout:
        for i in range(shard_idx):
            shard_file = shard_dir / f"shard_{i:04d}.pkl"
            with open(shard_file, 'rb') as f:
                chunk = pickle.load(f)
            all_merged.extend(chunk)
            print(f"   [{i+1}/{shard_idx}] 已读取 {len(chunk)} eps")
        pickle.dump(all_merged, fout, protocol=4)
    tmp_output.rename(output_file)

    print(f"[INFO] 最终文件: {output_file}")
    print(f"   总 episodes: {len(all_merged)}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 统计 failure/safe 分布。braking 数据不写 fault_type，按最后一帧 failure 判断。
    n_failed = sum(1 for d in all_merged if d.get('failure', [0]) and bool(d['failure'][-1]))
    n_safe = len(all_merged) - n_failed
    print(f"   failure 轨迹: {n_failed}, safe 轨迹: {n_safe}")
    if args.collection_mode == 'braking':
        _report_outcomes(count_outcome_stats(all_merged), spawn_retry_counts)
        print_braking_stats(all_merged)

    import shutil
    shutil.rmtree(shard_dir)
    print(f"[INFO] 已清理分片目录: {shard_dir}")

    # 打印示例
    if all_merged:
        s = all_merged[0]
        print(f"\n[INFO] 数据结构示例（第1条）:")
        print(f"   trajectory_length={len(s['actions'])}, top_keys={sorted(s.keys())}")
        print(f"   state shape:      {s['obs']['state'][0].shape}")
        print(f"   priv_state shape: {s['obs']['priv_state'][0].shape}")
        print(f"   action shape:     {s['actions'][0].shape}")
        print(f"   failure labels:   {s['failure'][:5]} ... {s['failure'][-5:]}")


if __name__ == "__main__":
    collect_data()
    simulation_app.close()
