#!/usr/bin/env python3
"""
play_with_ood_friction_eval.py — OOD 低摩擦实验版 (基于 play_with_critic_filter_eval.py)

在 episode 进行到第 N 步时，将 robot 足部摩擦系数强制切换为 OOD 低值，
模拟 robot 从正常地面走入光滑区域。观察 WM+Q vs ABS 的预警时机差异。

核心 API (IsaacLab):
  materials = robot.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3)
  materials[env_ids, :, 0] = new_static_friction
  materials[env_ids, :, 1] = new_dynamic_friction
  robot.root_physx_view.set_material_properties(materials, env_ids.cpu())

新增 CLI 参数:
  --ood_friction_step     : Episode 第几步触发低摩擦 (None=禁用)
  --ood_static_friction   : OOD 静摩擦系数，默认 0.15 (训练范围 0.6-1.0)
  --ood_dynamic_friction  : OOD 动摩擦系数，默认 0.10 (训练范围 0.4-0.8)

使用方法：见 OOD_FRICTION_EVAL_TODO.md
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path

# ==================== 参数解析 (必须在 IsaacSim import 之前) ====================
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(parent_dir)


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play with Critic Safety Filter + OOD Friction Eval")
parser.add_argument("--task", type=str, default="go2_flat")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=250,
                    help="Script-level per-episode step limit (default 250 = 5s @ 50Hz).")
parser.add_argument("--max_episodes", type=int, default=500)
parser.add_argument("--max_episode_length_s", type=float, default=10.0)

# Critic Filter 参数
parser.add_argument("--safety_threshold", type=float, default=0.0)
parser.add_argument("--monitor_only", action="store_true")
parser.add_argument("--wm_path", type=str, default=None)
parser.add_argument("--critic_path", type=str, default=None)
parser.add_argument("--gx_tau", type=float, default=None)
parser.add_argument("--print_interval", type=int, default=50)
parser.add_argument("--save_stats", type=str, default=None)
parser.add_argument("--log_values", action="store_true")
parser.add_argument("--takeover_duration", type=int, default=50)
parser.add_argument("--reset_grace_k", type=int, default=5)
parser.add_argument("--alarm_k", type=int, default=1)
parser.add_argument("--speed_match_frac", type=float, default=0.0,
                    help="Suppress alarm until robot speed >= frac * |cmd_vx|. "
                         "Prevents false alarms during acceleration. 0.0 = disabled (default).")
parser.add_argument("--cmd_accel_rate", type=float, default=0.0,
                    help="Cmd ramp: max rate (m/s²) at which cmd_vx grows from 0 toward target each episode. "
                         "0.0 = disabled / step command (default).")
parser.add_argument("--cmd_decel_rate", type=float, default=0.0,
                    help="Takeover decel ramp: max rate (m/s²) at which cmd_vx decreases toward 0 "
                         "during takeover. 0.0 = immediate stop (default).")
parser.add_argument("--show_q_rand", action="store_true")
parser.add_argument("--diagnose_q_target", action="store_true",
                    help="Compute explicit WM+V_stop Q target alongside critic Q.")
parser.add_argument("--v_stop_path", type=str, default=None,
                    help="Frozen V_stop_post checkpoint used by --diagnose_q_target.")
parser.add_argument("--diagnostic_repeat", type=int, default=1,
                    help="Command repeat R for explicit Q target diagnostics.")
parser.add_argument("--diagnostic_gamma", type=float, default=0.995)
parser.add_argument("--adaptive_k", type=float, default=None)
parser.add_argument("--adaptive_burnin", type=int, default=200)
parser.add_argument("--eval_terrain", type=str, default=None, choices=["flat", "cliff"])
parser.add_argument("--eval_h", type=int, default=30)
parser.add_argument("--eval_safe_strict_f", type=int, default=None)

# 受控速度实验
parser.add_argument("--fixed_vx", type=float, default=None)
parser.add_argument("--fixed_vy", type=float, default=0.0)
parser.add_argument("--fixed_yaw", type=float, default=0.0)
parser.add_argument("--platform_half_width", type=float, default=5.0)
parser.add_argument("--premature_speed_frac", type=float, default=0.0)

# Push 实验控制
parser.add_argument("--push_vx", type=float, default=None)
parser.add_argument("--no_push", action="store_true")
parser.add_argument("--push_frac_low", type=float, default=0.7)
parser.add_argument("--push_frac_high", type=float, default=0.85)

# Video recording 参数
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_path", type=str, default="./videos/ood_friction_eval.mp4")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_follow_offset", type=float, nargs=3, default=[-1.5, 0.0, 0.8],
                    metavar=("DX", "DY", "DZ"))
parser.add_argument("--video_env_idx", type=int, default=0)
parser.add_argument("--video_no_follow", action="store_true")

# Per-step logging
parser.add_argument("--log_step_data", action="store_true",
                    help="Log per-step per-env state data for trajectory analysis.")
parser.add_argument("--save_step_log", type=str, default=None,
                    help="Path to save per-step log as .npz (requires --log_step_data).")

# RSSM warmup experiment
parser.add_argument("--warmup_steps", type=int, default=0,
                    help="Per-episode RSSM warmup: stand still for this many steps at episode start "
                         "BEFORE beginning the fixed_vx approach. 0 = disabled (default).")

# ── OOD 低摩擦实验 ──────────────────────────────────────────────────────────
parser.add_argument("--ood_friction_step", type=int, default=None,
                    help="Step-mode zone entry: episode step after which OOD zone is active. "
                         "None = disabled (unless --ood_zone_mode=position).")
parser.add_argument("--ood_static_friction", type=float, default=0.15,
                    help="OOD static friction coefficient. Training range 0.6-1.0. Default 0.15.")
parser.add_argument("--ood_dynamic_friction", type=float, default=0.10,
                    help="OOD dynamic friction coefficient. Training range 0.4-0.8. Default 0.10.")
parser.add_argument("--ood_friction_decay", action="store_true",
                    help="Decay mode: gradually reduce friction from normal to OOD target over "
                         "--ood_friction_decay_steps steps after zone entry, instead of instant switch.")
parser.add_argument("--ood_friction_decay_steps", type=int, default=50,
                    help="Steps to linearly decay friction from normal to OOD values. "
                         "Default 50 (~1s @ 50Hz). Only used when --ood_friction_decay is set.")
parser.add_argument("--ood_random_flip", action="store_true",
                    help="Random flip mode: in OOD zone, randomly toggle between normal/OOD "
                         "friction every --ood_friction_interval steps. "
                         "Key metric: Q variance in zone should be lower than V variance (temporal integration).")
parser.add_argument("--ood_friction_interval", type=int, default=10,
                    help="Steps between friction flips in random mode (~1 gait cycle). Default 10.")
parser.add_argument("--ood_friction_prob", type=float, default=0.5,
                    help="Probability of LOW friction at each flip. Default 0.5.")
parser.add_argument("--ood_zone_mode", type=str, default="step",
                    choices=["step", "position"],
                    help="OOD zone detection: 'step' = enter after --ood_friction_step episode steps (default); "
                         "'position' = enter when within --ood_zone_dist meters of cliff edge.")
parser.add_argument("--ood_zone_dist", type=float, default=6.0,
                    help="Distance from cliff edge (m) at which OOD zone starts, position mode. Default 6.0.")
parser.add_argument("--ood_cmd_vx", type=float, default=None,
                    help="Override cmd_vx on OOD zone entry to stress proprio signals. "
                         "Set higher than --fixed_vx to trigger slip on low friction (e.g. 2.5 when "
                         "fixed_vx=0.5). Set 0.0 to test emergency braking on slippery surface. "
                         "Accel-suppress is disabled in zone so filter alarms immediately. "
                         "None = no cmd override (default).")
parser.add_argument("--ood_cmd_vy", type=float, default=0.0,
                    help="Override cmd_vy on OOD zone entry. Default 0.0.")
# ── 场景 F1: HAA 关节刚度渐进衰减 ────────────────────────────────────────────
parser.add_argument("--f1_enable", action="store_true",
                    help="F1 scenario: progressively decay one HAA joint's KP linearly from "
                         "--f1_fault_step to zero over --f1_decay_steps steps. "
                         "Depth unchanged; joint_pos error grows over time → WM h_t detects "
                         "the trend earlier than ABS single-frame V(s).")
parser.add_argument("--f1_fault_step", type=int, default=30,
                    help="Episode step at which KP decay begins. Default 30 "
                         "(gives ~0.6s of normal walking before fault injection @ 50Hz).")
parser.add_argument("--f1_target_joint", type=str, default="FL_hip_joint",
                    help="Joint whose KP degrades. Default 'FL_hip_joint' (FL HAA). "
                         "Other options: FR_hip_joint, RL_hip_joint, RR_hip_joint.")
parser.add_argument("--f1_decay_steps", type=int, default=60,
                    help="Steps for KP to decay from nominal to f1_kp_min. Default 60 "
                         "(~1.2s @ 50Hz; gives a clear pre-failure window).")
parser.add_argument("--f1_kp_min", type=float, default=0.0,
                    help="Minimum KP after full decay. Default 0.0 (complete stiffness loss). "
                         "Set >0 for partial degradation.")
# ────────────────────────────────────────────────────────────────────────────
# ── 方案 A: 执行器弱化 ────────────────────────────────────────────────────────
parser.add_argument("--ood_actuator_weakness", action="store_true",
                    help="Approach A: reduce joint stiffness/damping to ood_kp_scale*normal "
                         "when entering OOD zone. Depth unchanged; joint tracking errors explode.")
parser.add_argument("--ood_kp_scale", type=float, default=0.1,
                    help="Approach A: kp/kd scale factor in OOD zone (0.1 = 10%% of normal). Default 0.1.")
# ── 方案 B: 持续侧向外力 ──────────────────────────────────────────────────────
parser.add_argument("--ood_sustained_force", type=float, nargs=3,
                    default=None, metavar=("FX", "FY", "FZ"),
                    help="Approach B: apply a constant body-frame force [FX FY FZ] (N) "
                         "every step while in OOD zone. Depth unchanged; accumulating tilt "
                         "tests WM temporal integration. E.g.: --ood_sustained_force 0 10 0 "
                         "for 10N lateral. None = disabled (default).")
# ────────────────────────────────────────────────────────────────────────────

import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.diagnose_q_target and not args_cli.v_stop_path:
    parser.error("--diagnose_q_target requires --v_stop_path PATH_TO_V_STOP_POST")
if args_cli.diagnose_q_target and not Path(args_cli.v_stop_path).is_file():
    parser.error(f"--v_stop_path does not exist: {args_cli.v_stop_path}")

# ==================== 启动 IsaacSim ====================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==================== IsaacSim 启动后的 import ====================
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.assets.articulation import Articulation

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


# ── Helpers ──────────────────────────────────────────────────────────────────
def _first_consec_sim(arr, tau, K):
    count = 0
    for i, v in enumerate(arr):
        count = count + 1 if v <= tau else 0
        if count >= K:
            return i - K + 1
    return -1


def _sweep_tau_sim(fall_q, safe_q, H, safe_strict_f, tau_list):
    out = dict(tpr=[], fpr_all=[], fpr_strict=[], fnr=[], f1=[], lead=[])
    for tau in tau_list:
        tp, fn = 0, 0
        for arr in fall_q:
            w = arr[-H:] if len(arr) >= H else arr
            tp += int(np.sum(w <= tau))
            fn += int(np.sum(w > tau))
        fp_all, tn_all, fp_str, tn_str = 0, 0, 0, 0
        for arr in safe_q:
            fp_all += int(np.sum(arr <= tau))
            tn_all += int(np.sum(arr > tau))
            s = arr[:safe_strict_f]
            fp_str += int(np.sum(s <= tau))
            tn_str += int(np.sum(s > tau))
        tpr     = tp      / max(tp + fn, 1)
        fpr_all = fp_all  / max(fp_all + tn_all, 1)
        fpr_str = fp_str  / max(fp_str + tn_str, 1)
        pre     = tp      / max(tp + fp_all, 1)
        f1      = 2 * pre * tpr / max(pre + tpr, 1e-12)
        leads   = [len(a) - int(np.where(a <= tau)[0][0])
                   for a in fall_q if len(np.where(a <= tau)[0])]
        out['tpr'].append(tpr);            out['fpr_all'].append(fpr_all)
        out['fpr_strict'].append(fpr_str); out['fnr'].append(1.0 - tpr)
        out['f1'].append(f1);              out['lead'].append(np.mean(leads) if leads else 0.0)
    return out


def load_critic_filter(env, args):
    print("\n" + "=" * 60)
    print("Loading Critic Safety Filter (delayed import)...")
    print("=" * 60)
    from critic_safety_filter import CriticSafetyFilter
    filt = CriticSafetyFilter(
        wm_path=args.wm_path,
        critic_path=args.critic_path,
        threshold=args.safety_threshold,
        gx_tau=args.gx_tau,
        device="cuda:0",
        env=env,
        adaptive_k=args.adaptive_k,
        adaptive_burnin=args.adaptive_burnin,
        v_stop_path=args.v_stop_path,
        diagnostic_repeat=args.diagnostic_repeat,
        diagnostic_gamma=args.diagnostic_gamma,
    )
    print("=" * 60 + "\n")
    return filt


# ── OOD friction helpers ──────────────────────────────────────────────────────
def _compute_cohens_d(samples: list, key: str = 'q') -> dict:
    """Compute scale-free friction-sensitivity metrics from (val, is_low) sample list.

    Cohen's d = |mean_low - mean_high| / pooled_std
      Low d  → signal doesn't distinguish friction state → temporal integration (WM)
      High d → signal reacts to each friction flip        → frame-reactive (ABS)
    Also computes lag-1 autocorrelation (high = slow temporal change = RSSM smoothing).
    """
    if not samples:
        return {f'ood_zone_{key}_n': 0, f'ood_zone_{key}_cohens_d': None,
                f'ood_zone_{key}_autocorr_lag1': None,
                f'ood_zone_{key}_mean_low': None, f'ood_zone_{key}_mean_high': None}
    vals_low  = np.array([v for v, lo in samples if lo],     dtype=float)
    vals_high = np.array([v for v, lo in samples if not lo], dtype=float)
    all_vals  = np.array([v for v, _  in samples],           dtype=float)
    autocorr = (float(np.corrcoef(all_vals[:-1], all_vals[1:])[0, 1])
                if len(all_vals) > 2 else float('nan'))
    if len(vals_low) > 1 and len(vals_high) > 1:
        sp = np.sqrt((vals_low.std() ** 2 + vals_high.std() ** 2) / 2)
        cohens_d = float(abs(vals_low.mean() - vals_high.mean()) / (sp + 1e-9))
    else:
        cohens_d = float('nan')
    return {
        f'ood_zone_{key}_n':           len(samples),
        f'ood_zone_{key}_n_low':       len(vals_low),
        f'ood_zone_{key}_n_high':      len(vals_high),
        f'ood_zone_{key}_mean_low':    float(vals_low.mean())  if len(vals_low)  else None,
        f'ood_zone_{key}_mean_high':   float(vals_high.mean()) if len(vals_high) else None,
        f'ood_zone_{key}_cohens_d':    cohens_d,   # KEY: low = integrates; high = frame-reactive
        f'ood_zone_{key}_autocorr_lag1': autocorr, # KEY: high = slow change = RSSM smoothing
    }


def apply_ood_friction(robot, env_ids_cpu: torch.Tensor, static_mu: float, dynamic_mu: float):
    mats = robot.root_physx_view.get_material_properties()
    mats[env_ids_cpu, :, 0] = static_mu
    mats[env_ids_cpu, :, 1] = dynamic_mu
    robot.root_physx_view.set_material_properties(mats, env_ids_cpu)


def apply_ood_friction_lerp(robot, env_ids_cpu: torch.Tensor, saved_materials: torch.Tensor,
                             static_mu: float, dynamic_mu: float, alpha_cpu: torch.Tensor):
    """Apply per-env linearly interpolated friction between saved (normal) and OOD target values.
    alpha_cpu: CPU float tensor of shape (num_envs,), values in [0.0, 1.0].
    0.0 = normal (saved) friction; 1.0 = full OOD friction.
    """
    mats = robot.root_physx_view.get_material_properties()
    a = alpha_cpu[env_ids_cpu].view(-1, 1)  # (N, 1) for broadcasting over shapes
    mats[env_ids_cpu, :, 0] = saved_materials[env_ids_cpu, :, 0] * (1.0 - a) + static_mu * a
    mats[env_ids_cpu, :, 1] = saved_materials[env_ids_cpu, :, 1] * (1.0 - a) + dynamic_mu * a
    robot.root_physx_view.set_material_properties(mats, env_ids_cpu)


def restore_friction(robot, env_ids_cpu: torch.Tensor, saved_materials: torch.Tensor):
    robot.root_physx_view.set_material_properties(saved_materials, env_ids_cpu)


def apply_ood_actuator_weakness(robot, env_ids_gpu: torch.Tensor, kp_scale: float,
                                 saved_stiffness: dict, saved_damping: dict):
    """Approach A: scale down actuator stiffness/damping for given envs.
    saved_stiffness/damping: dict[act_name -> (num_envs, num_dofs) tensor]
    """
    for name, act in robot.actuators.items():
        act.stiffness[env_ids_gpu] = saved_stiffness[name][env_ids_gpu] * kp_scale
        act.damping[env_ids_gpu]   = saved_damping[name][env_ids_gpu]   * kp_scale


def restore_actuator_normal(robot, env_ids_gpu: torch.Tensor,
                             saved_stiffness: dict, saved_damping: dict):
    """Approach A: restore actuator stiffness/damping to saved values.
    saved_stiffness/damping: dict[act_name -> (num_envs, num_dofs) tensor]
    """
    for name, act in robot.actuators.items():
        act.stiffness[env_ids_gpu] = saved_stiffness[name][env_ids_gpu]
        act.damping[env_ids_gpu]   = saved_damping[name][env_ids_gpu]


# ── F1 helpers ───────────────────────────────────────────────────────────────
def apply_f1_decay_step(robot, env_ids: torch.Tensor,
                        f1_act_name: str, f1_joint_col: int,
                        f1_kp_nominal: torch.Tensor,
                        kp_min: float,
                        alpha: torch.Tensor):
    """F1: set KP for one joint = nominal*(1-α) + kp_min*α for given envs.

    alpha (num_envs,) in [0,1]: 0 = intact, 1 = fully degraded.
    Only the column f1_joint_col in the actuator stiffness tensor is modified;
    all other joints are untouched.
    """
    act = robot.actuators[f1_act_name]
    kp_target = (f1_kp_nominal[env_ids] * (1.0 - alpha[env_ids])
                 + kp_min * alpha[env_ids])
    act.stiffness[env_ids, f1_joint_col] = kp_target


def restore_f1_stiffness(robot, env_ids: torch.Tensor,
                         f1_act_name: str, f1_joint_col: int,
                         f1_kp_nominal: torch.Tensor):
    """F1: restore nominal KP for the target joint after episode reset."""
    act = robot.actuators[f1_act_name]
    act.stiffness[env_ids, f1_joint_col] = f1_kp_nominal[env_ids]
# ─────────────────────────────────────────────────────────────────────────────


def apply_ood_sustained_force(robot, env, zone_ids_gpu: torch.Tensor,
                               force_body: torch.Tensor):
    """Approach B: apply constant body-frame force to base link of zone envs.

    force_body: (3,) tensor in body frame, broadcast to all zone envs.
    Uses set_external_force_and_torque which accumulates until next env.step().
    """
    from isaaclab.utils.math import quat_apply
    n = zone_ids_gpu.shape[0]
    if n == 0:
        return
    quat_w = robot.data.root_quat_w[zone_ids_gpu]          # (N, 4) wxyz
    force_b = force_body.unsqueeze(0).expand(n, -1)         # (N, 3)
    force_w = quat_apply(quat_w, force_b)                   # body → world

    num_bodies = robot.num_bodies
    forces_all  = torch.zeros(env.num_envs, num_bodies, 3, device=env.device)
    torques_all = torch.zeros(env.num_envs, num_bodies, 3, device=env.device)
    forces_all[zone_ids_gpu, 0, :] = force_w               # body idx 0 = base link
    robot.set_external_force_and_torque(forces_all, torques_all)


# =====================================================================
#               Tier-2 Episode Metrics Tracker
# =====================================================================
class EpisodeMetrics:
    """
    Track per-episode Tier-2 metrics: alarm_dist, alarm_vel, stop_margin, late_fail.
    """

    def __init__(self, num_envs: int, platform_half_width: float, device):
        self.num_envs = num_envs
        self.phw = platform_half_width
        self.device = device

        nan = float('nan')
        self.alarm_pos_xy = torch.full((num_envs, 2), nan, device=device)
        self.alarm_origin_xy = torch.full((num_envs, 2), nan, device=device)
        self.alarm_vel_x = torch.full((num_envs,), nan, device=device)
        self.had_takeover = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alarm_is_premature = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alarm_is_ramp_phase = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pre_step_pos_xy = torch.full((num_envs, 2), nan, device=device)
        self.pre_step_origin_xy = torch.full((num_envs, 2), nan, device=device)

        self.alarm_records: list = []
        self.stop_records: list = []
        self.late_fail_count = 0
        self.late_fail_past_edge_count = 0
        self.late_fail_on_platform_count = 0
        self.missed_fall_count = 0

    def update_pre_step(self, robot, env):
        self.pre_step_pos_xy[:] = robot.data.root_pos_w[:, :2].detach()
        try:
            origins = env.scene.terrain.env_origins
            self.pre_step_origin_xy[:, 0] = origins[:, 0]
            self.pre_step_origin_xy[:, 1] = origins[:, 1]
        except Exception:
            pass

    def try_load_terrain_edge(self, env):
        try:
            origins = env.scene.terrain.env_origins
            print(f"[INFO][Metrics] Circular platform model: radius={self.phw}m  "
                  f"terrain origins x ∈ ({origins[:, 0].min():.1f}, {origins[:, 0].max():.1f}), "
                  f"y ∈ ({origins[:, 1].min():.1f}, {origins[:, 1].max():.1f})")
            print(f"[INFO][Metrics] NOTE: terrain origin is captured per-alarm (not fixed at init).")
        except Exception as e:
            print(f"[WARN][Metrics] Terrain env_origins unavailable ({e}). "
                  f"distance metrics will be NaN.")

    def record_alarm(self, robot, alarm_now: torch.Tensor, env,
                     is_premature: torch.Tensor, is_ramp_phase: torch.Tensor):
        new_alarm = alarm_now & torch.isnan(self.alarm_pos_xy[:, 0])
        if new_alarm.any():
            self.alarm_pos_xy[new_alarm, 0] = robot.data.root_pos_w[new_alarm, 0]
            self.alarm_pos_xy[new_alarm, 1] = robot.data.root_pos_w[new_alarm, 1]
            self.alarm_vel_x[new_alarm] = robot.data.root_lin_vel_b[new_alarm, 0]
            self.alarm_is_premature[new_alarm] = is_premature[new_alarm]
            self.alarm_is_ramp_phase[new_alarm] = is_ramp_phase[new_alarm]
            try:
                origins = env.scene.terrain.env_origins
                self.alarm_origin_xy[new_alarm, 0] = origins[new_alarm, 0]
                self.alarm_origin_xy[new_alarm, 1] = origins[new_alarm, 1]
            except Exception:
                pass

    def mark_takeover(self, danger: torch.Tensor):
        self.had_takeover[danger] = True

    def _compute_distance_to_edge(self, pos_xy: torch.Tensor, center_xy: torch.Tensor) -> float:
        if torch.isnan(center_xy[0]):
            return float('nan')
        dx = abs(pos_xy[0].item() - center_xy[0].item())
        dy = abs(pos_xy[1].item() - center_xy[1].item())
        return self.phw - max(dx, dy)

    def record_stop(self, robot, done_envs: torch.Tensor, env, use_snapshot: bool = False):
        try:
            origins = env.scene.terrain.env_origins
        except Exception:
            origins = None
        for i in done_envs.tolist():
            if origins is not None:
                center_xy = origins[i, :2]
            else:
                center_xy = torch.full((2,), float('nan'), device=self.device)
            stop_xy = self.pre_step_pos_xy[i] if use_snapshot else robot.data.root_pos_w[i, :2]
            stop_dist = self._compute_distance_to_edge(stop_xy, center_xy)

            alarm_xy = self.alarm_pos_xy[i]
            alarm_origin = self.alarm_origin_xy[i]
            alarm_dist = (self._compute_distance_to_edge(alarm_xy, alarm_origin)
                          if not torch.isnan(alarm_xy[0]) else float('nan'))

            is_prem = bool(self.alarm_is_premature[i].item())
            self.stop_records.append({'stop_margin_m': stop_dist, 'alarm_dist_m': alarm_dist,
                                      'is_premature': is_prem})

    def on_episode_done(self, env_ids: torch.Tensor, is_fall_vec: torch.Tensor,
                        robot=None, env=None):
        for _li, _ei in enumerate(env_ids.tolist()):
            is_fall = bool(is_fall_vec[_li].item())
            had_tk = bool(self.had_takeover[_ei].item())

            fall_dist = float('nan')
            if is_fall:
                pos_xy = self.pre_step_pos_xy[_ei]
                origin_xy = self.pre_step_origin_xy[_ei]
                fall_dist = self._compute_distance_to_edge(pos_xy, origin_xy)

            if had_tk and is_fall:
                self.late_fail_count += 1
                if not np.isnan(fall_dist):
                    if fall_dist > 0:
                        self.late_fail_on_platform_count += 1
                    else:
                        self.late_fail_past_edge_count += 1
            elif is_fall and not had_tk:
                if np.isnan(fall_dist) or fall_dist <= 0:
                    self.missed_fall_count += 1

            a_xy = self.alarm_pos_xy[_ei]
            a_origin = self.alarm_origin_xy[_ei]
            if not torch.isnan(a_xy[0]):
                alarm_dist = self._compute_distance_to_edge(a_xy, a_origin)
                alarm_vel = float(self.alarm_vel_x[_ei].item())
                is_prem = bool(self.alarm_is_premature[_ei].item())
                is_ramp = bool(self.alarm_is_ramp_phase[_ei].item())
                self.alarm_records.append({
                    'alarm_dist_m': alarm_dist,
                    'alarm_vel_mps': alarm_vel,
                    'is_fall': is_fall,
                    'had_takeover': had_tk,
                    'is_premature': is_prem,
                    'is_ramp_phase': is_ramp,
                    'fall_dist_m': fall_dist,
                })
        self._reset(env_ids)

    def _reset(self, env_ids: torch.Tensor):
        nan = float('nan')
        self.alarm_pos_xy[env_ids] = nan
        self.alarm_origin_xy[env_ids] = nan
        self.alarm_vel_x[env_ids] = nan
        self.had_takeover[env_ids] = False
        self.alarm_is_premature[env_ids] = False
        self.alarm_is_ramp_phase[env_ids] = False

    def summary(self) -> dict:
        out = {
            'late_fail_count':              self.late_fail_count,
            'late_fail_past_edge_count':    self.late_fail_past_edge_count,
            'late_fail_on_platform_count':  self.late_fail_on_platform_count,
            'missed_fall_count':            self.missed_fall_count,
        }
        if self.alarm_records:
            n_prem = sum(1 for r in self.alarm_records if r.get('is_premature', False))
            n_preterm = sum(1 for r in self.alarm_records
                           if r.get('is_premature', False) and r.get('is_ramp_phase', False))
            n_true_prem = n_prem - n_preterm
            out['premature_alarm_count']      = n_prem
            out['preterminated_alarm_count']  = n_preterm
            out['true_premature_alarm_count'] = n_true_prem
            out['total_alarm_count']          = len(self.alarm_records)
            prem_records = [r for r in self.alarm_records if r.get('is_premature', False)]
            if prem_records:
                prem_falls = sum(1 for r in prem_records if r.get('is_fall', False))
                out['premature_alarm_fall_rate'] = prem_falls / len(prem_records)
            else:
                out['premature_alarm_fall_rate'] = None
            mature = [r for r in self.alarm_records if not r.get('is_premature', False)]
            dists  = [r['alarm_dist_m']  for r in mature if not np.isnan(r['alarm_dist_m'])]
            vels   = [r['alarm_vel_mps'] for r in mature if not np.isnan(r['alarm_vel_mps'])]
            if dists:
                out['alarm_dist_mean_m'] = float(np.mean(dists))
                out['alarm_dist_std_m']  = float(np.std(dists))
                out['alarm_dist_min_m']  = float(np.min(dists))
            if vels:
                out['alarm_vel_mean_mps'] = float(np.mean(vels))
                out['alarm_vel_std_mps']  = float(np.std(vels))
        if self.stop_records:
            mature_stops = [r for r in self.stop_records if not r.get('is_premature', False)]
            margins = [r['stop_margin_m'] for r in mature_stops
                       if not np.isnan(r['stop_margin_m'])]
            if margins:
                out['stop_margin_mean_m'] = float(np.mean(margins))
                out['stop_margin_std_m']  = float(np.std(margins))
                out['stop_margin_min_m']  = float(np.min(margins))
        out['alarm_records'] = self.alarm_records
        out['stop_records']  = self.stop_records
        return out

    def print_summary(self, fall_count: int, takeover_ok_count: int, push_count: int = 0):
        valid = fall_count + takeover_ok_count
        print(f"\n  --- Tier-2 Metrics ---")
        print(f"  Push Count:     {push_count} total pushes across {valid} episodes"
              f"  (avg {push_count/max(valid,1):.2f}/ep, should be 1.0)")
        n_total_fall = self.late_fail_count + self.missed_fall_count
        if self.late_fail_count > 0 and valid > 0:
            print(f"  Late Fail:      {self.late_fail_count}/{valid} = "
                  f"{self.late_fail_count/valid*100:.1f}%  (takeover fired but fell anyway)")
            if self.late_fail_past_edge_count + self.late_fail_on_platform_count > 0:
                print(f"    ├─ past edge:    {self.late_fail_past_edge_count}"
                      f"  (alarm too late, robot already off platform)")
                print(f"    └─ on platform:  {self.late_fail_on_platform_count}"
                      f"  (braking collision, upper body hit platform)")
        else:
            print(f"  Late Fail:      {self.late_fail_count}  (0 = filter fully stopped robot)")
        if self.missed_fall_count > 0:
            print(f"  Missed (no alarm): {self.missed_fall_count}/{n_total_fall} falls"
                  f"  (filter never fired)")
        s = self.summary()
        n_prem    = s.get('premature_alarm_count', 0)
        n_preterm = s.get('preterminated_alarm_count', 0)
        n_trueprem = s.get('true_premature_alarm_count', 0)
        n_total   = s.get('total_alarm_count', 0)
        n_mat     = n_total - n_prem
        if n_prem > 0:
            print(f"  Premature Alarms: {n_prem}/{n_total} excluded from dist/margin stats"
                  f"  (speed < frac*cmd_vx at alarm time)")
            if n_preterm > 0 or n_trueprem > 0:
                print(f"    ├─ pre-terminated: {n_preterm}"
                      f"  (cmd still ramping after reset, robot not at target speed)")
                print(f"    └─ true premature: {n_trueprem}"
                      f"  (cmd at target speed, robot still slow — possible false alarm)")
        if 'alarm_dist_mean_m' in s:
            print(f"  Alarm Distance: mean={s['alarm_dist_mean_m']:.2f}m  "
                  f"std={s['alarm_dist_std_m']:.2f}m  "
                  f"min={s['alarm_dist_min_m']:.2f}m  (+ = alarm before edge, n={n_mat})")
        if 'alarm_vel_mean_mps' in s:
            print(f"  Alarm Velocity: mean={s['alarm_vel_mean_mps']:.2f}m  "
                  f"std={s['alarm_vel_std_mps']:.2f}m/s")
        if 'stop_margin_mean_m' in s:
            print(f"  Stop Margin:    mean={s['stop_margin_mean_m']:.2f}m  "
                  f"std={s['stop_margin_std_m']:.2f}m  "
                  f"min={s['stop_margin_min_m']:.2f}m  (+ = stopped before edge)")
        if not self.alarm_records:
            print(f"  (No alarms fired during this run)")


# =====================================================================
#               Main
# =====================================================================
def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.eval_terrain == 'flat':
        from legged_lab.terrains.terrain_generator_cfg import FLAT_MESH_TERRAINS_CFG
        env_cfg.scene.terrain_generator = FLAT_MESH_TERRAINS_CFG
        env_cfg.scene.terrain_generator.curriculum = False
    elif args_cli.eval_terrain == 'cliff':
        from legged_lab.terrains.terrain_generator_cfg import CLIFF_EVALUATION_TERRAINS_CFG
        env_cfg.scene.terrain_generator = CLIFF_EVALUATION_TERRAINS_CFG
        env_cfg.scene.terrain_generator.curriculum = False
        print(f"[INFO] Using CLIFF_EVALUATION_TERRAINS_CFG: "
              f"{CLIFF_EVALUATION_TERRAINS_CFG.num_rows}×{CLIFF_EVALUATION_TERRAINS_CFG.num_cols} grid, "
              f"tile size {CLIFF_EVALUATION_TERRAINS_CFG.size}")

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False

    _dc_cfg, _ = task_registry.get_cfgs("go2_data_collection")
    env_cfg.scene.camera = _dc_cfg.scene.camera

    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt
    env_cfg.scene.env_spacing = 10
    env_cfg.scene.max_episode_length_s = args_cli.max_episode_length_s

    import mdp
    if hasattr(env_cfg.domain_rand.events, 'push_robot'):
        env_cfg.domain_rand.events.push_robot = None

    if args_cli.no_push or args_cli.push_vx is None:
        _push_vx_val = None
        print("[INFO] Push DISABLED" + (" (--no_push)" if args_cli.no_push else " (no --push_vx)"))
    else:
        _push_vx_val = args_cli.push_vx
        print(f"[INFO] Fixed-step push: vx_impulse={_push_vx_val:.1f} m/s (body_frame)  "
              f"fires once per episode at step ∈ [{args_cli.push_frac_low:.2f}, {args_cli.push_frac_high:.2f}] * T_edge")

    if args_cli.fixed_vx is not None:
        env_cfg.commands.ranges.lin_vel_x = (args_cli.fixed_vx, args_cli.fixed_vx)
        env_cfg.commands.ranges.lin_vel_y = (args_cli.fixed_vy, args_cli.fixed_vy)
        env_cfg.commands.ranges.ang_vel_z = (args_cli.fixed_yaw, args_cli.fixed_yaw)

    if args_cli.fixed_vx is not None and abs(args_cli.fixed_vx) > 1e-3:
        _step_dt_auto = env_cfg.sim.dt * env_cfg.sim.decimation
        _t_edge_steps = int(np.ceil(args_cli.platform_half_width
                                    / (abs(args_cli.fixed_vx) * _step_dt_auto)))
        _min_steps = _t_edge_steps + args_cli.takeover_duration + 100 + args_cli.warmup_steps
        if args_cli.max_steps is None or args_cli.max_steps < _min_steps:
            _old = args_cli.max_steps
            args_cli.max_steps = _min_steps
            print(f"[INFO] Auto-adjusted max_steps: {_old} → {_min_steps}  "
                  f"(T_edge={_t_edge_steps} + takeover={args_cli.takeover_duration} + buffer=100"
                  f"{f' + warmup={args_cli.warmup_steps}' if args_cli.warmup_steps > 0 else ''})")
        _min_ep_s = (args_cli.max_steps + args_cli.warmup_steps) * _step_dt_auto + 1.0
        if env_cfg.scene.max_episode_length_s < _min_ep_s:
            _old_s = env_cfg.scene.max_episode_length_s
            env_cfg.scene.max_episode_length_s = _min_ep_s
            print(f"[INFO] Auto-adjusted max_episode_length_s: {_old_s:.1f}s → {_min_ep_s:.1f}s")

    env_class = task_registry.get_task_class("go2_data_collection")
    env = env_class(env_cfg, args_cli.headless)

    print(f"[INFO] Env ready: {env.num_envs} envs, {env.num_actions}D actions, device={env.device}")
    if args_cli.fixed_vx is not None:
        print(f"[INFO] Fixed command ranges: vx={args_cli.fixed_vx}  vy={args_cli.fixed_vy}  "
              f"yaw={args_cli.fixed_yaw}  (ranges locked, resample-safe)")

    # ==================== 视频录制 ====================
    video_writer = None
    rgb_annotator = None
    render_product = None
    if args_cli.record_video:
        import imageio
        import omni.replicator.core as rep
        vid_path = args_cli.video_path
        Path(vid_path).parent.mkdir(parents=True, exist_ok=True)
        W, H_vid = args_cli.video_width, args_cli.video_height
        env.sim.render()
        render_product = rep.create.render_product("/OmniverseKit_Persp", (W, H_vid))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])
        for _ in range(5):
            env.sim.render()
        video_fps = 1.0 / step_dt
        video_writer = imageio.get_writer(vid_path, fps=video_fps, codec="libx264",
                                          quality=8, pixelformat="yuv420p")
        print(f"[INFO] Video: {vid_path}  {W}x{H_vid} @ {video_fps:.1f}fps")

    # ==================== PPO policy ====================
    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume),
                            device=agent_cfg.device)
    runner.load(resume, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[INFO] PPO policy loaded: {resume}")

    # ==================== 初始化 ====================
    robot: Articulation = env.scene["robot"]
    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(all_ids)
    obs_dict = env.get_observations()

    dummy = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(3):
        obs_dict, _, _, _ = env.step(dummy)
    print("[INFO] Sensors warmed up")

    # ==================== Critic Filter ====================
    critic_filter = load_critic_filter(env, args_cli)

    # ==================== 评估状态 ====================
    step_count = 0
    episode_count = 0
    danger_count = 0
    fall_count = 0
    takeover_ok_count = 0
    alarm_step_count = 0

    takeover_steps   = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    pending_reset    = torch.zeros(env.num_envs, dtype=torch.bool,  device=env.device)
    episode_step_buf = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    low_q_run        = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    grace_left       = torch.full((env.num_envs,), fill_value=args_cli.reset_grace_k,
                                  dtype=torch.int32, device=env.device)
    TAKEOVER_DUR  = args_cli.takeover_duration
    ALARM_K         = max(1, int(args_cli.alarm_k))
    RESET_GRACE_K   = max(1, int(args_cli.reset_grace_k))
    SPEED_MATCH_FRAC = float(args_cli.speed_match_frac)
    CMD_ACCEL_RATE = float(args_cli.cmd_accel_rate)
    cmd_ramp = torch.zeros(env.num_envs, device=env.device)
    CMD_DECEL_RATE = float(args_cli.cmd_decel_rate)
    cmd_decel_vx = torch.zeros(env.num_envs, device=env.device)
    PREMATURE_SPEED_FRAC = float(args_cli.premature_speed_frac)

    _do_manual_push = _push_vx_val is not None and args_cli.fixed_vx is not None
    push_count_total = 0
    if _do_manual_push:
        _push_T  = args_cli.platform_half_width / (args_cli.fixed_vx * step_dt)
        _push_lo = max(1, int(_push_T * args_cli.push_frac_low))
        _push_hi = max(_push_lo + 1, int(_push_T * args_cli.push_frac_high))
        push_step_buf = torch.randint(_push_lo, _push_hi + 1, (env.num_envs,),
                                      dtype=torch.int32, device=env.device)
        print(f"[INFO] Push step range: [{_push_lo}, {_push_hi}] steps "
              f"(T_edge={_push_T:.1f} steps @ vx={args_cli.fixed_vx}m/s, dt={step_dt:.3f}s)")
    else:
        push_step_buf = torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device)
        if _push_vx_val is not None and args_cli.fixed_vx is None:
            print("[WARN] --push_vx set but --fixed_vx not set; push disabled (T_edge unknown).")

    # Eval Q buffers
    if args_cli.eval_terrain:
        ep_q_buf: list = [[] for _ in range(env.num_envs)]
        fall_q_trajs: list = []
        safe_q_trajs: list = []
        EVAL_H = args_cli.eval_h
        _ssf = args_cli.eval_safe_strict_f
        EVAL_SAFE_STRICT_F = _ssf if _ssf is not None else (
            (args_cli.max_steps // 2) if args_cli.max_steps else 120)
        TAU_SWEEP = np.linspace(-1.0, 1.0, 200)
        print(f"[INFO] Eval Q buffers: H={EVAL_H}  safe_strict_f={EVAL_SAFE_STRICT_F}")

    # RSSM warmup state
    WARMUP_STEPS = max(0, int(args_cli.warmup_steps))
    warmup_buf = torch.full((env.num_envs,), fill_value=WARMUP_STEPS,
                            dtype=torch.int32, device=env.device)
    if WARMUP_STEPS > 0:
        print(f"[INFO] RSSM warmup experiment: {WARMUP_STEPS} steps of cmd=0 standing "
              f"before each episode's fixed_vx approach.")

    # ── OOD 低摩擦实验状态 ──────────────────────────────────────────────────
    OOD_FRICTION_STEP    = args_cli.ood_friction_step
    OOD_STATIC_MU        = args_cli.ood_static_friction
    OOD_DYNAMIC_MU       = args_cli.ood_dynamic_friction
    OOD_FRICTION_DECAY       = args_cli.ood_friction_decay
    OOD_FRICTION_DECAY_STEPS = max(1, args_cli.ood_friction_decay_steps)
    OOD_RANDOM_FLIP      = args_cli.ood_random_flip
    OOD_FRICTION_INTERVAL = max(1, args_cli.ood_friction_interval)
    OOD_FRICTION_PROB    = float(args_cli.ood_friction_prob)
    OOD_ZONE_MODE        = args_cli.ood_zone_mode
    OOD_ZONE_DIST        = float(args_cli.ood_zone_dist)
    OOD_CMD_VX           = args_cli.ood_cmd_vx          # None = no cmd override
    OOD_CMD_VY           = float(args_cli.ood_cmd_vy)

    # ── 方案 A: 执行器弱化 ────────────────────────────────────────────────────
    OOD_ACTUATOR_WEAKNESS = args_cli.ood_actuator_weakness
    OOD_KP_SCALE          = float(args_cli.ood_kp_scale)
    # Save normal actuator gains (one tensor per actuator, shape [num_envs, num_dofs_for_act])
    _ood_saved_stiffness: dict | None = None
    _ood_saved_damping:   dict | None = None
    _ood_weakness_active  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if OOD_ACTUATOR_WEAKNESS:
        _ood_saved_stiffness = {k: v.stiffness.clone() for k, v in robot.actuators.items()}
        _ood_saved_damping   = {k: v.damping.clone()   for k, v in robot.actuators.items()}
        print(f"[INFO] Approach A – Actuator Weakness ENABLED: kp_scale={OOD_KP_SCALE}")
        for k, v in robot.actuators.items():
            print(f"         actuator '{k}': stiffness shape={v.stiffness.shape}  "
                  f"nominal kp={v.stiffness[0, 0].item():.1f}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── 方案 B: 持续侧向外力 ──────────────────────────────────────────────────
    _ood_force_body: torch.Tensor | None = None
    if args_cli.ood_sustained_force is not None:
        _ood_force_body = torch.tensor(args_cli.ood_sustained_force,
                                       dtype=torch.float32, device=env.device)
        _mag = float(_ood_force_body.norm().item())
        print(f"[INFO] Approach B – Sustained Body Force ENABLED: "
              f"body-frame [{args_cli.ood_sustained_force[0]:.1f}, "
              f"{args_cli.ood_sustained_force[1]:.1f}, "
              f"{args_cli.ood_sustained_force[2]:.1f}] N  |F|={_mag:.1f}N")
    # ─────────────────────────────────────────────────────────────────────────

    # ── 场景 F1: HAA 关节刚度渐进衰减 ──────────────────────────────────────────
    F1_ENABLE      = args_cli.f1_enable
    F1_FAULT_STEP  = args_cli.f1_fault_step
    F1_DECAY_STEPS = max(1, args_cli.f1_decay_steps)
    F1_KP_MIN      = float(args_cli.f1_kp_min)
    f1_act_name:   str | None          = None
    f1_joint_col:  int | None          = None
    f1_kp_nominal: torch.Tensor | None = None  # (num_envs,) nominal KP for target joint
    if F1_ENABLE:
        _f1_target = args_cli.f1_target_joint
        for _aname, _act in robot.actuators.items():
            _jnames = list(_act.joint_names) if hasattr(_act, 'joint_names') else []
            if _f1_target in _jnames:
                f1_act_name  = _aname
                f1_joint_col = _jnames.index(_f1_target)
                f1_kp_nominal = _act.stiffness[:, f1_joint_col].clone()
                print(f"[INFO] F1 – HAA progressive KP decay ENABLED:")
                print(f"         joint='{_f1_target}'  actuator='{_aname}'  col={f1_joint_col}")
                print(f"         fault_step={F1_FAULT_STEP}  decay_steps={F1_DECAY_STEPS}  "
                      f"kp_min={F1_KP_MIN}  kp_nominal={f1_kp_nominal[0].item():.1f}")
                print(f"         failure completes at ep_step ~ "
                      f"{F1_FAULT_STEP + F1_DECAY_STEPS} "
                      f"({(F1_FAULT_STEP + F1_DECAY_STEPS)*env_cfg.sim.dt*env_cfg.sim.decimation:.1f}s)")
                break
        if f1_act_name is None:
            print(f"[WARN] F1: joint '{_f1_target}' not found in any actuator. "
                  f"Disabling F1.")
            print(f"       Available actuators and their joints:")
            for _aname, _act in robot.actuators.items():
                _jn = list(_act.joint_names) if hasattr(_act, 'joint_names') else ["(no joint_names attr)"]
                print(f"         '{_aname}': {_jn}")
            F1_ENABLE = False
    # per-env F1 state
    f1_active = torch.zeros(env.num_envs, dtype=torch.bool,  device=env.device)
    f1_alpha  = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    # ─────────────────────────────────────────────────────────────────────────

    # zone enabled if step-mode with a step, or position-mode (always computed),
    # OR if any ood disturbance mode is active
    _ood_enabled = ((OOD_FRICTION_STEP is not None) or (OOD_ZONE_MODE == 'position')
                    or OOD_ACTUATOR_WEAKNESS or (_ood_force_body is not None))

    # per-env OOD state (all on GPU except _ood_saved_materials which must be CPU)
    ood_in_zone    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ood_current_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ood_ever_low   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ood_decay_alpha = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    _ood_saved_materials: torch.Tensor | None = None

    if _ood_enabled:
        if OOD_FRICTION_STEP is not None or OOD_ZONE_MODE == 'position':
            _ood_saved_materials = robot.root_physx_view.get_material_properties().clone()
        _mode_str = (f"zone_mode=step  entry=ep_step≥{OOD_FRICTION_STEP}"
                     if OOD_ZONE_MODE == 'step'
                     else f"zone_mode=position  entry=dist_to_edge≤{OOD_ZONE_DIST}m")
        if OOD_FRICTION_STEP is not None or OOD_ZONE_MODE == 'position':
            _flip_str = (f"random flip every {OOD_FRICTION_INTERVAL} steps  p_low={OOD_FRICTION_PROB}"
                         if OOD_RANDOM_FLIP
                         else (f"decay over {OOD_FRICTION_DECAY_STEPS} steps"
                               if OOD_FRICTION_DECAY else "fixed trigger on zone entry"))
            print(f"[INFO] OOD friction ENABLED:  {_flip_str}")
            print(f"         {_mode_str}")
            print(f"         static={OOD_STATIC_MU}  dynamic={OOD_DYNAMIC_MU}")
            if _ood_saved_materials is not None:
                print(f"         max_shapes={robot.root_physx_view.max_shapes}  "
                      f"saved_mats.shape={_ood_saved_materials.shape}")
        if OOD_CMD_VX is not None:
            print(f"         cmd_override on entry: vx={OOD_CMD_VX}  vy={OOD_CMD_VY}  "
                  f"(accel_suppress disabled in OOD zone)")
        if not (OOD_FRICTION_STEP is not None or OOD_ZONE_MODE == 'position'):
            # disturbance-only mode: still use step-based zone
            print(f"[INFO] OOD zone (disturbance-only): {_mode_str}")
    # ────────────────────────────────────────────────────────────────────────

    # OOD alarm records + Cohen's d collector
    # Collect (q_val, is_low_friction) tuples when in OOD zone
    # Cohen's d = |mean_low - mean_high| / pooled_std  →  scale-free, measures friction sensitivity
    ood_alarm_records: list = []
    _ood_zone_samples: list = []   # list of (q_val: float, is_low: bool)

    # Tier-2 metrics tracker
    metrics = EpisodeMetrics(env.num_envs, args_cli.platform_half_width, env.device)
    metrics.try_load_terrain_edge(env)

    # Per-step log init; state: 0=RUNNING 1=GRACE 2=ACCEL 3=ALARM 4=TAKEOVER
    if args_cli.log_step_data:
        _step_log = {k: [] for k in [
            'global_step', 'env_id', 'ep_id', 'ep_step',
            'state', 'q_val', 'g_val',
            'q_target', 'q_critic_minus_target',
            'target_g_min', 'target_g_argmin', 'target_v_stop_tail',
            'target_tail_is_bottleneck', 'target_g_seq',
            'cmd_vx', 'cmd_vy', 'cmd_yaw', 'spd',
            'ood_in_zone', 'ood_current_low',
            'f1_alpha',
        ]}
        _ep_id_per_env = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        _step_log = None
        _ep_id_per_env = None

    log_g, log_q, log_cmd = [], [], []

    env_timeout_steps = int(np.ceil(args_cli.max_episode_length_s / step_dt))
    print("\n[INFO] Starting evaluation " +
          ("(monitor-only)" if args_cli.monitor_only else "(filter active)") + "...")
    _ramp_info = (f"cmd_ramp={CMD_ACCEL_RATE:.1f}m/s²" if CMD_ACCEL_RATE > 0 else "cmd_ramp=disabled")
    print(f"[INFO] Alarm rule: {ALARM_K} consecutive frames with Q < {args_cli.safety_threshold:.4f}  "
          f"reset_grace={RESET_GRACE_K}f  {_ramp_info}")
    if args_cli.fixed_vx is not None:
        print(f"[INFO] Command override: vx={args_cli.fixed_vx}  vy={args_cli.fixed_vy}  "
              f"yaw={args_cli.fixed_yaw}")
    print()

    try:
        while simulation_app.is_running():
            if args_cli.max_episodes and episode_count >= args_cli.max_episodes:
                print(f"\n[INFO] Reached {args_cli.max_episodes} episodes, exiting.")
                break

            with torch.inference_mode():
                # RSSM warmup phase: zero cmd for envs still in warmup
                in_warmup = warmup_buf > 0
                if args_cli.fixed_vx is not None:
                    env.command_generator.command[:, 0] = args_cli.fixed_vx
                    env.command_generator.command[:, 1] = args_cli.fixed_vy
                    env.command_generator.command[:, 2] = args_cli.fixed_yaw
                if in_warmup.any():
                    env.command_generator.command[in_warmup, 0] = 0.0
                    env.command_generator.command[in_warmup, 1] = 0.0
                    env.command_generator.command[in_warmup, 2] = 0.0

                # 1. PPO action
                actions_raw = policy(obs_dict)

                # 2. Current command
                cmd_current = env.command_generator.command[:, :3].clone()

                # Cmd ramp
                _cmd_ramp_target = None
                if CMD_ACCEL_RATE > 0:
                    _target_abs = cmd_current[:, 0].abs()
                    _vx_sign    = cmd_current[:, 0].sign()
                    _advance    = ~(takeover_steps > 0)
                    cmd_ramp    = torch.where(
                        _advance,
                        (cmd_ramp + CMD_ACCEL_RATE * step_dt).clamp(max=_target_abs),
                        cmd_ramp,
                    )
                    _ramped_vx  = cmd_ramp * _vx_sign
                    env.command_generator.command[:, 0] = _ramped_vx
                    cmd_current[:, 0] = _ramped_vx
                    _cmd_ramp_target = _target_abs

                # OOD cmd override: spike cmd on zone entry to force proprio stress
                # (flat ground → depth unchanged; proprio explodes due to slip at low μ)
                # Takeover logic later in this loop still wins if alarm fires.
                if _ood_enabled and OOD_CMD_VX is not None and ood_in_zone.any():
                    _ood_cmd_mask = ood_in_zone & (takeover_steps == 0) & ~in_warmup
                    if _ood_cmd_mask.any():
                        env.command_generator.command[_ood_cmd_mask, 0] = OOD_CMD_VX
                        env.command_generator.command[_ood_cmd_mask, 1] = OOD_CMD_VY
                        cmd_current[_ood_cmd_mask, 0] = OOD_CMD_VX
                        cmd_current[_ood_cmd_mask, 1] = OOD_CMD_VY
                        if CMD_ACCEL_RATE > 0:
                            # Sync ramp state so it doesn't fight the override next iteration
                            cmd_ramp[_ood_cmd_mask] = abs(OOD_CMD_VX)

                # 3. Critic filter
                cmd_out, is_safe, q_vals, g_vals, q_rand_vals = critic_filter.filter_cmd(
                    obs_dict, cmd_current,
                    apply_filter=not args_cli.monitor_only,
                    compute_q_rand=args_cli.show_q_rand,
                )
                if args_cli.diagnose_q_target:
                    _qdiag = critic_filter.evaluate_brake_target(cmd_current)
                    q_target_vals = _qdiag["q_target"]
                    target_g_seq = _qdiag["g_seq"]
                    target_g_min = _qdiag["g_min"]
                    target_g_argmin = _qdiag["g_argmin"]
                    target_v_stop_tail = _qdiag["v_stop_tail"]
                    target_tail_is_bottleneck = _qdiag["tail_is_bottleneck"]
                else:
                    q_target_vals = torch.full_like(q_vals, float("nan"))
                    target_g_seq = torch.full(
                        (env.num_envs, 1), float("nan"), device=env.device)
                    target_g_min = torch.full_like(q_vals, float("nan"))
                    target_g_argmin = torch.zeros_like(q_vals, dtype=torch.long)
                    target_v_stop_tail = torch.full_like(q_vals, float("nan"))
                    target_tail_is_bottleneck = torch.zeros_like(q_vals, dtype=torch.bool)

                # 4. K-consecutive alarm
                unsafe_now = ~is_safe
                in_grace = grace_left > 0
                if SPEED_MATCH_FRAC > 0:
                    _actual_spd = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    _cmd_vx = env.command_generator.command[:, 0].abs()
                    in_accel_suppress = (_actual_spd < SPEED_MATCH_FRAC * _cmd_vx) & (_cmd_vx > 0.3)
                else:
                    in_accel_suppress = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                    in_accel = (cmd_ramp < _cmd_ramp_target - 0.05) & (_cmd_ramp_target > 0.3)
                else:
                    in_accel = in_accel_suppress
                # In OOD zone: disable accel-suppress so filter alarms on proprio stress immediately
                if _ood_enabled and ood_in_zone.any():
                    in_accel_suppress = in_accel_suppress & ~ood_in_zone
                    in_accel = in_accel & ~ood_in_zone
                low_q_run = torch.where(unsafe_now & ~in_grace & ~in_accel_suppress, low_q_run + 1,
                                        torch.zeros_like(low_q_run))
                alarm_now = low_q_run >= ALARM_K
                if WARMUP_STEPS > 0:
                    alarm_now = alarm_now & ~in_warmup
                    low_q_run = torch.where(in_warmup, torch.zeros_like(low_q_run), low_q_run)
                alarm_step_count += int(alarm_now.sum().item())

                # Record first alarm position/velocity
                if PREMATURE_SPEED_FRAC > 0:
                    _spd_prem   = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    _cmdvx_prem = env.command_generator.command[:, 0].abs()
                    _alarm_premature = (_spd_prem < PREMATURE_SPEED_FRAC * _cmdvx_prem) & (_cmdvx_prem > 0.3)
                else:
                    _alarm_premature = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                    _alarm_ramp_phase = (cmd_ramp < _cmd_ramp_target - 0.05) & (_cmd_ramp_target > 0.3)
                else:
                    _alarm_ramp_phase = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                metrics.record_alarm(robot, alarm_now, env, _alarm_premature, _alarm_ramp_phase)

                # ── OOD alarm records: log alarms that fire while in OOD zone ────────
                if _ood_enabled and alarm_now.any():
                    for _i in torch.where(alarm_now)[0].tolist():
                        if ood_in_zone[_i]:
                            _since = (int(episode_step_buf[_i].item()) - (OOD_FRICTION_STEP or 0))
                            ood_alarm_records.append({
                                'env_id':          _i,
                                'episode_step':    int(episode_step_buf[_i].item()),
                                'steps_since_ood': _since,
                                'q_val':           float(q_vals[_i].item()),
                                'g_val':           float(g_vals[_i].item()),
                                'ood_current_low': bool(ood_current_low[_i].item()),
                                'is_fall':         None,
                            })
                # ─────────────────────────────────────────────────────────────────────

                if args_cli.eval_terrain:
                    for _i in range(env.num_envs):
                        ep_q_buf[_i].append(float(q_vals[_i].item()))

                if args_cli.log_values:
                    log_g.append(g_vals.cpu().numpy().copy())
                    log_q.append(q_vals.cpu().numpy().copy())
                    log_cmd.append(cmd_current.cpu().numpy().copy())

                # 5. Takeover logic
                if not args_cli.monitor_only:
                    danger = alarm_now & (takeover_steps == 0)
                    if danger.any():
                        takeover_steps[danger] = TAKEOVER_DUR
                        pending_reset[danger] = True
                        if CMD_DECEL_RATE > 0:
                            cmd_decel_vx[danger] = cmd_current[danger, 0]
                        d_envs = torch.where(danger)[0].tolist()
                        d_q = q_vals[danger].tolist()
                        d_q_target = q_target_vals[danger].tolist()
                        d_g = g_vals[danger].tolist()
                        d_cmd = cmd_current[danger].tolist()
                        danger_count += danger.sum().item()
                        metrics.mark_takeover(danger)
                        print(f"\033[93m[WARN] Step {step_count}: DANGER in envs {d_envs}\033[0m")
                        for i, e in enumerate(d_envs):
                            qr_diag = (f"  Q_rand={q_rand_vals[danger][i].item():.3f}"
                                       if args_cli.show_q_rand else "")
                            ood_tag = (" \033[95m[OOD-LOW]\033[0m" if ood_current_low[e]
                                       else (" \033[35m[OOD-ZONE]\033[0m" if ood_in_zone[e] else ""))
                            print(f"  env{e}: Q={d_q[i]:.3f}{qr_diag}  "
                                  f"Qtarget={d_q_target[i]:.3f}  "
                                  f"gap={d_q[i] - d_q_target[i]:+.3f}  "
                                  f"g={d_g[i]:.3f}  "
                                  f"rollout_gmin={target_g_min[e].item():+.3f}"
                                  f"@{int(target_g_argmin[e].item())}  "
                                  f"Vstop_tail={target_v_stop_tail[e].item():+.3f}  "
                                  f"bottleneck={'Vstop' if target_tail_is_bottleneck[e] else 'g'}  "
                                  f"cmd=({d_cmd[i][0]:.2f},{d_cmd[i][1]:.2f},{d_cmd[i][2]:.2f}){ood_tag}")
                        print(f"  → Standing still for {TAKEOVER_DUR} steps")

                    is_takeover = takeover_steps > 0
                    if is_takeover.any():
                        if CMD_DECEL_RATE > 0:
                            _d_mag = cmd_decel_vx.abs()
                            _d_sgn = cmd_decel_vx.sign()
                            _d_mag_new = (_d_mag - CMD_DECEL_RATE * step_dt).clamp(min=0)
                            cmd_decel_vx = torch.where(is_takeover, _d_mag_new * _d_sgn, cmd_decel_vx)
                            env.command_generator.command[is_takeover, 0] = cmd_decel_vx[is_takeover]
                            env.command_generator.command[is_takeover, 1] = 0.0
                            env.command_generator.command[is_takeover, 2] = 0.0
                        else:
                            env.command_generator.command[is_takeover] = 0.0
                        actions_safe = policy(obs_dict)
                        actions_raw[is_takeover] = actions_safe[is_takeover]
                        low_q_run[is_takeover] = 0
                        if step_count % 20 == 0:
                            tk_envs = torch.where(is_takeover)[0].tolist()
                            remaining = takeover_steps[is_takeover].tolist()
                            print(f"\033[91m[TAKEOVER] Step {step_count}: envs {tk_envs}, "
                                  f"remaining {remaining}\033[0m")

                    takeover_steps = torch.clamp(takeover_steps - 1, min=0)

                    tk_done = pending_reset & (takeover_steps == 0)
                    if tk_done.any():
                        done_envs = torch.where(tk_done)[0]
                        print(f"\033[94m[RESET] Step {step_count}: envs {done_envs.tolist()}\033[0m")
                        metrics.record_stop(robot, done_envs, env)
                        env.reset(done_envs)
                        obs_dict = env.get_observations()
                        pending_reset[tk_done] = False
                        low_q_run[tk_done] = 0
                        episode_step_buf[tk_done] = 0
                        grace_left[tk_done] = RESET_GRACE_K
                        cmd_ramp[tk_done] = 0.0
                        cmd_decel_vx[tk_done] = 0.0
                        warmup_buf[tk_done] = WARMUP_STEPS

                        # ── [OOD] restore friction / actuator on takeover-done reset ──
                        if _ood_enabled:
                            if _ood_saved_materials is not None and ood_current_low[tk_done].any():
                                _r = torch.where(tk_done & ood_current_low)[0].cpu()
                                restore_friction(robot, _r, _ood_saved_materials)
                            if OOD_ACTUATOR_WEAKNESS and _ood_weakness_active[tk_done].any():
                                restore_actuator_normal(
                                    robot, torch.where(tk_done & _ood_weakness_active)[0],
                                    _ood_saved_stiffness, _ood_saved_damping)
                                _ood_weakness_active[tk_done] = False
                        ood_current_low[tk_done] = False
                        ood_in_zone[tk_done]     = False
                        ood_ever_low[tk_done]    = False
                        ood_decay_alpha[tk_done] = 0.0
                        # ─────────────────────────────────────────────────────────────
                        # ── F1 restore on takeover-done reset ────────────────────────
                        if F1_ENABLE and f1_act_name is not None and f1_active[tk_done].any():
                            restore_f1_stiffness(robot, torch.where(tk_done)[0],
                                                 f1_act_name, f1_joint_col, f1_kp_nominal)
                        f1_active[tk_done] = False
                        f1_alpha[tk_done]  = 0.0
                        # ─────────────────────────────────────────────────────────────

                        if _do_manual_push:
                            push_step_buf[tk_done] = torch.randint(
                                _push_lo, _push_hi + 1, (int(tk_done.sum().item()),),
                                dtype=torch.int32, device=env.device)
                        critic_filter.finalize_episode(done_envs, is_fall=False)
                        episode_count += tk_done.sum().item()
                        takeover_ok_count += tk_done.sum().item()
                        if _ep_id_per_env is not None:
                            _ep_id_per_env[done_envs] += 1
                        is_fall_vec_tk = torch.zeros(done_envs.shape[0], dtype=torch.bool,
                                                     device=env.device)
                        metrics.on_episode_done(done_envs, is_fall_vec_tk)
                        if args_cli.eval_terrain:
                            for _i in done_envs.tolist():
                                _arr = np.array(ep_q_buf[_i])
                                if len(_arr):
                                    safe_q_trajs.append(_arr)
                                ep_q_buf[_i] = []
                        continue

                # 6. Step env
                if args_cli.record_video and video_writer is not None and not args_cli.video_no_follow:
                    _follow_idx = min(args_cli.video_env_idx, env.num_envs - 1)
                    _robot_pos = env.robot.data.root_pos_w[_follow_idx].cpu().numpy()
                    _offset = np.array(args_cli.video_follow_offset, dtype=np.float64)
                    env.sim.set_camera_view(eye=(_robot_pos + _offset).tolist(),
                                            target=_robot_pos.tolist())

                metrics.update_pre_step(robot, env)

                # ── [OOD] zone detection + random/fixed friction flip ────────────
                if _ood_enabled:
                    # 1. Update zone membership
                    if OOD_ZONE_MODE == 'position':
                        try:
                            _orig = env.scene.terrain.env_origins
                            _pxy  = robot.data.root_pos_w[:, :2]
                            _dx   = (_pxy[:, 0] - _orig[:, 0]).abs()
                            _dy   = (_pxy[:, 1] - _orig[:, 1]).abs()
                            _d2e  = args_cli.platform_half_width - torch.max(_dx, _dy)
                            ood_in_zone = _d2e.le(OOD_ZONE_DIST)
                        except Exception:
                            _fb = OOD_FRICTION_STEP if OOD_FRICTION_STEP is not None else 999999
                            ood_in_zone = episode_step_buf >= _fb
                    else:
                        # step mode: use ood_friction_step if set, else 0 (always in zone)
                        _entry = OOD_FRICTION_STEP if OOD_FRICTION_STEP is not None else 0
                        ood_in_zone = (episode_step_buf >= _entry) & ~in_warmup

                    # 2. Restore friction for envs that LEFT the zone
                    _left = ood_current_low & ~ood_in_zone
                    if _left.any() and _ood_saved_materials is not None:
                        restore_friction(robot, torch.where(_left)[0].cpu(), _ood_saved_materials)
                        ood_current_low[_left] = False
                        ood_decay_alpha[_left] = 0.0

                    # 3a. Approach B: apply sustained body force every step while in zone
                    if _ood_force_body is not None and ood_in_zone.any():
                        _zone_ids = torch.where(ood_in_zone)[0]
                        apply_ood_sustained_force(robot, env, _zone_ids, _ood_force_body)

                    # 3b. Approach A: apply/restore actuator weakness on zone entry/exit
                    if OOD_ACTUATOR_WEAKNESS:
                        # Apply weakness to envs that JUST entered zone
                        _weakness_entry = ood_in_zone & ~_ood_weakness_active
                        if _weakness_entry.any():
                            apply_ood_actuator_weakness(
                                robot, torch.where(_weakness_entry)[0],
                                OOD_KP_SCALE, _ood_saved_stiffness, _ood_saved_damping)
                            _ood_weakness_active[_weakness_entry] = True
                            print(f"\033[95m[OOD-A] Step {step_count}: "
                                  f"actuator weakness kp×{OOD_KP_SCALE} "
                                  f"envs {torch.where(_weakness_entry)[0].tolist()}\033[0m")
                        # Restore for envs that LEFT zone
                        _weakness_exit = _ood_weakness_active & ~ood_in_zone
                        if _weakness_exit.any():
                            restore_actuator_normal(
                                robot, torch.where(_weakness_exit)[0],
                                _ood_saved_stiffness, _ood_saved_damping)
                            _ood_weakness_active[_weakness_exit] = False

                    # 3c. Apply friction for envs IN zone
                    if ood_in_zone.any() and _ood_saved_materials is not None:
                        if OOD_RANDOM_FLIP:
                            _flip = ood_in_zone & (episode_step_buf % OOD_FRICTION_INTERVAL == 0)
                            if _flip.any():
                                for _oi in torch.where(_flip)[0].tolist():
                                    _go_low = torch.rand(1).item() < OOD_FRICTION_PROB
                                    if _go_low and not bool(ood_current_low[_oi]):
                                        apply_ood_friction(robot, torch.tensor([_oi]),
                                                           OOD_STATIC_MU, OOD_DYNAMIC_MU)
                                        ood_current_low[_oi] = True
                                        ood_ever_low[_oi] = True
                                    elif not _go_low and bool(ood_current_low[_oi]):
                                        restore_friction(robot, torch.tensor([_oi]),
                                                         _ood_saved_materials)
                                        ood_current_low[_oi] = False
                        elif OOD_FRICTION_DECAY:
                            # Decay mode: linearly ramp friction from normal → OOD
                            _entry_mask = ood_in_zone & ~ood_ever_low
                            if _entry_mask.any():
                                ood_ever_low[_entry_mask] = True
                                ood_current_low[_entry_mask] = True
                                print(f"\033[95m[OOD-DECAY] Step {step_count}: "
                                      f"friction decay started over {OOD_FRICTION_DECAY_STEPS} steps "
                                      f"→ μ_s={OOD_STATIC_MU} μ_d={OOD_DYNAMIC_MU} "
                                      f"envs {torch.where(_entry_mask)[0].tolist()}\033[0m")
                            if ood_in_zone.any():
                                _prev_alpha = ood_decay_alpha[ood_in_zone].clone()
                                ood_decay_alpha[ood_in_zone] = (
                                    ood_decay_alpha[ood_in_zone] + 1.0 / OOD_FRICTION_DECAY_STEPS
                                ).clamp(max=1.0)
                                _decay_ids_gpu = torch.where(ood_in_zone)[0]
                                _decay_ids_cpu = _decay_ids_gpu.cpu()
                                apply_ood_friction_lerp(
                                    robot, _decay_ids_cpu, _ood_saved_materials,
                                    OOD_STATIC_MU, OOD_DYNAMIC_MU,
                                    ood_decay_alpha.cpu())
                                # Log first step when any env fully reaches OOD friction
                                _newly_full = _decay_ids_gpu[
                                    (_prev_alpha < 1.0) & (ood_decay_alpha[_decay_ids_gpu] >= 1.0)]
                                if _newly_full.numel() > 0:
                                    print(f"\033[95m[OOD-DECAY] Step {step_count}: "
                                          f"full OOD friction reached "
                                          f"envs {_newly_full.tolist()}\033[0m")
                        else:
                            # Fixed: apply once on zone entry
                            _entry_mask = ood_in_zone & ~ood_ever_low
                            if _entry_mask.any():
                                _eids = torch.where(_entry_mask)[0].cpu()
                                apply_ood_friction(robot, _eids, OOD_STATIC_MU, OOD_DYNAMIC_MU)
                                ood_current_low[_entry_mask] = True
                                ood_ever_low[_entry_mask] = True
                                print(f"\033[95m[OOD] Step {step_count}: "
                                      f"μ={OOD_STATIC_MU}/{OOD_DYNAMIC_MU} "
                                      f"envs {_eids.tolist()}\033[0m")

                    # 4. Collect (q_val, is_low) for Cohen's d computation
                    for _oi in torch.where(ood_in_zone)[0].tolist():
                        _ood_zone_samples.append(
                            (float(q_vals[_oi].item()), bool(ood_current_low[_oi].item()))
                        )
                # ─────────────────────────────────────────────────────────────────

                # ── 场景 F1: HAA 关节刚度渐进衰减 (每步更新 KP) ───────────────────────
                if F1_ENABLE and f1_act_name is not None:
                    _f1_past_fault = (episode_step_buf >= F1_FAULT_STEP) & ~in_warmup
                    if _f1_past_fault.any():
                        _f1_ids = torch.where(_f1_past_fault)[0]
                        # 首次激活：打印日志
                        _newly = _f1_ids[~f1_active[_f1_ids]]
                        if _newly.numel() > 0:
                            print(f"\033[95m[F1] Step {step_count}: "
                                  f"KP decay started envs {_newly.tolist()}\033[0m")
                            f1_active[_newly] = True
                        # 计算衰减进度 α ∈ [0, 1]
                        _steps_since = (episode_step_buf.float() - F1_FAULT_STEP).clamp(min=0.0)
                        f1_alpha[_f1_ids] = (_steps_since[_f1_ids] / F1_DECAY_STEPS).clamp(max=1.0)
                        # 写入 actuator stiffness
                        apply_f1_decay_step(robot, _f1_ids, f1_act_name, f1_joint_col,
                                            f1_kp_nominal, F1_KP_MIN, f1_alpha)
                        # 打印 α=1 首次到达（KP 已归零）
                        _fully_degraded = _f1_ids[f1_alpha[_f1_ids] >= 1.0]
                        _newly_full = _fully_degraded[~f1_active[_fully_degraded + env.num_envs * 0]]
                        # 简化：每隔 print_interval 步打印当前 KP 值
                        if step_count % args_cli.print_interval == 0 and _f1_ids.numel() > 0:
                            _ei0 = _f1_ids[0].item()
                            _kp_now = robot.actuators[f1_act_name].stiffness[_ei0, f1_joint_col].item()
                            print(f"[F1] ep_step={episode_step_buf[_ei0].item()}  "
                                  f"α={f1_alpha[_ei0].item():.3f}  "
                                  f"KP[{args_cli.f1_target_joint}]={_kp_now:.2f}")
                # ────────────────────────────────────────────────────────────────────

                obs_dict, rewards, dones, infos = env.step(actions_raw)
                step_count += 1

                if args_cli.record_video and video_writer is not None:
                    try:
                        _rgb_raw = rgb_annotator.get_data()
                        if isinstance(_rgb_raw, np.ndarray) and _rgb_raw.size > 0:
                            video_writer.append_data(_rgb_raw[:, :, :3])
                        else:
                            video_writer.append_data(np.zeros(
                                (args_cli.video_height, args_cli.video_width, 3), dtype=np.uint8))
                    except Exception as _verr:
                        print(f"[WARN] Video frame error: {_verr}")
                        video_writer.append_data(np.zeros(
                            (args_cli.video_height, args_cli.video_width, 3), dtype=np.uint8))
                episode_step_buf += 1
                grace_left = torch.clamp(grace_left - 1, min=0)
                if WARMUP_STEPS > 0 and in_warmup.any():
                    warmup_buf = torch.clamp(warmup_buf - 1, min=0)
                    episode_step_buf[in_warmup] -= 1

                # Per-step logging
                if _step_log is not None:
                    _spd_all = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    _cmd_all = env.command_generator.command
                    for _si in range(env.num_envs):
                        if int(takeover_steps[_si]) > 0:
                            _st = 4
                        elif int(low_q_run[_si]) > 0:
                            _st = 3
                        elif int(grace_left[_si]) > 0:
                            _st = 1
                        elif (SPEED_MATCH_FRAC > 0 or CMD_ACCEL_RATE > 0) and bool(in_accel[_si].item()):
                            _st = 2
                        else:
                            _st = 0
                        _step_log['global_step'].append(step_count)
                        _step_log['env_id'].append(_si)
                        _step_log['ep_id'].append(int(_ep_id_per_env[_si].item()))
                        _step_log['ep_step'].append(int(episode_step_buf[_si].item()))
                        _step_log['state'].append(_st)
                        _step_log['q_val'].append(float(q_vals[_si].item()))
                        _step_log['g_val'].append(float(g_vals[_si].item()))
                        _step_log['q_target'].append(float(q_target_vals[_si].item()))
                        _step_log['q_critic_minus_target'].append(
                            float((q_vals[_si] - q_target_vals[_si]).item()))
                        _step_log['target_g_min'].append(float(target_g_min[_si].item()))
                        _step_log['target_g_argmin'].append(int(target_g_argmin[_si].item()))
                        _step_log['target_v_stop_tail'].append(
                            float(target_v_stop_tail[_si].item()))
                        _step_log['target_tail_is_bottleneck'].append(
                            bool(target_tail_is_bottleneck[_si].item()))
                        _step_log['target_g_seq'].append(
                            target_g_seq[_si].detach().cpu().numpy().copy())
                        _step_log['cmd_vx'].append(float(_cmd_all[_si, 0].item()))
                        _step_log['cmd_vy'].append(float(_cmd_all[_si, 1].item()))
                        _step_log['cmd_yaw'].append(float(_cmd_all[_si, 2].item()))
                        _step_log['spd'].append(float(_spd_all[_si].item()))
                        _step_log['ood_in_zone'].append(bool(ood_in_zone[_si].item()))
                        _step_log['ood_current_low'].append(bool(ood_current_low[_si].item()))
                        _step_log['f1_alpha'].append(float(f1_alpha[_si].item()))

                # Fixed-step push
                if _do_manual_push:
                    push_now = (episode_step_buf == push_step_buf)
                    if push_now.any():
                        push_env_ids = torch.where(push_now)[0]
                        mdp.push_by_setting_velocity_body_frame(
                            env, push_env_ids,
                            {"x": (_push_vx_val, _push_vx_val),
                             "y": (0.0, 0.0), "z": (0.0, 0.0)},
                        )
                        push_step_buf[push_now] = -1
                        push_count_total += int(push_now.sum().item())
                        print(f"\033[96m[PUSH] Step {step_count}: envs {push_env_ids.tolist()}, "
                              f"vx={_push_vx_val:.1f} m/s\033[0m")

                # 7. Env dones
                if dones.any():
                    done_env_ids = torch.where(dones)[0]
                    was_in_takeover = (takeover_steps > 0) & dones

                    if isinstance(infos, dict) and "time_outs" in infos:
                        time_outs = infos["time_outs"]
                        if not isinstance(time_outs, torch.Tensor):
                            time_outs = torch.as_tensor(time_outs, device=dones.device)
                        is_fall_per_env = ~time_outs[dones].bool()
                    else:
                        is_fall_per_env = g_vals[dones] < 0

                    fall_count += is_fall_per_env.sum().item()
                    episode_count += done_env_ids.shape[0]
                    if _ep_id_per_env is not None:
                        _ep_id_per_env[done_env_ids] += 1
                    critic_filter.finalize_episode(done_env_ids, is_fall_per_env)

                    interrupted_tk = pending_reset[dones] & ~is_fall_per_env
                    if interrupted_tk.any():
                        interrupted_envs = done_env_ids[interrupted_tk]
                        metrics.record_stop(robot, interrupted_envs, env, use_snapshot=True)
                        takeover_ok_count += int(interrupted_envs.shape[0])

                    metrics.on_episode_done(done_env_ids, is_fall_per_env,
                                            robot=robot, env=env)

                    # ── [OOD] restore friction / actuator on env-done reset ────────
                    if _ood_enabled:
                        if _ood_saved_materials is not None and ood_current_low[dones].any():
                            _r = torch.where(dones & ood_current_low)[0].cpu()
                            restore_friction(robot, _r, _ood_saved_materials)
                        if OOD_ACTUATOR_WEAKNESS and _ood_weakness_active[dones].any():
                            restore_actuator_normal(
                                robot, torch.where(dones & _ood_weakness_active)[0],
                                _ood_saved_stiffness, _ood_saved_damping)
                            _ood_weakness_active[dones] = False
                    ood_current_low[dones] = False
                    ood_in_zone[dones]     = False
                    ood_ever_low[dones]    = False
                    ood_decay_alpha[dones] = 0.0
                    # ─────────────────────────────────────────────────────────────
                    # ── F1 restore on env-done reset ─────────────────────────────
                    if F1_ENABLE and f1_act_name is not None and f1_active[dones].any():
                        restore_f1_stiffness(robot, done_env_ids,
                                             f1_act_name, f1_joint_col, f1_kp_nominal)
                    f1_active[dones] = False
                    f1_alpha[dones]  = 0.0
                    # ─────────────────────────────────────────────────────────────

                    low_q_run[dones] = 0
                    episode_step_buf[dones] = 0
                    takeover_steps[dones] = 0
                    pending_reset[dones] = False
                    grace_left[dones] = RESET_GRACE_K
                    cmd_ramp[dones] = 0.0
                    cmd_decel_vx[dones] = 0.0
                    warmup_buf[dones] = WARMUP_STEPS
                    if _do_manual_push:
                        push_step_buf[dones] = torch.randint(
                            _push_lo, _push_hi + 1, (int(dones.sum().item()),),
                            dtype=torch.int32, device=env.device)
                    if args_cli.eval_terrain:
                        for _li, _ei in enumerate(done_env_ids.tolist()):
                            _arr = np.array(ep_q_buf[_ei])
                            if len(_arr):
                                (fall_q_trajs if bool(is_fall_per_env[_li].item())
                                 else safe_q_trajs).append(_arr)
                            ep_q_buf[_ei] = []

                    if was_in_takeover.any():
                        tk_ids = torch.where(was_in_takeover)[0].tolist()
                        print(f"\033[91m[DONE DURING TAKEOVER] Step {step_count}: "
                              f"envs {tk_ids}\033[0m")

                # 8. Script forced timeout
                if args_cli.max_steps is not None:
                    forced_timeout = (episode_step_buf >= args_cli.max_steps) & ~dones
                    if not args_cli.monitor_only:
                        forced_timeout &= ~(takeover_steps > 0)
                    if forced_timeout.any():
                        ft_ids = torch.where(forced_timeout)[0]
                        episode_count += ft_ids.shape[0]
                        if _ep_id_per_env is not None:
                            _ep_id_per_env[ft_ids] += 1
                        env.reset(ft_ids)
                        obs_dict = env.get_observations()
                        critic_filter.finalize_episode(ft_ids, is_fall=False)
                        is_fall_vec_ft = torch.zeros(ft_ids.shape[0], dtype=torch.bool,
                                                     device=env.device)
                        metrics.on_episode_done(ft_ids, is_fall_vec_ft)

                        # ── [OOD] restore friction / actuator on forced-timeout reset ─
                        if _ood_enabled:
                            if _ood_saved_materials is not None and ood_current_low[forced_timeout].any():
                                _r = torch.where(forced_timeout & ood_current_low)[0].cpu()
                                restore_friction(robot, _r, _ood_saved_materials)
                            if OOD_ACTUATOR_WEAKNESS and _ood_weakness_active[forced_timeout].any():
                                restore_actuator_normal(
                                    robot, torch.where(forced_timeout & _ood_weakness_active)[0],
                                    _ood_saved_stiffness, _ood_saved_damping)
                                _ood_weakness_active[forced_timeout] = False
                        ood_current_low[forced_timeout] = False
                        ood_in_zone[forced_timeout]     = False
                        ood_ever_low[forced_timeout]    = False
                        ood_decay_alpha[forced_timeout] = 0.0
                        # ─────────────────────────────────────────────────────────
                        # ── F1 restore on forced-timeout reset ───────────────────
                        if F1_ENABLE and f1_act_name is not None and f1_active[forced_timeout].any():
                            restore_f1_stiffness(robot, ft_ids,
                                                 f1_act_name, f1_joint_col, f1_kp_nominal)
                        f1_active[forced_timeout] = False
                        f1_alpha[forced_timeout]  = 0.0
                        # ─────────────────────────────────────────────────────────

                        low_q_run[forced_timeout] = 0
                        episode_step_buf[forced_timeout] = 0
                        takeover_steps[forced_timeout] = 0
                        pending_reset[forced_timeout] = False
                        grace_left[forced_timeout] = RESET_GRACE_K
                        cmd_ramp[forced_timeout] = 0.0
                        cmd_decel_vx[forced_timeout] = 0.0
                        warmup_buf[forced_timeout] = WARMUP_STEPS
                        if _do_manual_push:
                            push_step_buf[forced_timeout] = torch.randint(
                                _push_lo, _push_hi + 1, (int(forced_timeout.sum().item()),),
                                dtype=torch.int32, device=env.device)
                        if args_cli.eval_terrain:
                            for _i in ft_ids.tolist():
                                _arr = np.array(ep_q_buf[_i])
                                if len(_arr):
                                    safe_q_trajs.append(_arr)
                                ep_q_buf[_i] = []
                        print(f"\033[96m[TIMEOUT] Step {step_count}: envs {ft_ids.tolist()}\033[0m")

                # 9. Periodic status
                if step_count % args_cli.print_interval == 0:
                    fall_rate = fall_count / episode_count * 100 if episode_count > 0 else 0.0
                    is_tk  = takeover_steps > 0
                    is_alm = (low_q_run > 0) & ~is_tk
                    is_grc = (grace_left > 0) & ~is_tk & ~is_alm
                    n_tk, n_alm, n_grc = int(is_tk.sum()), int(is_alm.sum()), int(is_grc.sum())
                    is_acc = in_accel & ~is_tk & ~is_alm & ~is_grc
                    n_acc  = int(is_acc.sum())
                    n_run  = env.num_envs - n_tk - n_alm - n_grc - n_acc
                    qr_str = f"  Qr={q_rand_vals.mean():+.3f}" if args_cli.show_q_rand else ""
                    n_ood_low  = int(ood_current_low.sum().item())
                    n_ood_zone = int(ood_in_zone.sum().item())
                    ood_str = (f"  OOD_zone={n_ood_zone}/low={n_ood_low}" if _ood_enabled else "")
                    if SPEED_MATCH_FRAC > 0:
                        accel_str = f"  ACCEL={n_acc}"
                    elif CMD_ACCEL_RATE > 0:
                        accel_str = f"  RAMP={n_acc}"
                    else:
                        accel_str = ""
                    print(f"[Step {step_count:7d} | Ep {episode_count:5d}]  "
                          f"Q={q_vals.mean():+.4f}  "
                          f"Qtgt={q_target_vals.nanmean():+.4f}  "
                          f"g={g_vals.mean():+.4f}{qr_str}  "
                          f"RUNNING={n_run}  ALARM={n_alm}  TAKEOVER={n_tk}  GRACE={n_grc}"
                          f"{accel_str}{ood_str}  "
                          f"falls={fall_count}/{episode_count} ({fall_rate:.1f}%)  total_danger={danger_count}")
                    show_all = env.num_envs <= 8
                    for _i in range(env.num_envs):
                        _tk  = int(takeover_steps[_i].item())
                        _lqr = int(low_q_run[_i].item())
                        _gl  = int(grace_left[_i].item())
                        _acc = bool(in_accel[_i].item()) if (SPEED_MATCH_FRAC > 0 or CMD_ACCEL_RATE > 0) else False
                        if _tk > 0:
                            _lbl = f"\033[91m[TAKEOVER rem={_tk:3d}]\033[0m"
                        elif _lqr > 0:
                            _lbl = f"\033[93m[ALARM    run={_lqr:3d}]\033[0m"
                        elif _gl > 0:
                            _lbl = f"\033[94m[GRACE    left={_gl:3d}]\033[0m"
                        elif _acc:
                            if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                                _rv = float(cmd_ramp[_i].item())
                                _tv = float(_cmd_ramp_target[_i].item())
                                _lbl = f"\033[35m[RAMP  {_rv:.2f}→{_tv:.2f}m/s]\033[0m"
                            else:
                                _lbl = f"\033[35m[ACCEL    spd<cmd  ]\033[0m"
                        elif show_all:
                            _lbl = "\033[92m[RUNNING          ]\033[0m"
                        else:
                            continue
                        _ep  = int(episode_step_buf[_i].item())
                        _qi  = q_vals[_i].item()
                        _qti = q_target_vals[_i].item()
                        _gi  = g_vals[_i].item()
                        _spd = torch.linalg.vector_norm(robot.data.root_lin_vel_b[_i, :2]).item()
                        _cmd = env.command_generator.command[_i]
                        _need = f"{SPEED_MATCH_FRAC * _cmd[0].item():.2f}" if SPEED_MATCH_FRAC > 0 else ""
                        _spd_tag = f"  spd={_spd:.2f}(need≥{_need})" if _acc else f"  spd={_spd:.2f}"
                        _ood_tag = (" \033[95m[OOD-LOW]\033[0m" if ood_current_low[_i]
                                    else (" \033[35m[OOD-zone]\033[0m" if ood_in_zone[_i] else ""))
                        print(f"  env{_i:3d} {_lbl}  ep_step={_ep:4d}  "
                              f"Q={_qi:+.4f}  Qtgt={_qti:+.4f}  g={_gi:+.4f}  "
                              f"cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f}){_spd_tag}{_ood_tag}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as _exc:
        print(f"\n[ERROR] {_exc}")
        import traceback; traceback.print_exc()
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"  Saved video: {args_cli.video_path}")
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass

    # ==================== 结果 ====================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE  (WM Critic Filter + OOD Friction)")
    print("=" * 60)
    print(f"  Steps:      {step_count}")
    print(f"  Episodes:   {episode_count}")
    print(f"  Dangers:    {danger_count}")
    print(f"  Falls:      {fall_count}")
    if episode_count > 0:
        print(f"  Fall rate:  {fall_count/episode_count*100:.1f}%")
    critic_filter.print_stats(monitor_only=args_cli.monitor_only)

    # Tier-2
    metrics.print_summary(fall_count, takeover_ok_count, push_count_total)

    # ── OOD summary (variance + alarm latency) ───────────────────────────────
    if _ood_enabled:
        print(f"\n  --- OOD Friction Summary (WM Q) ---")
        _zone_str = (f"step≥{OOD_FRICTION_STEP}" if OOD_ZONE_MODE == 'step'
                     else f"dist_to_edge≤{OOD_ZONE_DIST}m")
        _mode_str = (f"random-flip  interval={OOD_FRICTION_INTERVAL}  p_low={OOD_FRICTION_PROB}"
                     if OOD_RANDOM_FLIP else "fixed-trigger")
        print(f"  zone={_zone_str}  mode={_mode_str}")
        print(f"  static={OOD_STATIC_MU}  dynamic={OOD_DYNAMIC_MU}")

        # Cohen's d: scale-free friction sensitivity (low → temporal integration)
        if _ood_zone_samples:
            _q_low  = np.array([q for q, lo in _ood_zone_samples if lo])
            _q_high = np.array([q for q, lo in _ood_zone_samples if not lo])
            print(f"\n  Q in OOD zone:  n_total={len(_ood_zone_samples)}  "
                  f"n_low={len(_q_low)}  n_high={len(_q_high)}")
            if len(_q_low) > 1 and len(_q_high) > 1:
                _mu_lo, _mu_hi = _q_low.mean(), _q_high.mean()
                _sp = np.sqrt(((_q_low.std()**2 + _q_high.std()**2) / 2))
                _cohens_d = abs(_mu_lo - _mu_hi) / (_sp + 1e-9)
                _r1_all = np.array([q for q, _ in _ood_zone_samples])
                _autocorr = float(np.corrcoef(_r1_all[:-1], _r1_all[1:])[0, 1]) if len(_r1_all) > 2 else float('nan')
                print(f"  mean_low={_mu_lo:.4f}  mean_high={_mu_hi:.4f}")
                print(f"  Cohen's d = {_cohens_d:.4f}  "
                      f"(low → WM integrates friction; compare with ABS V Cohen's d)")
                print(f"  lag-1 autocorr = {_autocorr:.4f}  "
                      f"(high → slow temporal change → RSSM smoothing)")
            else:
                print(f"  (insufficient low or high samples for Cohen's d)")
        else:
            print(f"  Q in OOD zone:  no samples (OOD zone never entered)")

        n_ood_alarm = len(ood_alarm_records)
        if n_ood_alarm > 0:
            latencies = [r['steps_since_ood'] for r in ood_alarm_records]
            print(f"\n  Alarms while in zone: n={n_ood_alarm}  "
                  f"latency mean={np.mean(latencies):.1f}  "
                  f"std={np.std(latencies):.1f}  min={np.min(latencies)}")
            print(f"  (latency = steps from zone entry to alarm; low std → WM integrates, not reactive)")
        else:
            print(f"\n  Alarms while in zone: 0")
    # ─────────────────────────────────────────────────────────────────────────

    # Eval terrain summary
    if args_cli.eval_terrain and episode_count > 0:
        total_env_steps = step_count * env.num_envs
        print(f"\n{'='*60}")
        print(f"EVAL SUMMARY  terrain={args_cli.eval_terrain}  "
              f"mode={'monitor-only' if args_cli.monitor_only else 'filter-active'}")
        print(f"{'='*60}")
        if args_cli.eval_terrain == 'cliff':
            if args_cli.monitor_only:
                timeout_count = episode_count - fall_count
                print(f"  Episodes  total:         {episode_count:5d}")
                print(f"  Falls     (collision):   {fall_count:5d}")
                print(f"  Timeouts  (survived):    {timeout_count:5d}")
                print(f"  Baseline collision rate: {fall_count/episode_count*100:.1f}%")
            else:
                valid = fall_count + takeover_ok_count
                ambiguous = episode_count - valid
                print(f"  Episodes  total:         {episode_count:5d}")
                print(f"  FN  (fell, Q missed):    {fall_count:5d}")
                print(f"  TP  (saved by filter):   {takeover_ok_count:5d}")
                print(f"  Ambiguous (timeout):     {ambiguous:5d}")
                if valid > 0:
                    print(f"  FNR = {fall_count}/{valid} = {fall_count/valid*100:.1f}%")
                    print(f"  TPR = {takeover_ok_count}/{valid} = {takeover_ok_count/valid*100:.1f}%")
                print(f"  Late Fail = {metrics.late_fail_count}/{valid} = "
                      f"{metrics.late_fail_count/max(valid,1)*100:.1f}%")
        elif args_cli.eval_terrain == 'flat':
            step_fpr = alarm_step_count / total_env_steps if total_env_steps > 0 else 0.0
            if args_cli.monitor_only:
                print(f"  Total env-steps:         {total_env_steps:7d}")
                print(f"  Alarm env-steps:         {alarm_step_count:7d}")
                print(f"  Step-level alarm rate:   {step_fpr*100:.2f}%")
            else:
                tn_count = episode_count - takeover_ok_count - fall_count
                ep_fpr = takeover_ok_count / episode_count if episode_count > 0 else 0.0
                print(f"  Episodes  total:         {episode_count:5d}")
                print(f"  FP  (false alarm):       {takeover_ok_count:5d}")
                print(f"  TN  (no alarm):          {tn_count:5d}")
                print(f"  Episode-level FPR:       {ep_fpr*100:.1f}%")
                print(f"  Step-level alarm rate:   {step_fpr*100:.2f}%")

        nf, ns = len(fall_q_trajs), len(safe_q_trajs)
        print(f"\n  Tau sweep: {nf} fall, {ns} safe  (H={EVAL_H} strict_f={EVAL_SAFE_STRICT_F})")
        if nf >= 5 and ns >= 5:
            sw = _sweep_tau_sim(fall_q_trajs, safe_q_trajs,
                                EVAL_H, EVAL_SAFE_STRICT_F, TAU_SWEEP)
            youden = [t - f for t, f in zip(sw['tpr'], sw['fpr_strict'])]
            yi = int(np.argmax(youden))
            tau_sim = float(TAU_SWEEP[yi])
            print(f"  tau_opt (Youden-strict) = {tau_sim:.3f}")
            print(f"  {'':12}  {'TPR':>6}  {'FPR_all':>8}  {'FPR_str':>8}  "
                  f"{'FNR':>6}  {'F1':>6}  {'Lead':>7}")
            print(f"  {'tau_opt':<12}  "
                  f"{sw['tpr'][yi]*100:>5.1f}%  "
                  f"{sw['fpr_all'][yi]*100:>7.1f}%  "
                  f"{sw['fpr_strict'][yi]*100:>7.1f}%  "
                  f"{sw['fnr'][yi]*100:>5.1f}%  "
                  f"{sw['f1'][yi]:>6.3f}  "
                  f"{sw['lead'][yi]:>6.1f}f")
        else:
            print(f"  Insufficient data (need ≥5+5; got {nf}+{ns}).")

    # Logs
    if args_cli.log_values and log_g:
        out = Path("./logs/ood_friction_eval_log.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 g_values=np.stack(log_g),
                 q_values=np.stack(log_q),
                 cmd_values=np.stack(log_cmd),
                 threshold=args_cli.safety_threshold,
                 num_envs=env.num_envs)
        print(f"  Saved log: {out}")

    if args_cli.save_stats:
        sp = Path(args_cli.save_stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        stats_dict = critic_filter.stats.summary()
        stats_dict.update(metrics.summary())
        stats_dict.update({
            'config': {
                'task':                  args_cli.task,
                'num_envs':              env.num_envs,
                'threshold':             args_cli.safety_threshold,
                'alarm_k':               ALARM_K,
                'cmd_accel_rate':        CMD_ACCEL_RATE,
                'cmd_decel_rate':        CMD_DECEL_RATE,
                'speed_match_frac':      SPEED_MATCH_FRAC,
                'premature_speed_frac':  PREMATURE_SPEED_FRAC,
                'monitor_only':          args_cli.monitor_only,
                'max_episode_length_s':  args_cli.max_episode_length_s,
                'max_steps_per_ep':      args_cli.max_steps,
                'max_episodes':          args_cli.max_episodes,
                'fixed_vx':              args_cli.fixed_vx,
                'fixed_vy':              args_cli.fixed_vy,
                'fixed_yaw':             args_cli.fixed_yaw,
                'platform_half_width':   args_cli.platform_half_width,
                'ood_friction_step':     OOD_FRICTION_STEP,
                'ood_static_friction':   OOD_STATIC_MU,
                'ood_dynamic_friction':  OOD_DYNAMIC_MU,
                'total_steps':           step_count,
                'episodes':              episode_count,
                'danger_events':         danger_count,
                'falls':                 fall_count,
                'takeover_ok':           takeover_ok_count,
                'late_fail':             metrics.late_fail_count,
                'fall_rate':             fall_count / episode_count if episode_count > 0 else None,
            }
        })
        _m = metrics.summary()
        _valid = fall_count + takeover_ok_count
        stats_dict['result'] = {
            'method':                    'critic_filter_ood_friction',
            'vx':                        args_cli.fixed_vx,
            'push_vx':                   _push_vx_val,
            'ood_friction_step':         OOD_FRICTION_STEP,
            'ood_static_friction':       OOD_STATIC_MU,
            'ood_dynamic_friction':      OOD_DYNAMIC_MU,
            'n_episodes':                episode_count,
            'n_fall':                    fall_count,
            'n_takeover_ok':             takeover_ok_count,
            'n_danger_events':           danger_count,
            'fall_rate':                 fall_count / episode_count if episode_count > 0 else None,
            'late_fail_count':           metrics.late_fail_count,
            'late_fail_rate':            metrics.late_fail_count / max(_valid, 1),
            'alarm_dist_mean_m':         _m.get('alarm_dist_mean_m'),
            'alarm_dist_std_m':          _m.get('alarm_dist_std_m'),
            'alarm_dist_min_m':          _m.get('alarm_dist_min_m'),
            'alarm_vel_mean_mps':        _m.get('alarm_vel_mean_mps'),
            'alarm_vel_std_mps':         _m.get('alarm_vel_std_mps'),
            'stop_margin_mean_m':        _m.get('stop_margin_mean_m'),
            'stop_margin_std_m':         _m.get('stop_margin_std_m'),
            'stop_margin_min_m':         _m.get('stop_margin_min_m'),
            'n_alarm_episodes':               _m.get('total_alarm_count', 0) - _m.get('premature_alarm_count', 0),
            'premature_alarm_count':          _m.get('premature_alarm_count', 0),
            'preterminated_alarm_count':      _m.get('preterminated_alarm_count', 0),
            'true_premature_alarm_count':     _m.get('true_premature_alarm_count', 0),
            'premature_alarm_rate':           _m.get('premature_alarm_count', 0) / max(episode_count, 1),
            'premature_alarm_fall_rate':      _m.get('premature_alarm_fall_rate'),
            'late_fail_past_edge_count':       metrics.late_fail_past_edge_count,
            'late_fail_past_edge_rate':        metrics.late_fail_past_edge_count / max(_valid, 1),
            'late_fail_on_platform_count':     metrics.late_fail_on_platform_count,
            'late_fail_on_platform_rate':      metrics.late_fail_on_platform_count / max(_valid, 1),
            'missed_fall_count':               metrics.missed_fall_count,
            'missed_fall_rate':                metrics.missed_fall_count / max(episode_count, 1),
            # OOD-specific
            'ood_zone_mode':                   OOD_ZONE_MODE,
            'ood_random_flip':                 OOD_RANDOM_FLIP,
            'ood_friction_interval':           OOD_FRICTION_INTERVAL if OOD_RANDOM_FLIP else None,
            'ood_friction_prob':               OOD_FRICTION_PROB if OOD_RANDOM_FLIP else None,
            # KEY METRIC: Cohen's d (scale-free friction sensitivity; low → temporal integration)
            **_compute_cohens_d(_ood_zone_samples, key='q'),
            'ood_alarm_count':                 len(ood_alarm_records),
            'ood_alarm_latency_mean_steps':    float(np.mean([r['steps_since_ood'] for r in ood_alarm_records]))
                                               if ood_alarm_records else None,
            'ood_alarm_latency_std_steps':     float(np.std([r['steps_since_ood'] for r in ood_alarm_records]))
                                               if ood_alarm_records else None,
            'ood_alarm_latency_min_steps':     int(np.min([r['steps_since_ood'] for r in ood_alarm_records]))
                                               if ood_alarm_records else None,
            'ood_alarm_records':               ood_alarm_records,
        }
        with open(sp, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"  Saved stats: {sp}")

    if _step_log is not None and args_cli.save_step_log and _step_log['global_step']:
        sp = Path(args_cli.save_step_log)
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp, **{k: np.array(v) for k, v in _step_log.items()})
        print(f"  Saved step log: {sp}  ({len(_step_log['global_step'])} records, "
              f"{len(set(_step_log['ep_id']))} episodes)")
        print("  State encoding: 0=RUNNING 1=GRACE 2=ACCEL 3=ALARM 4=TAKEOVER")
        print("  ood_triggered: bool, True = OOD low friction active for this env at this step")

    print("\nDone!")


if __name__ == "__main__":
    main()
    simulation_app.close()
