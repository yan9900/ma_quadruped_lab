#!/usr/bin/env python3
"""
play_with_critic_filter_eval.py — 扩展评估版 (WM Q(z,cmd))

在 play_with_critic_filter.py 基础上新增 Tier-2 指标:
  --fixed_vx / --fixed_vy / --fixed_yaw  : 固定 velocity command (受控实验)
  --platform_half_width                  : 平台半宽/半径 (m)，用于计算到 edge 的距离

新增指标（圆形平台模型）:
  distance_to_edge = sqrt((pos_x - edge_x)² + pos_y²) - platform_half_width
  
  alarm_dist_m     首次 alarm 时 robot 距平台边缘的距离 (m)
                   正值=边缘前预警，负值=已越过边缘
  alarm_vel_mps    首次 alarm 时 robot 前向速度 vx (m/s)
  stop_margin_m    takeover 完成后 robot 距平台边缘的距离 (m)
                   正值=安全停下，负值=停止时已过边缘
  late_fail        触发了 takeover 但仍坠落的 episode 数 / rate

使用方法：见文档 EVAL_SCRIPTS_USAGE.md 及 EXPERIMENT_DESIGN_DISCUSSION_0420.md
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

parser = argparse.ArgumentParser(description="Play with Critic Safety Filter (Eval)")
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
# ── [MOD 1] Cmd ramp argument ──────────────────────────────────────────────────
parser.add_argument("--cmd_accel_rate", type=float, default=0.0,
                    help="Cmd ramp: max rate (m/s²) at which cmd_vx grows from 0 toward target each episode. "
                         "0.0 = disabled / step command (default). "
                         "Must be < robot's min net acceleration (~2.1 m/s² across domain rand). "
                         "Recommended 1.5 m/s² (~30%% safety margin). "
                         "Eliminates OOD (speed<<cmd) false alarms without needing --speed_match_frac.")
parser.add_argument("--cmd_decel_rate", type=float, default=0.0,
                    help="Takeover decel ramp: max rate (m/s²) at which cmd_vx decreases toward 0 "
                         "during takeover. 0.0 = immediate stop (default). "
                         "Symmetric to --cmd_accel_rate. E.g. 2.0 m/s² for smooth braking.")
# ───────────────────────────────────────────────────────────────────────────────
parser.add_argument("--show_q_rand", action="store_true")
parser.add_argument("--adaptive_k", type=float, default=None)
parser.add_argument("--adaptive_burnin", type=int, default=200)
parser.add_argument("--eval_terrain", type=str, default=None, choices=["flat", "cliff"])
parser.add_argument("--eval_h", type=int, default=30)
parser.add_argument("--eval_safe_strict_f", type=int, default=None)

# ── 新增：受控速度实验 ──
parser.add_argument("--fixed_vx", type=float, default=None,
                    help="Fix command vx to this value each step (controlled experiment). "
                         "None = use env's random command generator (default).")
parser.add_argument("--fixed_vy", type=float, default=0.0,
                    help="Fix command vy when --fixed_vx is set (default 0.0).")
parser.add_argument("--fixed_yaw", type=float, default=0.0,
                    help="Fix command yaw when --fixed_vx is set (default 0.0).")
parser.add_argument("--platform_half_width", type=float, default=5.0,
                    help="Half-width of cliff platform in meters, used to compute edge_x = "
                         "env_origin_x + platform_half_width. Default 5.0m (matches "
                         "CLIFF_EVALUATION_TERRAINS_CFG platform_width=10.0m). "
                         "Adjust to match CLIFF_EVAL_TERRAINS_CFG tile size.")
parser.add_argument("--premature_speed_frac", type=float, default=0.0,
                    help="Fraction of |cmd_vx| the robot must reach before an alarm counts toward "
                         "alarm_dist/stop_margin metrics. 0.0 = disabled (all alarms count). "
                         "E.g. 0.85: exclude alarms fired before robot reaches 85%% of commanded speed. "
                         "Excluded alarms are still logged with is_premature=True flag in alarm_records.")
# ── Push 实验控制 ──
parser.add_argument("--push_vx", type=float, default=None,
                    help="If set, override push_robot event vx with this impulse magnitude (m/s). "
                         "None = use env config default (or disable push if --no_push is set).")
parser.add_argument("--no_push", action="store_true",
                    help="Disable push_robot domain rand event entirely. "
                         "Useful for clean stop-distance measurement.")
parser.add_argument("--push_frac_low", type=float, default=0.7,
                    help="Push timing lower bound, fraction of T=phw/(vx*dt). Default 0.7.")
parser.add_argument("--push_frac_high", type=float, default=0.85,
                    help="Push timing upper bound, fraction of T=phw/(vx*dt). Default 0.85.")

# Video recording 参数
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_path", type=str, default="./videos/critic_eval.mp4")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_follow_offset", type=float, nargs=3, default=[-1.5, 0.0, 0.8],
                    metavar=("DX", "DY", "DZ"))
parser.add_argument("--video_env_idx", type=int, default=0)
parser.add_argument("--video_no_follow", action="store_true")

# ── Per-step logging ──
parser.add_argument("--log_step_data", action="store_true",
                    help="Log per-step per-env state data (Q, g, cmd, speed, state) for trajectory analysis.")
parser.add_argument("--save_step_log", type=str, default=None,
                    help="Path to save per-step log as .npz (requires --log_step_data).")

# ── RSSM warmup experiment ──
parser.add_argument("--warmup_steps", type=int, default=0,
                    help="Per-episode RSSM warmup: stand still for this many steps at episode start "
                         "BEFORE beginning the fixed_vx approach. During warmup, cmd=0, alarm is "
                         "suppressed, and episode_step_buf is NOT incremented. "
                         "Use to test RSSM temporal-OOD hypothesis: --warmup_steps 500 on a 5x5 "
                         "platform should reproduce the late-alarm pattern of the 10x10 platform. "
                         "0 = disabled (default).")

import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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


# ── Helpers (同 play_with_critic_filter.py) ──────────────────────────────────
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
    )
    print("=" * 60 + "\n")
    return filt


# =====================================================================
#               Tier-2 Episode Metrics Tracker
# =====================================================================
class EpisodeMetrics:
    """
    Track per-episode Tier-2 metrics: alarm_dist, alarm_vel, stop_margin, late_fail.

    地形建模：将平台近似为圆形（中心在 (edge_x, 0), 半径 = platform_half_width）。
    
    distance_to_edge = sqrt((pos_x - edge_x)^2 + pos_y^2) - platform_half_width
        正值 = robot 在平台内侧（安全）
        负值 = robot 已越过平台边缘（危险/坠落）

    alarm_dist_m      首次 alarm 时 robot 距平台边缘的距离 (m)
    stop_margin_m     takeover 完成后 robot 距平台边缘的距离 (m)
    alarm_vel_mps     首次 alarm 时 robot 前向速度 vx (m/s)
    late_fail         takeover fired but episode ended in fall
    """

    def __init__(self, num_envs: int, platform_half_width: float, device):
        self.num_envs = num_envs
        self.phw = platform_half_width  # platform radius
        self.device = device

        # Per-env state (reset each episode)
        nan = float('nan')
        self.alarm_pos_xy = torch.full((num_envs, 2), nan, device=device)  # [x, y] at alarm
        self.alarm_origin_xy = torch.full((num_envs, 2), nan, device=device)  # terrain origin at alarm
        self.alarm_vel_x = torch.full((num_envs,), nan, device=device)
        self.had_takeover = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alarm_is_premature = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alarm_is_ramp_phase = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Pre-step position/origin snapshot: updated every step BEFORE env.step() to capture
        # the last valid state. Used in on_episode_done / record_stop(interrupted) because
        # IsaacLab auto-resets terminated envs inside env.step(), making post-step positions AND
        # terrain origins stale (curriculum may move env to a new tile after reset).
        self.pre_step_pos_xy = torch.full((num_envs, 2), nan, device=device)
        self.pre_step_origin_xy = torch.full((num_envs, 2), nan, device=device)

        # Aggregate records
        self.alarm_records: list = []   # alarm_dist_m, alarm_vel_mps, is_fall, had_takeover, is_premature, fall_dist_m
        self.stop_records: list = []    # stop_margin_m, alarm_dist_m, is_premature
        self.late_fail_count = 0
        self.late_fail_past_edge_count = 0    # had takeover, fell past edge (alarm too late)
        self.late_fail_on_platform_count = 0  # had takeover, fell on platform (braking/body collision)
        self.missed_fall_count = 0            # no takeover fired, robot fell (filter missed)

    def update_pre_step(self, robot, env):
        """Snapshot robot XY positions and terrain origins BEFORE env.step().

        Must be called every loop iteration before env.step() so that on_episode_done
        and record_stop(use_snapshot=True) see the pre-reset position and origin when
        IsaacLab auto-resets fallen/timed-out envs inside env.step() (curriculum may
        move an env to a new terrain tile, changing env_origins after the reset).
        """
        self.pre_step_pos_xy[:] = robot.data.root_pos_w[:, :2].detach()
        try:
            origins = env.scene.terrain.env_origins
            self.pre_step_origin_xy[:, 0] = origins[:, 0]
            self.pre_step_origin_xy[:, 1] = origins[:, 1]
        except Exception:
            pass  # stays NaN → fall_dist will be NaN

    def try_load_terrain_edge(self, env):
        """Validation print only — origins are now captured dynamically per alarm."""
        try:
            origins = env.scene.terrain.env_origins  # (num_envs, 3)
            print(f"[INFO][Metrics] Circular platform model: radius={self.phw}m  "
                  f"terrain origins x ∈ ({origins[:, 0].min():.1f}, {origins[:, 0].max():.1f}), "
                  f"y ∈ ({origins[:, 1].min():.1f}, {origins[:, 1].max():.1f})")
            print(f"[INFO][Metrics] NOTE: terrain origin is captured per-alarm (not fixed at init).")
        except Exception as e:
            print(f"[WARN][Metrics] Terrain env_origins unavailable ({e}). "
                  f"distance metrics will be NaN.")

    def record_alarm(self, robot, alarm_now: torch.Tensor, env,
                     is_premature: torch.Tensor, is_ramp_phase: torch.Tensor):
        """Record robot state at FIRST alarm per episode. Call every step.

        is_premature:  bool tensor (num_envs,) — True if robot speed < premature_speed_frac * cmd_vx
                       at alarm time. Premature alarms are logged but excluded from aggregate stats.
        is_ramp_phase: bool tensor (num_envs,) — True if cmd_ramp < target_vx at alarm time
                       (command still ramping up after reset). Used to sub-classify premature alarms:
                         is_premature &  is_ramp_phase → pre-terminated (reset-induced low speed)
                         is_premature & ~is_ramp_phase → true premature (cmd at target, robot slow)
        """
        new_alarm = alarm_now & torch.isnan(self.alarm_pos_xy[:, 0])
        if new_alarm.any():
            self.alarm_pos_xy[new_alarm, 0] = robot.data.root_pos_w[new_alarm, 0]
            self.alarm_pos_xy[new_alarm, 1] = robot.data.root_pos_w[new_alarm, 1]
            self.alarm_vel_x[new_alarm] = robot.data.root_lin_vel_b[new_alarm, 0]
            self.alarm_is_premature[new_alarm] = is_premature[new_alarm]
            self.alarm_is_ramp_phase[new_alarm] = is_ramp_phase[new_alarm]
            # Capture terrain origin NOW (before episode resets)
            try:
                origins = env.scene.terrain.env_origins  # (num_envs, 3)
                self.alarm_origin_xy[new_alarm, 0] = origins[new_alarm, 0]
                self.alarm_origin_xy[new_alarm, 1] = origins[new_alarm, 1]
            except Exception:
                pass  # stays NaN → distance will be NaN

    def mark_takeover(self, danger: torch.Tensor):
        self.had_takeover[danger] = True

    def _compute_distance_to_edge(self, pos_xy: torch.Tensor, center_xy: torch.Tensor) -> float:
        """
        Compute signed distance from robot to nearest edge of a SQUARE platform.

        Uses L∞ (Chebyshev) norm: distance = phw - max(|dx|, |dy|)
          positive = robot is inside the square (safe)
          zero     = robot is exactly on an edge
          negative = robot is outside the square (even diagonally)
        """
        if torch.isnan(center_xy[0]):
            return float('nan')
        dx = abs(pos_xy[0].item() - center_xy[0].item())
        dy = abs(pos_xy[1].item() - center_xy[1].item())
        return self.phw - max(dx, dy)

    def record_stop(self, robot, done_envs: torch.Tensor, env, use_snapshot: bool = False):
        """Record stop margin when takeover completes. Call on tk_done.

        use_snapshot=True: read position from self.pre_step_pos_xy (snapshot taken before
            env.step()) — required for the interrupted-takeover path where env.step() has
            already auto-reset the environment before this is called.
        use_snapshot=False (default): read live robot.data.root_pos_w — correct when
            called before env.reset() (normal tk_done completion path).
        """
        try:
            origins = env.scene.terrain.env_origins  # current, still valid during episode
        except Exception:
            origins = None
        for i in done_envs.tolist():
            if origins is not None:
                center_xy = origins[i, :2]
            else:
                center_xy = torch.full((2,), float('nan'), device=self.device)
            # Use snapshot if caller indicates env.step() has already run (interrupted path)
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
        """Call on env dones and forced timeouts.

        Uses self.pre_step_pos_xy and self.pre_step_origin_xy (snapshots taken before
        env.step()) for fall-type classification. Both must be pre-step snapshots because
        IsaacLab auto-resets terminated envs inside env.step() — terrain curriculum may
        move an env to a new tile, changing env_origins after the reset and causing
        fall_dist to be computed against the wrong platform center.
        robot and env are kept for signature compatibility but are NOT read here.
        """
        for _li, _ei in enumerate(env_ids.tolist()):
            is_fall = bool(is_fall_vec[_li].item())
            had_tk = bool(self.had_takeover[_ei].item())

            # Classify fall type using PRE-STEP position AND PRE-STEP origin (both
            # snapshotted before env.step() in update_pre_step). Using the live
            # env.scene.terrain.env_origins here would be wrong: curriculum resets can
            # change the origin for a just-fallen env before this function is called.
            fall_dist = float('nan')
            if is_fall:
                pos_xy = self.pre_step_pos_xy[_ei]
                origin_xy = self.pre_step_origin_xy[_ei]
                fall_dist = self._compute_distance_to_edge(pos_xy, origin_xy)

            if had_tk and is_fall:
                self.late_fail_count += 1
                if not np.isnan(fall_dist):
                    if fall_dist > 0:
                        self.late_fail_on_platform_count += 1  # braking/body collision
                    else:
                        self.late_fail_past_edge_count += 1    # alarm too late
            elif is_fall and not had_tk:
                # Only count as missed cliff-fall if robot was already past the edge
                # (fall_dist <= 0). On-platform body-contact terminations (fall_dist > 0,
                # e.g. during post-reset acceleration) are excluded.
                if np.isnan(fall_dist) or fall_dist <= 0:
                    self.missed_fall_count += 1

            a_xy = self.alarm_pos_xy[_ei]
            a_origin = self.alarm_origin_xy[_ei]
            if not torch.isnan(a_xy[0]):
                # Use origin captured at alarm time (not current — may be post-reset)
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
                    'fall_dist_m': fall_dist,  # distance to edge at fall time (NaN if no fall)
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
            out['preterminated_alarm_count']  = n_preterm   # premature & cmd still ramping
            out['true_premature_alarm_count'] = n_true_prem  # premature & cmd at target
            out['total_alarm_count']          = len(self.alarm_records)
            prem_records = [r for r in self.alarm_records if r.get('is_premature', False)]
            if prem_records:
                prem_falls = sum(1 for r in prem_records if r.get('is_fall', False))
                out['premature_alarm_fall_rate'] = prem_falls / len(prem_records)
            else:
                out['premature_alarm_fall_rate'] = None
            # Stats computed on mature (non-premature) records only
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
            print(f"  Alarm Velocity: mean={s['alarm_vel_mean_mps']:.2f}m/s  "
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

    # ── 强制启用摄像头 ────────────────────────────────────────────────
    # go2_flat 默认 enable_camera=False，CriticSafetyFilter 需要 front_camera 深度图。
    # 从 go2_data_collection cfg 整体复制相机参数（spawn / offset / 分辨率等），
    # 然后用 step_dt 覆盖 update_period，与仿真步长对齐。
    _dc_cfg, _ = task_registry.get_cfgs("go2_data_collection")
    env_cfg.scene.camera = _dc_cfg.scene.camera

    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt
    env_cfg.scene.env_spacing = 10
    env_cfg.scene.max_episode_length_s = args_cli.max_episode_length_s

    # ── Push 控制：禁用 domain rand interval 事件，改为固定步数 one-shot push ──
    # 实验设计：push 在 episode 进行到 [frac_low, frac_high]*T_edge 时触发，filter 有时间响应
    import mdp
    if hasattr(env_cfg.domain_rand.events, 'push_robot'):
        env_cfg.domain_rand.events.push_robot = None  # 禁用 interval push

    if args_cli.no_push or args_cli.push_vx is None:
        _push_vx_val = None
        print("[INFO] Push DISABLED" + (" (--no_push)" if args_cli.no_push else " (no --push_vx)"))
    else:
        _push_vx_val = args_cli.push_vx
        print(f"[INFO] Fixed-step push: vx_impulse={_push_vx_val:.1f} m/s (body_frame)  "
              f"fires once per episode at step ∈ [{args_cli.push_frac_low:.2f}, {args_cli.push_frac_high:.2f}] * T_edge")

    # ── 固定命令采样范围：必须在 env 创建之前设置 ──────────────────────────
    if args_cli.fixed_vx is not None:
        env_cfg.commands.ranges.lin_vel_x = (args_cli.fixed_vx, args_cli.fixed_vx)
        env_cfg.commands.ranges.lin_vel_y = (args_cli.fixed_vy, args_cli.fixed_vy)
        env_cfg.commands.ranges.ang_vel_z = (args_cli.fixed_yaw, args_cli.fixed_yaw)

    # ── 自动扩展 max_steps / max_episode_length_s 以保证 filter 有时间响应 ────
    # 问题：当 T_edge = phw / (vx * dt) ≈ max_steps 时，env 内部 timeout 会先于
    #       filter takeover 完成而触发 auto-reset，导致 n_takeover_ok=0。
    # 修复：自动将 max_steps 和 max_episode_length_s 扩展到
    #       T_edge + takeover_duration + 100 步缓冲，并打印提示。
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
    # ── [MOD 2] Cmd ramp state ─────────────────────────────────────────────────
    CMD_ACCEL_RATE = float(args_cli.cmd_accel_rate)   # m/s²; 0.0 = disabled
    cmd_ramp = torch.zeros(env.num_envs, device=env.device)
    CMD_DECEL_RATE = float(args_cli.cmd_decel_rate)   # m/s²; 0.0 = immediate stop
    cmd_decel_vx = torch.zeros(env.num_envs, device=env.device)  # signed, init at takeover onset
    PREMATURE_SPEED_FRAC = float(args_cli.premature_speed_frac)  # 0.0 = disabled
    # ── Fixed-step push 状态 ──
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

    # ── Eval Q buffers ──
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

    # ── RSSM warmup state ──
    WARMUP_STEPS = max(0, int(args_cli.warmup_steps))
    warmup_buf = torch.full((env.num_envs,), fill_value=WARMUP_STEPS,
                            dtype=torch.int32, device=env.device)
    if WARMUP_STEPS > 0:
        print(f"[INFO] RSSM warmup experiment: {WARMUP_STEPS} steps of cmd=0 standing "
              f"before each episode's fixed_vx approach. "
              f"Alarm suppressed during warmup.")

    # ── Tier-2 metrics tracker ──
    metrics = EpisodeMetrics(env.num_envs, args_cli.platform_half_width, env.device)
    metrics.try_load_terrain_edge(env)

    # ── Per-step log init ──
    # state encoding: 0=RUNNING 1=GRACE 2=ACCEL 3=ALARM 4=TAKEOVER
    if args_cli.log_step_data:
        _step_log = {k: [] for k in [
            'global_step', 'env_id', 'ep_id', 'ep_step',
            'state', 'q_val', 'g_val',
            'cmd_vx', 'cmd_vy', 'cmd_yaw', 'spd',
        ]}
        _ep_id_per_env = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        _step_log = None
        _ep_id_per_env = None

    log_g, log_q, log_cmd = [], [], []

    env_timeout_steps = int(np.ceil(args_cli.max_episode_length_s / step_dt))
    print("\n[INFO] Starting evaluation " +
          ("(monitor-only)" if args_cli.monitor_only else "(filter active)") + "...")
    # ── [MOD 3] startup print ────────────────────────────────────────────────────
    _ramp_info = (f"cmd_ramp={CMD_ACCEL_RATE:.1f}m/s²" if CMD_ACCEL_RATE > 0 else "cmd_ramp=disabled")
    print(f"[INFO] Alarm rule: {ALARM_K} consecutive frames with Q < {args_cli.safety_threshold:.4f}  "
          f"reset_grace={RESET_GRACE_K}f  {_ramp_info}")
    # ─────────────────────────────────────────────────────────────────────────────
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
                # ── RSSM warmup phase: zero cmd for envs still in warmup ──
                in_warmup = warmup_buf > 0
                # ── 受控速度覆盖 (fixed_vx 实验) ──
                if args_cli.fixed_vx is not None:
                    env.command_generator.command[:, 0] = args_cli.fixed_vx
                    env.command_generator.command[:, 1] = args_cli.fixed_vy
                    env.command_generator.command[:, 2] = args_cli.fixed_yaw
                # Override warmup envs to standing still AFTER fixed_vx so alarm suppression
                # and RSSM accumulate the zero-cmd context we want to test.
                if in_warmup.any():
                    env.command_generator.command[in_warmup, 0] = 0.0
                    env.command_generator.command[in_warmup, 1] = 0.0
                    env.command_generator.command[in_warmup, 2] = 0.0

                # 1. PPO action
                actions_raw = policy(obs_dict)

                # 2. Current command (target — may have been set by fixed_vx above)
                cmd_current = env.command_generator.command[:, :3].clone()

                # ── [MOD 4] Cmd ramp ──────────────────────────────────────────────────
                # Must come AFTER fixed_vx override (which sets the target) and AFTER
                # reading cmd_current (which captures that target).  The ramp then
                # overwrites env.command_generator and cmd_current with the ramped value
                # so that the policy obs next step and the critic this step both see it.
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
                # ─────────────────────────────────────────────────────────────────────

                # 3. Critic filter
                cmd_out, is_safe, q_vals, g_vals, q_rand_vals = critic_filter.filter_cmd(
                    obs_dict, cmd_current,
                    apply_filter=not args_cli.monitor_only,
                    compute_q_rand=args_cli.show_q_rand,
                )

                # 4. K-consecutive alarm
                unsafe_now = ~is_safe
                in_grace = grace_left > 0
                # ── [MOD 5] in_accel split: suppress (K-filter) vs display ────────────
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
                # ─────────────────────────────────────────────────────────────────────
                low_q_run = torch.where(unsafe_now & ~in_grace & ~in_accel_suppress, low_q_run + 1,
                                        torch.zeros_like(low_q_run))
                alarm_now = low_q_run >= ALARM_K
                # ── Warmup suppression: never trigger alarm during warmup phase ──
                if WARMUP_STEPS > 0:
                    alarm_now = alarm_now & ~in_warmup
                    low_q_run = torch.where(in_warmup, torch.zeros_like(low_q_run), low_q_run)
                alarm_step_count += int(alarm_now.sum().item())
                # grace_left decremented after env.step() below — counts sim steps not loop iters

                # ── 记录首次 alarm 的位置和速度 ──
                if PREMATURE_SPEED_FRAC > 0:
                    _spd_prem   = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    _cmdvx_prem = env.command_generator.command[:, 0].abs()
                    _alarm_premature = (_spd_prem < PREMATURE_SPEED_FRAC * _cmdvx_prem) & (_cmdvx_prem > 0.3)
                else:
                    _alarm_premature = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                # Ramp-phase flag: cmd still below target → alarm is "pre-terminated" (reset-induced),
                # not a true premature. Only meaningful when CMD_ACCEL_RATE > 0.
                if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                    _alarm_ramp_phase = (cmd_ramp < _cmd_ramp_target - 0.05) & (_cmd_ramp_target > 0.3)
                else:
                    _alarm_ramp_phase = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                metrics.record_alarm(robot, alarm_now, env, _alarm_premature, _alarm_ramp_phase)

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
                            cmd_decel_vx[danger] = cmd_current[danger, 0]  # signed; ramp toward 0
                        d_envs = torch.where(danger)[0].tolist()
                        d_q = q_vals[danger].tolist()
                        d_g = g_vals[danger].tolist()
                        d_cmd = cmd_current[danger].tolist()
                        danger_count += danger.sum().item()
                        metrics.mark_takeover(danger)   # ← Tier-2: mark takeover
                        print(f"\033[93m[WARN] Step {step_count}: DANGER in envs {d_envs}\033[0m")
                        for i, e in enumerate(d_envs):
                            qr_diag = (f"  Q_rand={q_rand_vals[danger][i].item():.3f}"
                                       if args_cli.show_q_rand else "")
                            print(f"  env{e}: Q={d_q[i]:.3f}{qr_diag}  "
                                  f"g={d_g[i]:.3f}  "
                                  f"cmd=({d_cmd[i][0]:.2f},{d_cmd[i][1]:.2f},{d_cmd[i][2]:.2f})")
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
                        # ── Tier-2: record stop margin before reset ──
                        metrics.record_stop(robot, done_envs, env)
                        env.reset(done_envs)
                        obs_dict = env.get_observations()
                        pending_reset[tk_done] = False
                        low_q_run[tk_done] = 0
                        episode_step_buf[tk_done] = 0
                        grace_left[tk_done] = RESET_GRACE_K
                        cmd_ramp[tk_done] = 0.0               # [MOD 6a] restart ramp
                        cmd_decel_vx[tk_done] = 0.0
                        warmup_buf[tk_done] = WARMUP_STEPS    # restart warmup
                        if _do_manual_push:
                            push_step_buf[tk_done] = torch.randint(
                                _push_lo, _push_hi + 1, (int(tk_done.sum().item()),),
                                dtype=torch.int32, device=env.device)
                        critic_filter.finalize_episode(done_envs, is_fall=False)
                        episode_count += tk_done.sum().item()
                        takeover_ok_count += tk_done.sum().item()
                        if _ep_id_per_env is not None:
                            _ep_id_per_env[done_envs] += 1
                        # ── Tier-2: on_episode_done for takeover-save envs ──
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

                # Snapshot position BEFORE env.step() — IsaacLab auto-resets fallen envs
                # inside step(), making post-step positions stale for terminated envs.
                metrics.update_pre_step(robot, env)

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
                # ── Warmup: decrement warmup counter; freeze episode_step_buf while in warmup ──
                if WARMUP_STEPS > 0 and in_warmup.any():
                    warmup_buf = torch.clamp(warmup_buf - 1, min=0)
                    # Undo the episode_step_buf increment for envs still in warmup phase
                    # (episode steps are counted from the moment the robot starts moving)
                    episode_step_buf[in_warmup] -= 1

                # ── Per-step logging ──
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
                            _st = 2   # [MOD 7] ACCEL state also triggered by cmd ramp
                        else:
                            _st = 0
                        _step_log['global_step'].append(step_count)
                        _step_log['env_id'].append(_si)
                        _step_log['ep_id'].append(int(_ep_id_per_env[_si].item()))
                        _step_log['ep_step'].append(int(episode_step_buf[_si].item()))
                        _step_log['state'].append(_st)
                        _step_log['q_val'].append(float(q_vals[_si].item()))
                        _step_log['g_val'].append(float(g_vals[_si].item()))
                        _step_log['cmd_vx'].append(float(_cmd_all[_si, 0].item()))
                        _step_log['cmd_vy'].append(float(_cmd_all[_si, 1].item()))
                        _step_log['cmd_yaw'].append(float(_cmd_all[_si, 2].item()))
                        _step_log['spd'].append(float(_spd_all[_si].item()))

                # ── Fixed-step push ──
                if _do_manual_push:
                    push_now = (episode_step_buf == push_step_buf)
                    if push_now.any():
                        push_env_ids = torch.where(push_now)[0]
                        mdp.push_by_setting_velocity_body_frame(
                            env, push_env_ids,
                            {"x": (_push_vx_val, _push_vx_val),
                             "y": (0.0, 0.0), "z": (0.0, 0.0)},
                        )
                        push_step_buf[push_now] = -1  # mark as fired
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

                    # ── Tier-2: record stop for envs whose takeover was interrupted by env timeout ──
                    # env.step() already auto-reset these envs; use pre-step snapshot for position.
                    # Also count as takeover_ok: robot was already braking; env timeout ≠ failure.
                    interrupted_tk = pending_reset[dones] & ~is_fall_per_env
                    if interrupted_tk.any():
                        interrupted_envs = done_env_ids[interrupted_tk]
                        metrics.record_stop(robot, interrupted_envs, env, use_snapshot=True)
                        takeover_ok_count += int(interrupted_envs.shape[0])

                    # ── Tier-2: on_episode_done ──
                    metrics.on_episode_done(done_env_ids, is_fall_per_env,
                                            robot=robot, env=env)

                    low_q_run[dones] = 0
                    episode_step_buf[dones] = 0
                    takeover_steps[dones] = 0
                    pending_reset[dones] = False
                    grace_left[dones] = RESET_GRACE_K
                    cmd_ramp[dones] = 0.0               # [MOD 6b] restart ramp
                    cmd_decel_vx[dones] = 0.0
                    warmup_buf[dones] = WARMUP_STEPS     # restart warmup
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
                        low_q_run[forced_timeout] = 0
                        episode_step_buf[forced_timeout] = 0
                        takeover_steps[forced_timeout] = 0
                        pending_reset[forced_timeout] = False
                        grace_left[forced_timeout] = RESET_GRACE_K
                        cmd_ramp[forced_timeout] = 0.0               # [MOD 6c] restart ramp
                        cmd_decel_vx[forced_timeout] = 0.0
                        warmup_buf[forced_timeout] = WARMUP_STEPS   # restart warmup
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
                    # ── [MOD 8a] accel_str covers both speed-frac and cmd-ramp modes ──
                    if SPEED_MATCH_FRAC > 0:
                        accel_str = f"  ACCEL={n_acc}"
                    elif CMD_ACCEL_RATE > 0:
                        accel_str = f"  RAMP={n_acc}"
                    else:
                        accel_str = ""
                    # ─────────────────────────────────────────────────────────────────
                    print(f"[Step {step_count:7d} | Ep {episode_count:5d}]  "
                          f"Q={q_vals.mean():+.4f}  g={g_vals.mean():+.4f}{qr_str}  "
                          f"RUNNING={n_run}  ALARM={n_alm}  TAKEOVER={n_tk}  GRACE={n_grc}{accel_str}  "
                          f"falls={fall_count}/{episode_count} ({fall_rate:.1f}%)  total_danger={danger_count}")
                    show_all = env.num_envs <= 8
                    for _i in range(env.num_envs):
                        _tk  = int(takeover_steps[_i].item())
                        _lqr = int(low_q_run[_i].item())
                        _gl  = int(grace_left[_i].item())
                        # ── [MOD 8b] per-env: _acc and labels cover both modes ───────
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
                        # ─────────────────────────────────────────────────────────────
                        _ep  = int(episode_step_buf[_i].item())
                        _qi  = q_vals[_i].item()
                        _gi  = g_vals[_i].item()
                        _spd = torch.linalg.vector_norm(robot.data.root_lin_vel_b[_i, :2]).item()
                        _cmd = env.command_generator.command[_i]
                        _need = f"{SPEED_MATCH_FRAC * _cmd[0].item():.2f}" if SPEED_MATCH_FRAC > 0 else ""
                        _spd_tag = f"  spd={_spd:.2f}(need≥{_need})" if _acc else f"  spd={_spd:.2f}"
                        print(f"  env{_i:3d} {_lbl}  ep_step={_ep:4d}  "
                              f"Q={_qi:+.4f}  g={_gi:+.4f}  "
                              f"cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f}){_spd_tag}")

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
    print("EVALUATION COMPLETE  (WM Critic Filter)")
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

    # Eval terrain summary (同原版)
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
        out = Path("./logs/critic_filter_eval_log.npz")
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
            'method':                    'critic_filter',
            'vx':                        args_cli.fixed_vx,
            'push_vx':                   _push_vx_val,
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

    print("\nDone!")


if __name__ == "__main__":
    main()
    simulation_app.close()
