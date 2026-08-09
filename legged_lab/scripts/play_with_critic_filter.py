#!/usr/bin/env python3
"""
Go2 评估 + Q(z, cmd) Critic Safety Filter

基于 play_with_safety_v2.py 的升级版:
  - g(x): 单帧安全分类器 → "现在摔了没？"
  + Q(z, cmd): 未来安全价值 → "按这个 cmd 走会不会摔？" (proactive, +7.7f lead)

使用方法:
    # Monitor-only (仅记录 Q 值，不干预)
    python scripts/play_with_critic_filter.py \\
        --task go2_flat --enable_cameras --headless \\
        --num_envs 4 --max_steps 2000 --monitor_only

    # Filter 模式 (Q < threshold → 停下来)
    python scripts/play_with_critic_filter.py \\
        --task go2_flat --enable_cameras --headless \\
        --num_envs 4 --max_steps 2000

    # 自定义阈值
    python scripts/play_with_critic_filter.py \\
        --task go2_flat --enable_cameras --headless \\
        --safety_threshold -0.1 --max_steps 5000
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

parser = argparse.ArgumentParser(description="Play with Critic Safety Filter")
parser.add_argument("--task", type=str, default="go2_flat", help="Task name for LeggedLab policy")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=150,
                    help="Script-level per-episode step limit (truncates before env timeout). "
                         "None = only env's max_episode_length_s applies.")
parser.add_argument("--max_episodes", type=int, default=500, help="Auto-exit after N total episodes")
parser.add_argument("--max_episode_length_s", type=float, default=10.0,
                    help="Env-level episode timeout in seconds (default 10.0 = 500 steps @ 50Hz)")

# Critic Filter 参数
parser.add_argument("--safety_threshold", type=float, default=0.0,
                    help="Q < threshold → intervene (default 0.0)")
parser.add_argument("--monitor_only", action="store_true",
                    help="Only log Q values, don't filter commands")
parser.add_argument("--wm_path", type=str, default=None,
                    help="World model checkpoint (default: best known)")
parser.add_argument("--critic_path", type=str, default=None,
                    help="DDPG Critic checkpoint (default: nd_ep20)")
parser.add_argument("--gx_tau", type=float, default=None,
                    help="Raw-space g(x) threshold tau (default: from CriticSafetyFilter.DEFAULT_GX_TAU). "
                         "noStumble no-detach WM: 0.4627; old lxdetach WM: -0.4338")
parser.add_argument("--print_interval", type=int, default=50,
                    help="Print status every N steps (default 50)")
parser.add_argument("--save_stats", type=str, default=None,
                    help="Path to save statistics JSON")
parser.add_argument("--log_values", action="store_true",
                    help="Log per-step g(x) and Q(z,cmd) to npz file")
parser.add_argument("--takeover_duration", type=int, default=50,
                    help="How many steps to stand still after danger (default 50 ≈ 1s @ 50Hz)")
parser.add_argument("--reset_grace_k", type=int, default=5,
                    help="Grace frames after ANY reset where K-filter is suppressed (RSSM warmup). "
                         "Must be >= 1. Prevents the RSSM's uninitialized state from triggering "
                         "a false takeover in the first few frames of a new episode. Default 5.")
parser.add_argument("--alarm_k", type=int, default=1,
                    help="Require K consecutive Q<threshold frames before takeover (default 1)")
parser.add_argument("--speed_match_frac", type=float, default=0.0,
                    help="Suppress alarm until robot speed >= frac * |cmd_vx|. "
                         "Prevents false alarms during the acceleration phase when speed << cmd. "
                         "Recommended: 0.7 for high-speed eval (cmd>=2.0). 0.0 = disabled (default).")
# ── [MOD 1] Cmd ramp argument ──────────────────────────────────────────────────
parser.add_argument("--cmd_accel_rate", type=float, default=0.0,
                    help="Cmd ramp: max rate (m/s²) at which cmd_vx grows from 0 toward target each episode. "
                         "0.0 = disabled / step command (default). "
                         "Recommended 1.0~2.0 for high-speed eval: keeps speed≈cmd so Q(z,cmd) "
                         "stays in-distribution, eliminating OOD false alarms during acceleration. "
                         "When enabled, --speed_match_frac is not needed and can stay at 0.")
# ───────────────────────────────────────────────────────────────────────────────
parser.add_argument("--show_q_rand", action="store_true",
                    help="Also compute Q_rand (random-action avg) each step for diagnostic comparison. "
                         "Adds ~20 extra critic forward passes per step; off by default.")
parser.add_argument("--adaptive_k", type=float, default=None,
                    help="Adaptive threshold: tau = mean(Q_rand) - k*std(Q_rand). None=disabled")
parser.add_argument("--adaptive_burnin", type=int, default=200,
                    help="Steps before adaptive threshold activates (default 200)")
parser.add_argument("--eval_terrain", type=str, default=None, choices=["flat", "cliff"],
                    help="Override terrain for evaluation: 'flat' → FPR (safe baseline), "
                         "'cliff' → FNR/TPR (dangerous, no-flat cliff-only). "
                         "In cliff mode, every episode is guaranteed to collide if filter does not intervene.")
parser.add_argument("--eval_h", type=int, default=30,
                    help="Eval: last H frames of fall episode used for TP/FN (default 30, matches demo eval)")
parser.add_argument("--eval_safe_strict_f", type=int, default=None,
                    help="Eval: first N frames of safe episode for FPR_strict (default max_steps//2 or 120). "
                         "Tail frames of timeout trajs may be near-danger; strict window isolates clearly-safe segment.")

# Video recording 参数
parser.add_argument("--record_video", action="store_true",
                    help="Record a video from the viewport camera.")
parser.add_argument("--video_path", type=str, default="./videos/recorded.mp4",
                    help="Output video file path (default: ./videos/recorded.mp4)")
parser.add_argument("--video_width", type=int, default=1280,
                    help="Video width in pixels (default: 1280)")
parser.add_argument("--video_height", type=int, default=720,
                    help="Video height in pixels (default: 720)")
parser.add_argument("--video_follow_offset", type=float, nargs=3, default=[-1.5, 0.0, 0.8],
                    metavar=("DX", "DY", "DZ"),
                    help="Camera offset from robot base in world frame [dx, dy, dz] (default: -1.5 0.0 0.8)")
parser.add_argument("--video_env_idx", type=int, default=0,
                    help="Which env to follow (default: 0)")
parser.add_argument("--video_no_follow", action="store_true",
                    help="Disable auto camera-follow. Viewport can be dragged manually; recorded angle is whatever the viewport shows.")

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


# ── Sim Q-value evaluation helpers (adapted from eval_q, same metrics/conventions) ─────────
def _first_consec_sim(arr, tau, K):
    """Return index of first K-consecutive frames with Q <= tau. Returns -1 if not found."""
    count = 0
    for i, v in enumerate(arr):
        count = count + 1 if v <= tau else 0
        if count >= K:
            return i - K + 1
    return -1


def _sweep_tau_sim(fall_q, safe_q, H, safe_strict_f, tau_list):
    """
    Frame-level metric sweep over tau for sim Q trajectories.
    Alarm convention: Q <= tau → unsafe  (same HJ convention as eval_q).

    fall_q        : list of 1-D np arrays — Q per step, one per fall episode
    safe_q        : list of 1-D np arrays — Q per step, one per timeout/safe episode
    H             : last H frames of fall traj → TP/FN window
    safe_strict_f : first N frames of safe traj → FPR_strict denominator
                    (tail frames may be near-danger, excluded to avoid penalising valid warnings)

    tau_opt selection: Youden-strict  max(TPR − FPR_strict)
      - FPR_strict is more conservative than FPR_all, giving a lower (stricter) tau.
      - Equivalent to eval_q.youden_tau_strict for offline calibration.
    """
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
    """
    延迟加载 CriticSafetyFilter

    关键：在 IsaacSim 完全初始化后再 import dreamerv3/PyHJ，避免 CUDA 冲突。
    """
    print("\n" + "=" * 60)
    print("Loading Critic Safety Filter (delayed import)...")
    print("=" * 60)

    from critic_safety_filter import CriticSafetyFilter

    filt = CriticSafetyFilter(
        wm_path=args.wm_path,
        critic_path=args.critic_path,
        threshold=args.safety_threshold,
        gx_tau=args.gx_tau,       # None → 使用 DEFAULT_GX_TAU
        device="cuda:0",
        env=env,
        adaptive_k=args.adaptive_k,
        adaptive_burnin=args.adaptive_burnin,
    )
    print("=" * 60 + "\n")
    return filt


def main():
    # ==================== 环境配置 ====================
    # --task 参数有最高优先级
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # ── Eval terrain override ──
    if args_cli.eval_terrain == 'flat':
        from legged_lab.terrains.terrain_generator_cfg import FLAT_MESH_TERRAINS_CFG
        env_cfg.scene.terrain_generator = FLAT_MESH_TERRAINS_CFG
        env_cfg.scene.terrain_generator.curriculum = False
    elif args_cli.eval_terrain == 'cliff':
        from legged_lab.terrains.terrain_generator_cfg import CLIFF_EVAL_TERRAINS_CFG
        env_cfg.scene.terrain_generator = CLIFF_EVAL_TERRAINS_CFG
        env_cfg.scene.terrain_generator.curriculum = False

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    env_cfg.noise.add_noise = False

    # 相机每步更新
    step_dt = env_cfg.sim.dt * env_cfg.sim.decimation
    env_cfg.scene.camera.update_period = step_dt

    env_cfg.scene.env_spacing = 10
    env_cfg.scene.max_episode_length_s = args_cli.max_episode_length_s

    # ==================== 创建环境 ====================
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    print(f"[INFO] Env ready: {env.num_envs} envs, {env.num_actions}D actions, device={env.device}")

    # ==================== 视频录制初始化 ====================
    video_writer = None
    rgb_annotator = None
    render_product = None
    if args_cli.record_video:
        import imageio
        import omni.replicator.core as rep

        vid_path = args_cli.video_path
        Path(vid_path).parent.mkdir(parents=True, exist_ok=True)

        W, H_vid = args_cli.video_width, args_cli.video_height
        # 先 render 一帧，让 annotator 稳定
        env.sim.render()

        render_product = rep.create.render_product(
            "/OmniverseKit_Persp", (W, H_vid)
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])
        # annotator 需要至少一次 render 后才有数据，再多渲染几帧稳定
        for _ in range(5):
            env.sim.render()

        video_fps = 1.0 / step_dt
        # 使用 imageio-ffmpeg（自带独立 ffmpeg 二进制）而非 cv2.VideoWriter，
        # 因为 Isaac Sim 进程会覆盖 OpenCV 链接的 libavcodec/libavformat，
        # 导致 cv2.VideoWriter.release() 无法正确写 MP4 moov atom → 文件无法播放。
        video_writer = imageio.get_writer(
            vid_path, fps=video_fps, codec="libx264",
            quality=8,  # 0-10, 高=好质量
            pixelformat="yuv420p",  # 确保各播放器兼容
        )
        print(f"[INFO] Video recording: {vid_path}  {W}x{H_vid} @ {video_fps:.1f}fps (imageio-ffmpeg)")

    # ==================== 加载 PPO policy ====================
    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume), device=agent_cfg.device)
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
    print("[INFO] Sensors warmed up (3 steps)")
 
    # ==================== 延迟加载 Critic Filter ====================
    critic_filter = load_critic_filter(env, args_cli)

    # ==================== 评估循环状态 ====================
    step_count = 0
    episode_count = 0
    danger_count = 0
    fall_count = 0
    takeover_ok_count = 0   # episodes ended by successful filter intervention (TP in cliff, FP in flat)
    alarm_step_count = 0    # cumulative (env×step) pairs where alarm_now is True (mode-independent)

    takeover_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    pending_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    episode_step_buf = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    TAKEOVER_DUR = args_cli.takeover_duration
    ALARM_K = max(1, int(args_cli.alarm_k))
    RESET_GRACE_K = max(1, int(args_cli.reset_grace_k))
    SPEED_MATCH_FRAC = float(args_cli.speed_match_frac)
    # ── [MOD 2] Cmd ramp state ─────────────────────────────────────────────────
    CMD_ACCEL_RATE = float(args_cli.cmd_accel_rate)   # m/s²; 0.0 = disabled
    # Per-env ramped |cmd_vx|: starts at 0 each episode, grows at CMD_ACCEL_RATE m/s/s.
    # Overrides env.command_generator.command[:, 0] so that policy, obs, and critic
    # all see the same in-distribution (speed ≈ cmd) value during acceleration.
    cmd_ramp = torch.zeros(env.num_envs, device=env.device)
    # ───────────────────────────────────────────────────────────────────────────
    low_q_run = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    # grace-period countdown: for RESET_GRACE_K frames after ANY reset, the K-filter is
    # suppressed and low_q_run is held at 0. Prevents RSSM warm-up artifacts (unreliable
    # Q values in the first few frames of a new episode) from triggering a spurious takeover
    # that looks like the old one is still running.
    grace_left = torch.full((env.num_envs,), fill_value=RESET_GRACE_K, dtype=torch.int32,
                            device=env.device)

    # ── Sim eval Q buffers (active when --eval_terrain is set) ──────────────────
    if args_cli.eval_terrain:
        ep_q_buf: list = [[] for _ in range(env.num_envs)]   # per-env per-episode Q values
        fall_q_trajs: list = []    # np arrays, one per fall episode
        safe_q_trajs: list = []    # np arrays, one per timeout/takeover-ok episode
        EVAL_H = args_cli.eval_h
        _ssf = args_cli.eval_safe_strict_f
        EVAL_SAFE_STRICT_F = _ssf if _ssf is not None else (
            (args_cli.max_steps // 2) if args_cli.max_steps else 120)
        TAU_SWEEP = np.linspace(-1.0, 1.0, 200)
        print(f"[INFO] Eval Q buffers active: H={EVAL_H}  safe_strict_f={EVAL_SAFE_STRICT_F}  "
              f"tau_sweep=200pts  (alarm: Q <= tau)")

    # 日志
    log_g, log_q, log_cmd = [], [], []

    env_timeout_steps = int(np.ceil(args_cli.max_episode_length_s / step_dt))
    print("\n[INFO] Starting evaluation " +
          ("(monitor-only)" if args_cli.monitor_only else "(filter active)") + "...")
    # ── [MOD 7] startup print: include cmd ramp info ─────────────────────────
    _ramp_info = (f"cmd_ramp={CMD_ACCEL_RATE:.1f}m/s²"
                  if CMD_ACCEL_RATE > 0 else "cmd_ramp=disabled")
    print(f"[INFO] Alarm rule: {ALARM_K} consecutive frames with Q < {args_cli.safety_threshold:.4f}  "
          f"reset_grace={RESET_GRACE_K}f  "
          f"speed_match={'disabled' if SPEED_MATCH_FRAC == 0 else f'{SPEED_MATCH_FRAC:.2f}×cmd_vx'}  "
          f"{_ramp_info}")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[INFO] Episode timeout (env-level):    {args_cli.max_episode_length_s:.1f}s = {env_timeout_steps} steps")
    if args_cli.max_steps:
        effective = min(args_cli.max_steps, env_timeout_steps)
        print(f"[INFO] Episode limit   (script-level): {args_cli.max_steps} steps  "
              f"→ effective cap = {effective} steps ({effective * step_dt:.1f}s)")
    else:
        print(f"[INFO] Episode limit   (script-level): None  → env timeout is the only cap")
    if args_cli.max_episodes:
        print(f"[INFO] Auto-exit after {args_cli.max_episodes} episodes")
    print()

    try:
        while simulation_app.is_running():
            if args_cli.max_episodes and episode_count >= args_cli.max_episodes:
                print(f"\n[INFO] Reached {args_cli.max_episodes} episodes, exiting.")
                break

            with torch.inference_mode():
                # 1. PPO generates joint actions
                actions_raw = policy(obs_dict)

                # 2. Get current velocity command from env (N, 3): [vx, vy, yaw]
                cmd_current = env.command_generator.command[:, :3].clone()

                # ── [MOD 3] Cmd ramp: grow cmd_vx at ≤ CMD_ACCEL_RATE m/s/s ──────────
                # Keeps (actual_speed, cmd) in-distribution for Q(z, cmd), so the critic
                # gives reliable readings throughout acceleration.  The ramped cmd is
                # written back to env.command_generator so policy obs, env reward, and
                # critic all see the same value.  Ramp is paused during takeover.
                _cmd_ramp_target = None          # populated below when CMD_ACCEL_RATE > 0
                if CMD_ACCEL_RATE > 0:
                    _target_abs = cmd_current[:, 0].abs()        # original target |vx|
                    _vx_sign    = cmd_current[:, 0].sign()
                    _advance    = ~(takeover_steps > 0)          # don't advance during takeover
                    cmd_ramp    = torch.where(
                        _advance,
                        (cmd_ramp + CMD_ACCEL_RATE * step_dt).clamp(max=_target_abs),
                        cmd_ramp,
                    )
                    _ramped_vx  = cmd_ramp * _vx_sign
                    env.command_generator.command[:, 0] = _ramped_vx   # robot follows ramped cmd
                    cmd_current[:, 0] = _ramped_vx                     # critic evaluates ramped cmd
                    _cmd_ramp_target = _target_abs                      # kept for display / in_accel
                # ─────────────────────────────────────────────────────────────────────

                # 3. Critic evaluates Q(z, cmd)
                cmd_out, is_safe, q_vals, g_vals, q_rand_vals = critic_filter.filter_cmd(
                    obs_dict, cmd_current,
                    apply_filter=not args_cli.monitor_only,
                    compute_q_rand=args_cli.show_q_rand,
                )

                # K-consecutive alarm state (online K-filter)
                unsafe_now = ~is_safe
                in_grace = grace_left > 0
                # Speed-match guard: suppress alarm while robot is still accelerating
                # (actual_speed < frac * cmd_vx). Q(z, cmd) is unreliable when speed << cmd
                # because the training data rarely includes (slow_robot, high_cmd) pairs.
                # ── [MOD 4] in_accel: split into suppress (K-filter) and display ──────
                # in_accel_suppress: gates low_q_run — only used by speed_match_frac mode.
                # in_accel:          display label only — also tracks cmd-ramp progress.
                # Keeping these separate preserves full alarm responsiveness when cmd ramp
                # is active: the Q is in-distribution so no suppression is needed.
                if SPEED_MATCH_FRAC > 0:
                    _actual_spd = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    _cmd_vx = env.command_generator.command[:, 0].abs()
                    in_accel_suppress = (_actual_spd < SPEED_MATCH_FRAC * _cmd_vx) & (_cmd_vx > 0.3)
                else:
                    in_accel_suppress = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

                if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                    # Display-only: which envs are still ramping toward their target cmd
                    in_accel = (cmd_ramp < _cmd_ramp_target - 0.05) & (_cmd_ramp_target > 0.3)
                else:
                    in_accel = in_accel_suppress   # same as suppress when ramp is off
                # ─────────────────────────────────────────────────────────────────────
                low_q_run = torch.where(unsafe_now & ~in_grace & ~in_accel_suppress, low_q_run + 1,
                                        torch.zeros_like(low_q_run))
                alarm_now = low_q_run >= ALARM_K
                alarm_step_count += int(alarm_now.sum().item())
                # grace_left is decremented after env.step() (below) so it counts simulation
                # steps, not loop iterations.  continue-cycles skip env.step() but still
                # executed this code block — decrementing here would silently eat grace frames
                # whenever other envs finish their takeover in the same simulation step.

                # ── per-step Q append for tau calibration ──
                if args_cli.eval_terrain:
                    for _i in range(env.num_envs):
                        ep_q_buf[_i].append(float(q_vals[_i].item()))

                # 4. log
                if args_cli.log_values:
                    log_g.append(g_vals.cpu().numpy().copy())
                    log_q.append(q_vals.cpu().numpy().copy())
                    log_cmd.append(cmd_current.cpu().numpy().copy())

                # 5. Takeover logic
                """
                Q < threshold → alarm → takeover (stand still) → reset after takeover duration
                如果已经跌倒，则不管 Q 值，直接 reset（不接管） → 让环境来判定 episode 结束，统计摔倒事件
                
                普通reset逻辑：
                
                """
                if not args_cli.monitor_only:
                    danger = alarm_now & (takeover_steps == 0) # shape = [num_envs]
                    if danger.any():
                        takeover_steps[danger] = TAKEOVER_DUR #接管持续时间
                        pending_reset[danger] = True #待重置
                        d_envs = torch.where(danger)[0].tolist()
                        d_q = q_vals[danger].tolist()
                        d_g = g_vals[danger].tolist()
                        d_cmd = cmd_current[danger].tolist()
                        danger_count += danger.sum().item()
                        print(f"\033[93m[WARN] Step {step_count}: DANGER in envs {d_envs}\033[0m")
                        for i, e in enumerate(d_envs):
                            qr_diag = (f"  Q_rand={q_rand_vals[danger][i].item():.3f}"
                                       if args_cli.show_q_rand else "")
                            print(f"  env{e}: Q={d_q[i]:.3f}{qr_diag}  "
                                  f"g={d_g[i]:.3f}  "
                                  f"cmd=({d_cmd[i][0]:.2f},{d_cmd[i][1]:.2f},{d_cmd[i][2]:.2f})")
                        print(f"  → Standing still for {TAKEOVER_DUR} steps (alarm_k={ALARM_K})")

                    is_takeover = takeover_steps > 0
                    if is_takeover.any():
                        # Override command to zero → policy stands still
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
                        env.reset(done_envs)
                        obs_dict = env.get_observations()
                        pending_reset[tk_done] = False
                        low_q_run[tk_done] = 0
                        episode_step_buf[tk_done] = 0
                        grace_left[tk_done] = RESET_GRACE_K   # RSSM warmup guard
                        cmd_ramp[tk_done] = 0.0               # [MOD 6a] restart ramp for new episode
                        # 接管后主动 reset → 安全轨迹（未摔倒）
                        critic_filter.finalize_episode(done_envs, is_fall=False)
                        episode_count += tk_done.sum().item()
                        takeover_ok_count += tk_done.sum().item()   # TP (cliff) or FP (flat)
                        if args_cli.eval_terrain:  # flush Q buf → safe (takeover prevented fall)
                            for _i in done_envs.tolist():
                                _arr = np.array(ep_q_buf[_i])
                                if len(_arr):
                                    safe_q_trajs.append(_arr)
                                ep_q_buf[_i] = []
                        continue


                # 6. Step env
                # ── 视频录制：先更新相机位置，step() 内的 sim.render() 用新位置渲染 ──
                if args_cli.record_video and video_writer is not None and not args_cli.video_no_follow:
                    _follow_idx = min(args_cli.video_env_idx, env.num_envs - 1)
                    _robot_pos = env.robot.data.root_pos_w[_follow_idx].cpu().numpy()
                    _offset = np.array(args_cli.video_follow_offset, dtype=np.float64)
                    _eye = (_robot_pos + _offset).tolist()
                    _target = _robot_pos.tolist()
                    env.sim.set_camera_view(eye=_eye, target=_target)

                obs_dict, rewards, dones, infos = env.step(actions_raw)  # sim.render() 在此发生
                step_count += 1

                # ── 视频录制：step() 后读取 annotator（帧已含上方设置的相机位置）──
                if args_cli.record_video and video_writer is not None:
                    try:
                        _rgb_raw = rgb_annotator.get_data()
                        # get_data(device='cpu') 返回 numpy uint8 array (H, W, 4) RGBA
                        if isinstance(_rgb_raw, np.ndarray) and _rgb_raw.size > 0:
                            # imageio 接受 RGB，取前3通道去掉 Alpha
                            video_writer.append_data(_rgb_raw[:, :, :3])
                        else:
                            # 渲染器预热期间返回空数据，写黑帧占位保证帧率一致
                            video_writer.append_data(np.zeros(
                                (args_cli.video_height, args_cli.video_width, 3), dtype=np.uint8))
                    except Exception as _verr:
                        print(f"[WARN] Video frame capture error: {_verr}")
                        video_writer.append_data(np.zeros(
                            (args_cli.video_height, args_cli.video_width, 3), dtype=np.uint8))
                episode_step_buf += 1
                grace_left = torch.clamp(grace_left - 1, min=0)

                # ── 7. Handle ALL env dones (最高优先级，无条件) ──
                # base_env.step() 内部已经 auto-reset 了所有 done 的 env，
                # 返回的 obs 已经是新 episode 的第一帧。
                # 不管 env 当前处于 RUNNING / ALARM / TAKEOVER，
                # 只要 dones=True 就必须清除全部脚本状态。
                if dones.any():
                    done_env_ids = torch.where(dones)[0]
                    was_in_takeover = (takeover_steps > 0) & dones

                    # 判定摔倒/超时
                    if isinstance(infos, dict) and ("time_outs" in infos):
                        time_outs = infos["time_outs"]
                        if not isinstance(time_outs, torch.Tensor):
                            time_outs = torch.as_tensor(time_outs, device=dones.device)
                        is_fall_per_env = ~time_outs[dones].bool()
                    else:
                        is_fall_per_env = g_vals[dones] < 0

                    fall_count += is_fall_per_env.sum().item()
                    episode_count += done_env_ids.shape[0]
                    critic_filter.finalize_episode(done_env_ids, is_fall_per_env)

                    # 无条件清除所有脚本状态
                    low_q_run[dones] = 0
                    episode_step_buf[dones] = 0
                    takeover_steps[dones] = 0
                    pending_reset[dones] = False
                    grace_left[dones] = RESET_GRACE_K   # RSSM warmup guard
                    cmd_ramp[dones] = 0.0               # [MOD 6b] restart ramp for new episode
                    if args_cli.eval_terrain:  # flush Q buf → fall or safe per episode type
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

                # ── 8. Script forced timeout (仅未 done 的 env) ──
                if args_cli.max_steps is not None:
                    forced_timeout = (episode_step_buf >= args_cli.max_steps) & ~dones
                    if not args_cli.monitor_only:
                        forced_timeout &= ~(takeover_steps > 0)
                    if forced_timeout.any():
                        ft_ids = torch.where(forced_timeout)[0]
                        episode_count += ft_ids.shape[0]
                        env.reset(ft_ids)
                        obs_dict = env.get_observations()
                        critic_filter.finalize_episode(ft_ids, is_fall=False)
                        low_q_run[forced_timeout] = 0
                        episode_step_buf[forced_timeout] = 0
                        takeover_steps[forced_timeout] = 0
                        pending_reset[forced_timeout] = False
                        grace_left[forced_timeout] = RESET_GRACE_K   # RSSM warmup guard
                        cmd_ramp[forced_timeout] = 0.0               # [MOD 6c] restart ramp for new episode
                        if args_cli.eval_terrain:  # flush Q buf → safe (script timeout = survived)
                            for _i in ft_ids.tolist():
                                _arr = np.array(ep_q_buf[_i])
                                if len(_arr):
                                    safe_q_trajs.append(_arr)
                                ep_q_buf[_i] = []
                        print(f"\033[96m[TIMEOUT] Step {step_count}: envs {ft_ids.tolist()} "
                              f"hit max_steps={args_cli.max_steps}\033[0m")

                # ── 9. Periodic status ──
                if step_count % args_cli.print_interval == 0:
                    fall_rate = fall_count / episode_count * 100 if episode_count > 0 else 0.0
                    is_tk  = takeover_steps > 0
                    is_alm = (low_q_run > 0) & ~is_tk
                    is_grc = (grace_left > 0) & ~is_tk & ~is_alm
                    is_acc = in_accel & ~is_tk & ~is_alm & ~is_grc
                    n_tk, n_alm, n_grc = int(is_tk.sum()), int(is_alm.sum()), int(is_grc.sum())
                    n_acc  = int(is_acc.sum())
                    n_run  = env.num_envs - n_tk - n_alm - n_grc - n_acc
                    qr_str = f"  Qr={q_rand_vals.mean():+.3f}" if args_cli.show_q_rand else ""
                    # ── [MOD 5a] show ACCEL/RAMP count in summary line ────────────────
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
                        _acc = bool(in_accel[_i].item()) if (SPEED_MATCH_FRAC > 0 or CMD_ACCEL_RATE > 0) else False
                        if _tk > 0:
                            _lbl = f"\033[91m[TAKEOVER rem={_tk:3d}]\033[0m"
                        elif _lqr > 0:
                            _lbl = f"\033[93m[ALARM    run={_lqr:3d}]\033[0m"
                        elif _gl > 0:
                            _lbl = f"\033[94m[GRACE    left={_gl:3d}]\033[0m"
                        elif _acc:
                            # ── [MOD 5b] per-env label: distinguish ramp vs speed-frac ──
                            if CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                                _rv = float(cmd_ramp[_i].item())
                                _tv = float(_cmd_ramp_target[_i].item())
                                _lbl = f"\033[35m[RAMP  {_rv:.2f}→{_tv:.2f}m/s]\033[0m"
                            else:
                                _lbl = f"\033[35m[ACCEL    spd<cmd  ]\033[0m"
                            # ─────────────────────────────────────────────────────────
                        elif show_all:
                            _lbl = "\033[92m[RUNNING          ]\033[0m"
                        else:
                            continue
                        _ep  = int(episode_step_buf[_i].item())
                        _qi  = q_vals[_i].item()
                        _gi  = g_vals[_i].item()
                        _spd = torch.linalg.vector_norm(robot.data.root_lin_vel_b[_i, :2]).item()
                        _cmd = env.command_generator.command[_i]
                        _need = f"{SPEED_MATCH_FRAC * _cmd[0].item():.2f}" if SPEED_MATCH_FRAC > 0 else ""
                        # ── [MOD 5c] spd tag: show ramp progress when cmd ramp active ─
                        if _acc and CMD_ACCEL_RATE > 0 and _cmd_ramp_target is not None:
                            _spd_tag = f"  spd={_spd:.2f}"
                        elif _acc:
                            _spd_tag = f"  spd={_spd:.2f}(need≥{_need})"
                        else:
                            _spd_tag = f"  spd={_spd:.2f}"
                        # ─────────────────────────────────────────────────────────────
                        print(f"  env{_i:3d} {_lbl}  ep_step={_ep:4d}  "
                              f"Q={_qi:+.4f}  g={_gi:+.4f}  "
                              f"cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f}){_spd_tag}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as _exc:
        print(f"\n[ERROR] Exception in main loop: {_exc}")
        import traceback; traceback.print_exc()
    finally:
        # 无论正常退出/Ctrl-C/异常，都确保 video_writer.close() 被调用，
        # imageio-ffmpeg 在 close() 时 flush 并写完整 MP4 容器。
        if video_writer is not None:
            video_writer.close()
            print(f"  Saved video: {args_cli.video_path}")
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass

    # ==================== 打印结果 ====================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Steps:      {step_count}")
    print(f"  Episodes:   {episode_count}")
    print(f"  Dangers:    {danger_count}")
    print(f"  Falls:      {fall_count}")
    if episode_count > 0:
        print(f"  Fall rate:  {fall_count/episode_count*100:.1f}%")
    critic_filter.print_stats(monitor_only=args_cli.monitor_only)

    # ── Eval terrain summary ──
    if args_cli.eval_terrain and episode_count > 0:
        total_env_steps = step_count * (env.num_envs if hasattr(env, 'num_envs') else 1)
        print(f"\n{'='*60}")
        print(f"EVAL SUMMARY  terrain={args_cli.eval_terrain}  "
              f"mode={'monitor-only' if args_cli.monitor_only else 'filter-active'}")
        print(f"{'='*60}")
        if args_cli.eval_terrain == 'cliff':
            if args_cli.monitor_only:
                # Baseline: no filter, pure collision rate
                timeout_count = episode_count - fall_count
                print(f"  Episodes  total:          {episode_count:5d}")
                print(f"  Falls     (collision):    {fall_count:5d}")
                print(f"  Timeouts  (survived):     {timeout_count:5d}")
                print(f"  Baseline collision rate:  {fall_count/episode_count*100:.1f}%")
            else:
                # Filter active: FN = fell without alarm, TP = saved by filter
                valid = fall_count + takeover_ok_count
                ambiguous = episode_count - valid   # forced_timeout on cliff (robot never reached edge)
                print(f"  Episodes  total:          {episode_count:5d}")
                print(f"  FN  (fell, Q missed):     {fall_count:5d}")
                print(f"  TP  (saved by filter):    {takeover_ok_count:5d}")
                print(f"  Ambiguous (timeout):      {ambiguous:5d}  ← excluded from FNR")
                if valid > 0:
                    print(f"  FNR = {fall_count}/{valid} = {fall_count/valid*100:.1f}%")
                    print(f"  TPR = {takeover_ok_count}/{valid} = {takeover_ok_count/valid*100:.1f}%")
                if ambiguous > 0:
                    print(f"  [WARN] {ambiguous} ambiguous episodes: robot hit max_steps without "
                          f"collision or filter trigger → consider reducing --max_steps")
        elif args_cli.eval_terrain == 'flat':
            step_fpr = alarm_step_count / total_env_steps if total_env_steps > 0 else 0.0
            if args_cli.monitor_only:
                print(f"  Total env-steps:          {total_env_steps:7d}")
                print(f"  Alarm env-steps (Q<tau):  {alarm_step_count:7d}")
                print(f"  Step-level alarm rate:    {step_fpr*100:.2f}%")
            else:
                # Filter active: intervention on safe terrain = FP episode
                tn_count = episode_count - takeover_ok_count - fall_count
                ep_fpr = takeover_ok_count / episode_count
                print(f"  Episodes  total:          {episode_count:5d}")
                print(f"  FP  (false alarm → reset):{takeover_ok_count:5d}")
                print(f"  TN  (completed, no alarm):{tn_count:5d}")
                if fall_count > 0:
                    print(f"  Unexpected falls (!):     {fall_count:5d}")
                print(f"  Episode-level FPR:        {ep_fpr*100:.1f}%")
                print(f"  Step-level alarm rate:    {step_fpr*100:.2f}%")

        # ── Tau calibration sweep (equivalent to eval_q offline analysis) ────────
        # Recommended usage: --eval_terrain cliff --monitor_only
        # In MO mode, episode outcomes are unaffected by filter → unbiased TP/FP/FN/TN.
        # In filter-active mode, results are valid for cliff (FP=TN=0 by construction)
        # but biased for flat (TP indistinguishable from FP without counterfactual).
        nf, ns = len(fall_q_trajs), len(safe_q_trajs)
        print(f"\n  Tau sweep: {nf} fall episodes, {ns} safe/timeout episodes")
        print(f"  H={EVAL_H}  FPR_strict_window={EVAL_SAFE_STRICT_F} frames  alarm: Q <= tau")
        if not args_cli.monitor_only:
            print(f"  [NOTE] filter-active mode: TP/FP ambiguous on flat terrain.")
        if nf >= 5 and ns >= 5:
            sw = _sweep_tau_sim(fall_q_trajs, safe_q_trajs,
                                EVAL_H, EVAL_SAFE_STRICT_F, TAU_SWEEP)
            # Youden-strict: max(TPR - FPR_strict)
            # FPR_strict uses only first EVAL_SAFE_STRICT_F frames → clearly-safe segment.
            # Rationale identical to eval_q.youden_tau_strict: late frames of timeout trajs
            # can be near-cliff and should not count as false alarms against the filter.
            youden = [t - f for t, f in zip(sw['tpr'], sw['fpr_strict'])]
            yi = int(np.argmax(youden))
            tau_sim = float(TAU_SWEEP[yi])
            print(f"\n  tau_opt (Youden-strict) = {tau_sim:.3f}")
            print(f"  {'':12}  {'TPR':>6}  {'FPR_all':>8}  {'FPR_str':>8}  {'FNR':>6}  {'F1':>6}  {'Lead':>7}")
            print(f"  {'tau_opt':<12}  "
                  f"{sw['tpr'][yi]*100:>5.1f}%  "
                  f"{sw['fpr_all'][yi]*100:>7.1f}%  "
                  f"{sw['fpr_strict'][yi]*100:>7.1f}%  "
                  f"{sw['fnr'][yi]*100:>5.1f}%  "
                  f"{sw['f1'][yi]:>6.3f}  "
                  f"{sw['lead'][yi]:>6.1f}f")
            # K-filter sweep at tau_sim
            print(f"\n  K-filter at tau_sim={tau_sim:.3f}:")
            print(f"  {'K':>3}  {'FPR_K%':>8}  {'FNR_K%':>8}  {'Lead_K':>8}")
            for K in range(1, 9):
                fpr_hits, fnr_miss, k_leads = 0, 0, []
                for arr in safe_q_trajs:
                    if _first_consec_sim(arr, tau_sim, K) >= 0:
                        fpr_hits += 1
                for arr in fall_q_trajs:
                    idx = _first_consec_sim(arr, tau_sim, K)
                    if idx < 0:
                        fnr_miss += 1
                    else:
                        k_leads.append(len(arr) - idx)
                fpr_k = fpr_hits / max(ns, 1)
                fnr_k = fnr_miss / max(nf, 1)
                print(f"  {K:>3}  {fpr_k*100:>7.1f}%  {fnr_k*100:>7.1f}%  "
                      f"{np.mean(k_leads):>7.1f}f" if k_leads else
                      f"  {K:>3}  {fpr_k*100:>7.1f}%  {fnr_k*100:>7.1f}%  {'N/A':>8}")
        else:
            print(f"  Insufficient data for sweep (need ≥5 fall + ≥5 safe; got {nf}+{ns}).")
            print(f"  Suggestions: increase --max_episodes, or use --eval_terrain cliff --monitor_only")

    # ── save logs ──
    if args_cli.log_values and log_g:
        out = Path("./logs/critic_filter_log.npz")
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
        stats = critic_filter.stats.summary()
        stats.update({
            'config': {
                'task': args_cli.task, 'num_envs': env.num_envs,
                'threshold': args_cli.safety_threshold,
                'alarm_k': ALARM_K,
                'cmd_accel_rate': CMD_ACCEL_RATE,      # [MOD 8]
                'speed_match_frac': SPEED_MATCH_FRAC,  # [MOD 8]
                'monitor_only': args_cli.monitor_only,
                'max_episode_length_s': args_cli.max_episode_length_s,
                'max_steps_per_ep': args_cli.max_steps,
                'max_episodes': args_cli.max_episodes,
                'total_steps': step_count, 'episodes': episode_count,
                'danger_events': danger_count, 'falls': fall_count,
                'fall_rate': fall_count / episode_count if episode_count > 0 else None,
            }
        })
        with open(sp, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats: {sp}")

    # ── 视频已在上方 finally 块中释放 ──

    print("\nDone!")


if __name__ == "__main__":
    main()
    simulation_app.close()
