"""
Critic-as-Filter: Q(z, cmd) 安全过滤器

基于 DDPG Critic 的面向未来安全过滤器，用于 LeggedLab/IsaacSim Go2 部署。

核心区别 (vs 旧版 safety_monitor.py 的 g(x)):
  ┌──────────┬──────────────────────────────────────────────────┐
  │  g(x)    │ 单帧安全分类 — "现在摔了没？"  (reactive)         │
  │  Q(z, u) │ 未来安全价值 — "按这个 cmd 走会不会摔？" (proactive)│
  │          │ action-conditioned: 同一状态 + 不同 cmd → 不同安全值│
  └──────────┴──────────────────────────────────────────────────┘

评估结果 (hvr_nodecay ep25, lxdetach WM, gamma=0.995):
  ablation winner — HVR+NoDec(1.0), actor_frozen, best_epoch=25
  gx_tau = -0.4338

运行方式:
    python scripts/play_with_critic_filter.py \\
        --task go2_flat --enable_cameras --headless \\
        --num_envs 4 --max_steps 2000

依赖:
    - latent-safety 仓库里的 WM + Critic checkpoint
    - dreamerv3-torch (WM encoder + RSSM + margin head)
    - PyHJ (Critic 网络结构)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
import sys
import os


# ─────────────────────────────────────────────────────────────────────────────
# 路径设置 (在第一次 import 时执行)
# ─────────────────────────────────────────────────────────────────────────────
_LATENT_SAFETY_ROOT = Path("/home/lcy/latent-safety")
_DREAMER_DIR = _LATENT_SAFETY_ROOT / "dreamerv3-torch"

for p in [str(_LATENT_SAFETY_ROOT), str(_DREAMER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# Action-space mapping (与 DDPG 训练一致)
# ─────────────────────────────────────────────────────────────────────────────
# Must match PyHJ/reach_rl_gym_envs/go2_wm.py used for DDPG training.
CMD_LOW  = np.array([0.0, -0.3, -0.3], dtype=np.float32)  # vx, vy, yaw_rate
CMD_HIGH = np.array([2.8,  0.3,  0.3], dtype=np.float32)

def cmd_to_normalized(cmd_physical: np.ndarray) -> np.ndarray:
    """物理 cmd -> [-1, 1]，超出训练范围的命令 clip 到边界"""
    raw = (cmd_physical - CMD_LOW) / (CMD_HIGH - CMD_LOW) * 2.0 - 1.0
    return np.clip(raw, -1.0, 1.0)

def normalized_to_cmd(act_norm: np.ndarray) -> np.ndarray:
    """[-1, 1] → 物理 cmd"""
    act_clipped = np.clip(act_norm, -1.0, 1.0)
    return CMD_LOW + (CMD_HIGH - CMD_LOW) * (act_clipped + 1.0) / 2.0


def preprocess_depth_like_training(depth_hw: np.ndarray) -> np.ndarray:
    """
    与训练完全一致的深度预处理 (对齐 tools.py::fill_expert_dataset_go2)：
      1) nan/neginf → 0.0，posinf → 3.0m
      2) 线性映射到 [0,1]：clip(img / 3.0, 0, 1)
      3) 映射到 [0,255]，输出 (H,W,1) float32

    ⚠️ 不使用 p2-p98 分位拉伸！训练代码已切换为线性 [0, 3.0] 映射。
    """
    img = np.array(depth_hw, dtype=np.float32)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    img = np.nan_to_num(img, nan=0.0, posinf=3.0, neginf=0.0)
    if img.max() > 1.0:
        img = np.clip(img / 3.0, 0, 1)

    img = (img * 255).astype(np.uint8).astype(np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 单条轨迹记录
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrajectoryRecord:
    """
    一条完整轨迹的 per-step Q / g 序列及元信息。

    is_fall=True  → 轨迹以摔倒结束（用于 early-warning 分析）
    is_fall=False → 超时安全结束（用于 false-positive 率计算）
    """
    env_id: int
    length: int
    is_fall: bool
    q_series: np.ndarray        # (T,) Q(z, actual_cmd)
    q_rand_series: np.ndarray   # (T,) Q_avg(z, random actions)  — ablation 语义
    g_series: np.ndarray        # (T,) g(x)
    threshold: float            # Q 报警阈值

    @property
    def q_alarm_step(self) -> Optional[int]:
        """Q 第一次 <= threshold 的步骤 (None = 从未报警)  — 与 ablation <= 一致"""
        steps = np.where(self.q_series <= self.threshold)[0]
        return int(steps[0]) if len(steps) > 0 else None

    @property
    def g_alarm_step(self) -> Optional[int]:
        """g(x) 第一次 <= 0 的步骤 (None = 从未报警)"""
        steps = np.where(self.g_series <= 0)[0]
        return int(steps[0]) if len(steps) > 0 else None

    @property
    def q_lead(self) -> Optional[int]:
        """
        Q 相对 g 的预警提前量（步数）。
        正值 = Q 更早报警；负值 = g 更早；None = 有一方从未报警
        """
        q, g = self.q_alarm_step, self.g_alarm_step
        if q is None or g is None:
            return None
        return g - q

    @property
    def q_lead_from_end(self) -> Optional[int]:
        """
        与 ablation 一致的 lead 定义：T - first_alarm_step
        即从首次报警到轨迹末尾的帧数（越大 = 预警越早）
        """
        q = self.q_alarm_step
        return self.length - q if q is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# 统计记录
# ─────────────────────────────────────────────────────────────────────────────
class CriticFilterStats:
    """
    Per-env / per-traj 级别统计。

    使用方式:
        stats.log_step(g_per_env, q_per_env)    # 每 sim step（传 numpy 向量）
        stats.finalize_episode(env_ids, is_fall) # 每轮 episode 结束
        stats.summary()                          # 汇总打印
    """

    def __init__(self, num_envs: int, threshold: float = 0.0):
        self.num_envs = num_envs
        self.threshold = threshold
        self.total_steps: int = 0
        self.interventions: int = 0

        # Per-env 轨迹缓冲（episode 内逐步累积，episode 结束后提交）
        self._env_q: List[List[float]] = [[] for _ in range(num_envs)]
        self._env_q_rand: List[List[float]] = [[] for _ in range(num_envs)]
        self._env_g: List[List[float]] = [[] for _ in range(num_envs)]

        # 已完成的轨迹记录
        self.trajectories: List[TrajectoryRecord] = []

    def log_step(self, g_per_env: np.ndarray, q_per_env: np.ndarray,
                 q_rand_per_env: np.ndarray = None):
        """每步记录所有 env 的 g 和 Q（不做 env 间平均）"""
        self.total_steps += 1
        for i in range(self.num_envs):
            self._env_q[i].append(float(q_per_env[i]))
            self._env_g[i].append(float(g_per_env[i]))
            if q_rand_per_env is not None:
                self._env_q_rand[i].append(float(q_rand_per_env[i]))
        # 任意 env Q < threshold → 本步计入"介入需求"
        if np.any(q_per_env < self.threshold):
            self.interventions += 1

    def finalize_episode(self, env_ids: List[int], is_fall: List[bool]):
        """episode 结束时提交轨迹记录并清空对应 env 的缓冲"""
        for idx, env_id in enumerate(env_ids):
            q_buf      = self._env_q[env_id]
            q_rand_buf = self._env_q_rand[env_id]
            g_buf      = self._env_g[env_id]
            if len(q_buf) == 0:
                continue
            # q_rand 可能尚未开始记录（实验六早期兼容），用 nan 填充
            if len(q_rand_buf) == 0:
                q_rand_arr = np.full(len(q_buf), float('nan'), dtype=np.float32)
            else:
                q_rand_arr = np.array(q_rand_buf, dtype=np.float32)
            rec = TrajectoryRecord(
                env_id=env_id,
                length=len(q_buf),
                is_fall=bool(is_fall[idx]),
                q_series=np.array(q_buf, dtype=np.float32),
                q_rand_series=q_rand_arr,
                g_series=np.array(g_buf, dtype=np.float32),
                threshold=self.threshold,
            )
            self.trajectories.append(rec)
            self._env_q[env_id]      = []
            self._env_q_rand[env_id] = []
            self._env_g[env_id]      = []

    def summary(self) -> Dict:
        trajs = self.trajectories
        n_steps = max(self.total_steps, 1)

        base: Dict = {
            "total_steps": self.total_steps,
            "intervention_rate": self.interventions / n_steps,
            "total_trajs": len(trajs),
            "fall_trajs": 0,
            "safe_trajs": 0,
        }

        if not trajs:
            return base

        fall_trajs = [t for t in trajs if t.is_fall]
        safe_trajs = [t for t in trajs if not t.is_fall]
        base["fall_trajs"] = len(fall_trajs)
        base["safe_trajs"] = len(safe_trajs)

        # ── 全局 g / Q 分布（跨所有轨迹、所有时间步）──
        all_g = np.concatenate([t.g_series for t in trajs])
        all_q = np.concatenate([t.q_series for t in trajs])
        base.update({
            "g_mean": float(np.mean(all_g)),
            "g_min":  float(np.min(all_g)),
            "g_negative_rate": float(np.mean(all_g < 0)),
            "q_mean": float(np.mean(all_q)),
            "q_min":  float(np.min(all_q)),
            "q_negative_rate": float(np.mean(all_q < self.threshold)),
        })

        # ── Q_rand (ablation 语义) 分布 ──
        q_rand_all = np.concatenate([t.q_rand_series for t in trajs])
        if np.isfinite(q_rand_all).any():
            valid = q_rand_all[np.isfinite(q_rand_all)]
            base["q_rand_mean"] = float(np.mean(valid))
            base["q_rand_negative_rate"] = float(np.mean(valid < self.threshold))
        else:
            base["q_rand_mean"] = float('nan')
            base["q_rand_negative_rate"] = float('nan')

        # ── 提前预警分析（仅摔倒轨迹，Q 和 g 都报警才能计算 lead）──
        leads = [t.q_lead for t in fall_trajs if t.q_lead is not None]
        if leads:
            base["lead_q_before_g_rate"]       = float(np.mean([l > 0 for l in leads]))
            base["lead_avg_steps"]              = float(np.mean(leads))
            leads_pos = [l for l in leads if l > 0]
            base["lead_avg_steps_when_q_first"] = float(np.mean(leads_pos)) if leads_pos else float('nan')
        else:
            base["lead_q_before_g_rate"]       = float('nan')
            base["lead_avg_steps"]              = float('nan')
            base["lead_avg_steps_when_q_first"] = float('nan')

        # Q 报警但 g 始终未报警的摔倒轨迹比（Q 独自发现危险）
        if fall_trajs:
            base["q_alarm_only_rate"] = float(np.mean(
                [t.q_alarm_step is not None and t.g_alarm_step is None
                 for t in fall_trajs]
            ))
        else:
            base["q_alarm_only_rate"] = float('nan')

        # ── 假阳性率（超时安全轨迹上的误报）——轨迹级别 ──
        if safe_trajs:
            base["q_fpr_traj"] = float(np.mean([t.q_alarm_step is not None for t in safe_trajs]))
            base["g_fpr_traj"] = float(np.mean([t.g_alarm_step is not None for t in safe_trajs]))
        else:
            base["q_fpr_traj"] = float('nan')
            base["g_fpr_traj"] = float('nan')

        # ── 帧级别 FPR / TPR —— 与 ablation_hvr_q.py 完全对齐 ──
        # FPR_frame: 安全帧(每traj前SAFE_MAX_F帧)中 Q<=threshold 的比例
        # TPR_frame: 摔倒轨迹末30帧中 Q<=threshold 的比例
        # noStumble 数据集 timeout=250 步，原 lxdetach 数据集=120 步
        SAFE_MAX_F = 250
        H_FALL = 30
        safe_frame_fp, safe_frame_total = 0, 0
        for t in safe_trajs:
            arr = t.q_series[:SAFE_MAX_F]
            safe_frame_fp    += int(np.sum(arr <= self.threshold))
            safe_frame_total += len(arr)
        fall_frame_tp, fall_frame_total = 0, 0
        for t in fall_trajs:
            arr = t.q_series[-H_FALL:]
            fall_frame_tp    += int(np.sum(arr <= self.threshold))
            fall_frame_total += len(arr)
        base["q_fpr_frame"] = float(safe_frame_fp  / max(safe_frame_total, 1))
        base["q_tpr_frame"] = float(fall_frame_tp  / max(fall_frame_total, 1))

        # 帧级别 lead（与 ablation 一致）: 仅摔倒轨迹，T - first_alarm
        leads_from_end = [t.q_lead_from_end for t in fall_trajs if t.q_lead_from_end is not None]
        base["q_lead_frame"] = float(np.mean(leads_from_end)) if leads_from_end else float('nan')

        # (legacy alias 保留向后兼容)
        base["q_fpr"] = base["q_fpr_traj"]
        base["g_fpr"] = base["g_fpr_traj"]

        return base

    def kfilter_sweep(self, K_range=None, tau=None, safe_max_f: int = 250) -> Dict:
        """
        K-consecutive filter sweep，完全 post-hoc（对已存储的 q_series 做滑窗判断）。
        与 ablation_hvr_q.py 的 compute_kfilter_metrics() 逻辑相同，可直接对比。

        K=1 等价于无 K-filter 的帧级轨迹判断（只要出现一帧 Q<=tau 就报警）。
        过滤有无 (monitor vs active) 不影响此函数——它只看已记录的 q_series。

        Args:
            K_range: list of K values, default [1..8]
            tau:     alarm threshold, default = self.threshold
            safe_max_f: 安全轨迹只取前 N 帧，与 ablation 保持一致

        Returns:
            dict: K -> {'fpr_K', 'fnr_K', 'lead_K', 'n_safe', 'n_fall'}
        """
        if K_range is None:
            K_range = list(range(1, 9))
        if tau is None:
            tau = self.threshold

        fall_trajs = [t for t in self.trajectories if t.is_fall]
        safe_trajs = [t for t in self.trajectories if not t.is_fall]

        def _first_consec(arr, k):
            """首次连续 k 帧 <= tau 的起始索引，不存在返回 -1"""
            count = 0
            for i, v in enumerate(arr):
                if v <= tau:
                    count += 1
                    if count >= k:
                        return i - k + 1
                else:
                    count = 0
            return -1

        results = {}
        for K in K_range:
            fpr_hits, fpr_total = 0, 0
            for t in safe_trajs:
                fpr_total += 1
                if _first_consec(t.q_series[:safe_max_f], K) >= 0:
                    fpr_hits += 1

            fnr_miss, fnr_total = 0, 0
            leads = []
            for t in fall_trajs:
                fnr_total += 1
                idx = _first_consec(t.q_series, K)
                if idx < 0:
                    fnr_miss += 1
                else:
                    leads.append(t.length - idx)

            results[K] = {
                'fpr_K':  float(fpr_hits / max(fpr_total, 1)),
                'fnr_K':  float(fnr_miss / max(fnr_total, 1)),
                'lead_K': float(np.mean(leads)) if leads else float('nan'),
                'n_safe': fpr_total,
                'n_fall': fnr_total,
            }
        return results

    def tau_sweep(self, key: str = 'q_rand', n_pts: int = 50,
                  tau_range: tuple = (-1.0, 0.5),
                  safe_max_f: int = 120, H_fall: int = 30) -> Dict:
        """
        在已采集的轨迹上扫描 tau，计算帧级 FPR/TPR/Youden，找最优部署阈值。

        使用 q_rand_series（ablation 语义）或 q_series（actual cmd）。

        Args:
            key:        'q_rand' (推荐) 或 'q'
            n_pts:      扫描点数
            tau_range:  (min_tau, max_tau)
            safe_max_f: 安全轨迹每条最多取多少帧计 FPR
            H_fall:     摔倒轨迹末 H 帧计 TPR

        Returns:
            dict with keys:
                'taus':    np.ndarray (n_pts,)
                'fpr':     np.ndarray (n_pts,)
                'tpr':     np.ndarray (n_pts,)
                'youden':  np.ndarray (n_pts,)  = TPR - FPR
                'f1':      np.ndarray (n_pts,)
                'tau_youden': float  optimal tau by Youden
                'tau_f1':     float  optimal tau by F1
        """
        fall_trajs = [t for t in self.trajectories if t.is_fall]
        safe_trajs = [t for t in self.trajectories if not t.is_fall]

        # Build frame arrays
        def _get_frames(traj, is_fall):
            arr = traj.q_rand_series if key == 'q_rand' else traj.q_series
            if is_fall:
                return arr[-H_fall:] if len(arr) >= H_fall else arr
            else:
                return arr[:safe_max_f]

        fall_frames = np.concatenate([_get_frames(t, True)  for t in fall_trajs]) if fall_trajs else np.array([])
        safe_frames = np.concatenate([_get_frames(t, False) for t in safe_trajs]) if safe_trajs else np.array([])

        # filter out non-finite
        fall_frames = fall_frames[np.isfinite(fall_frames)]
        safe_frames = safe_frames[np.isfinite(safe_frames)]

        taus = np.linspace(tau_range[0], tau_range[1], n_pts)
        fprs, tprs, f1s = [], [], []
        n_fall = len(fall_frames)
        n_safe = len(safe_frames)
        for tau in taus:
            if n_fall > 0:
                tp = int(np.sum(fall_frames <= tau))
                fn = n_fall - tp
                tpr = float(tp / n_fall)
            else:
                tp, fn = 0, 0
                tpr = float('nan')

            if n_safe > 0:
                fp = int(np.sum(safe_frames <= tau))
                fpr = float(fp / n_safe)
            else:
                fp = 0
                fpr = float('nan')

            # 标准帧级 F1：2TP / (2TP + FP + FN)
            den = (2 * tp + fp + fn)
            f1 = float((2 * tp) / den) if den > 0 else float('nan')

            tprs.append(tpr)
            fprs.append(fpr)
            f1s.append(f1)

        fprs = np.array(fprs)
        tprs = np.array(tprs)
        youden = tprs - fprs
        f1 = np.array(f1s)

        finite_y = np.isfinite(youden)
        finite_f1 = np.isfinite(f1)
        best_y = int(np.nanargmax(youden)) if finite_y.any() else None
        best_f = int(np.nanargmax(f1)) if finite_f1.any() else None

        tau_youden = float(taus[best_y]) if best_y is not None else float('nan')
        fpr_at_youden = float(fprs[best_y]) if best_y is not None else float('nan')
        tpr_at_youden = float(tprs[best_y]) if best_y is not None else float('nan')

        tau_f1 = float(taus[best_f]) if best_f is not None else float('nan')
        fpr_at_f1 = float(fprs[best_f]) if best_f is not None else float('nan')
        tpr_at_f1 = float(tprs[best_f]) if best_f is not None else float('nan')
        return {
            'taus':       taus,
            'fpr':        fprs,
            'tpr':        tprs,
            'youden':     youden,
            'f1':         f1,
            'tau_youden': tau_youden,
            'fpr_at_youden': fpr_at_youden,
            'tpr_at_youden': tpr_at_youden,
            'tau_f1':     tau_f1,
            'fpr_at_f1':  fpr_at_f1,
            'tpr_at_f1':  tpr_at_f1,
            'n_fall_frames': len(fall_frames),
            'n_safe_frames': len(safe_frames),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CriticSafetyFilter 主类
# ─────────────────────────────────────────────────────────────────────────────
class CriticSafetyFilter:
    """
    Q(z, cmd) 安全过滤器  —  用 Critic 评估 cmd 的未来安全性

    部署流程 (每个 sim step):
        1. env.step(joint_actions) → obs_dict
        2. filter.update(obs_dict)            # 编码 obs → latent z
        3. q = filter.evaluate(cmd)           # Q(z, cmd) - 正=安全, 负=危险
        4. if q < threshold: cmd = fallback   # 停下来或换 cmd
        5. 把 cmd 写回 env.command_generator

    关键参数:
        wm_path:    世界模型 checkpoint  (rssm_ckpt.pt)
        critic_path: DDPG Critic checkpoint (epoch_id_20/policy.pth)
        threshold:  安全阈值 (默认 0.0)
    """

    # 默认 checkpoint 路径 (noStumble no-HVR, no-detach WM, step_14999, ep19)
    # 切换到其他 run 时更新这三个常量，或用 CLI --wm_path / --critic_path / --gx_tau 覆盖
    DEFAULT_WM = str(
        _LATENT_SAFETY_ROOT
        # / "logs/go2_noStumble_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip00_30" 
        # / "rssm_ckpt_14999.pt"
        # / "logs/go2_noStumble_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale" 
        # / "rssm_ckpt_14999.pt"
        # / "logs/go2_push_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale" 
        # / "rssm_ckpt_14999.pt"
        # / "logs/go2_push_2300ep_finetune"
        # / "rssm_ckpt.pt"
        # / "logs/go2_0328_push"
        # / "logs/go2_0328_nopush_2900ep"
        # /"logs/go2_0328_ls_verge_2900ep"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06"
        # /"logs/go2_vel_failure"
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06"
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06_strat_tail64"
        # Phase1 
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10"
        # Phase2 lowfric x brake x pit训练的wm
        /"logs/go2_lowfric_PitAdded_3840ep_stage2_wm_fric&vel_recon/"
        / "rssm_ckpt.pt"
    )
    DEFAULT_CRITIC = str(
        _LATENT_SAFETY_ROOT
        # / "logs/go2_noStumble_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_ddpg/PyHJ/0324/161004_noStumble_nodetach_depthIm_Clip/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_19/policy.pth"
        # / "logs/go2_noStumble_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale_ddpg/PyHJ/0328/060723_noStumble_endzone_fallOnly_noWarmup_randTarget/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_1/policy.pth"
        # / "logs/go2_noStumble_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale_ddpg/PyHJ/0406/013815_endzone15_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_15/policy.pth"
        # / "logs/go2_push_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale_ddpg/PyHJ/0415/001648_push3000ep_endzone15_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_7/policy.pth"
        # / "logs/go2_push_2000ep_prefall5_gamma075_15000_pretrain2500_dImage_clip_fix_scale_ddpg/PyHJ/0417/011304_push3000ep_endzone15_randtgt_anchor1p0_lowspeed40/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_push_2300ep_finetune_ddpg/PyHJ/0417/053204_push2300ep_endzone15_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/" 
        # / "epoch_id_15/policy.pth"
        # / "logs/go2_0328_push_ddpg/PyHJ/0426/000654_push2000ep_0328_endzone40_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_nopush_ddpg/PyHJ/0425/211452_nopush2000ep_0328_endzone15_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_push_ddpg/PyHJ/0427/143819_push_endzone60_begin40_randtgt/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_push_ddpg/PyHJ/0427/164705_push_endzone60_begin40_frac10_randtgt/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_push_ddpg/PyHJ/0427/183532_push_endzone60_begin40_highspeed15/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/" 
        # / "logs/go2_0328_push_ddpg/PyHJ/0427/230345_push_endzone60_begin40_alltrajs/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_push_ddpg/PyHJ/0428/174453_push_endzone60_begin40_frac004_alltrajs/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        
        # /"logs/go2_0328_nopush_ddpg/PyHJ/0501/213058_nopush200ep_0328_15endzone_07alltraj_07lowspeed08_randtgt_anchor1p0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_nopush_2600ep/PyHJ/0508/100419_endzone40_lowspeed40_mixed20/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_5/policy.pth"
        # /"/home/lcy/latent-safety/logs/go2_0328_nopush_2900ep_ddpg/PyHJ/0509/212638_endzone40_lowspeed40_mixed20_0328/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_nopush_2900ep_ddpg/PyHJ/0510/165802_endzone25_lowspeed40_mixed20_0328/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_ls_verge_2900ep_ddpg/PyHJ/0513/003813_endzone40_lowspeed40_mixed20_0328_verge/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered_ddpg/PyHJ/0513/205902_endzone25_lowspeed40_mixed20_0328_newlabel/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "logs/go2_0328_nopush_2600ep_gz_filtered_ddpg/PyHJ/0520/194623_nopush_endzone60_begin40_frac03_alltrajs/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_15/policy.pth"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-09_sus03_ddpg/PyHJ/0521/004516_nopush_endzone60_begin40_frac03_alltrajs/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0522/015602_nopush_endzone40_begin40_nomixed/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0522/043203_nopush_endzone40_begin40_nomixed/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0522/055334_nopush_endzone70_begin10_mixed20/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0522/091250_nopush_endzone70_frac25_begin20_frac15_mixed10/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0522/190420_nopush_endzone70_frac25_begin30_frac20_mixed0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_nopush_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0523/103546_nopush_endzone80_frac25_begin20_frac20_mixed0/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_vel_failure/PyHJ/0530/175458_vel_failure_fz40_rt/PyHJ/go2-wm/wm_trainable_actor_ReLU_critic_ReLU_gd_5_tau_0.005_train_1_buf_100000_c256-3_a256-3_gamma_0.995/noise_0.25_alr_0.0001_clr_0.0003_batch_512_spe_40000_kwargs__seed_0/"
        # /"epoch_id_12/policy.pth"
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0608/143806_nopush_endzone70_begin30_nomixed_brake+stop_target/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"epoch_id_12/policy.pth"
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0609/233911_nopush_endzone70_frac30_brake_k=6_begin30_frac20_brake_stopreset+stop_target/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"epoch_id_12/policy.pth" 
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06_ddpg/PyHJ/0610/034202_nopush_endzone70_frac30_brake_begin30_frac20_stop_target_gamma099999/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.99999/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"epoch_id_7/policy.pth"
        # /"logs/go2_0328_nopush_brake_2600ep_gz_filtered-085_sus05_warmup06_strat_tail64_ddpg/PyHJ/0610/225012_nopush_endzone70_frac30_brake_begin30_frac20_brake_stopreset+stop_target_stratified_wm/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.99999/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0611/172513_nopush_endzone70_frac30_brake_begin30_frac20_stop_target_stratified_brake/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.99999/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"epoch_id_7/policy.pth"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0617/173108_nopush_endzone80_frac30_brake30_begin20_frac20_brake_stopreset+vstop_post_qbrake_r1/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0617/224437_nopush_endzone80_frac30_brake30_begin20_frac20_brake_stopreset+vstop_post_qbrake_r10/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_11/policy.pth"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0620/003030_nopush_endzone70_frac30_begin30_frac20_brake_cali_v_stop_post_qbrake_r10/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_12/policy.pth"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0624/221927_nopush_endzone70_frac30_ls20_rest_uniform_qbrake_r10_vstopV2/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0625/184315_nopush_z0v2_failwin70_hi50_lo20_begin30_qbrake_r10_vstopV2/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # Phase1 利用 brake x flat + walk x allterr训练的Vstop
        # /"logs/go2_0328_brake_newLabel_2400ep_fall40_brake25_safeearly15_saferand10_fallrand10_ddpg/PyHJ/0625/213757_nopush_z0v2_hisafe_hf35_hs15_lo20_begin30_qbrake_r10_vstopV2/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        # / "epoch_id_11/policy.pth"
        # Phase2 迁移中加入lowfric x brake x allterr训练的Vstop
        /"logs/go2_ddpg_lowfric_PitAdded/PyHJ/0712/233613_lowfric_pitAdded_z0v2_failwin70_hi50_lo20_begin30_qbrake_r10_vstopAllterrReal/PyHJ/go2-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_5_tau_0.005_training_num_1_buffer_size_100000_c_net_256_3_a1_256_3_gamma_0.995/noise_0.3_actor_lr_0.0001_critic_lr_0.0004_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/"
        / "epoch_id_11/policy.pth"
        
               
    )
    # gx_tau = tau_Youden from noStumble no-detach WM (rssm_ckpt_14999.pt)
    # raw_margin < 0.4627  →  g_x = tanh(raw - 0.4627) < 0  →  unsafe
    # DEFAULT_GX_TAU = -0.1945 #截至2026-04-01
    DEFAULT_GX_TAU = 0.0 #0.0818 #0.278
    

    def __init__(
        self,
        wm_path: str = None,
        critic_path: str = None,
        threshold: float = 0.0,
        gx_tau: float = None,
        device: str = "cuda:0",
        env: Optional[object] = None,
        obs_state_dim: int = 42,
        adaptive_k: float = None,
        adaptive_burnin: int = 200,
        v_stop_path: str = None,
        diagnostic_repeat: int = 1,
        diagnostic_gamma: float = 0.995,
    ):
        self.device = device
        self.threshold = threshold
        self.gx_tau = gx_tau if gx_tau is not None else self.DEFAULT_GX_TAU

        # ── 自适应阈值 ───────────────────────────────────────────────
        # adaptive_k 不为 None 时启用: tau = mean(Q(z,cmd)) - k * std(Q(z,cmd))
        # 前 adaptive_burnin 步只收集统计，不触发过滤
        self.adaptive_k = adaptive_k
        self.adaptive_burnin = adaptive_burnin
        self._q_history: deque = deque(maxlen=5000)  # 滚动窗口，收集 Q(z,cmd)
        self._adaptive_ready = False
        self._adaptive_tau = threshold  # 在 burn-in 期间用固定 threshold
        self.env = env
        self.obs_state_dim = obs_state_dim
        self.diagnostic_repeat = max(1, int(diagnostic_repeat))
        self.diagnostic_gamma = float(diagnostic_gamma)
        self.v_stop_model = None

        wm_path = wm_path or self.DEFAULT_WM
        critic_path = critic_path or self.DEFAULT_CRITIC

        # ── 加载世界模型 ──────────────────────────────────────────────
        self._load_world_model(wm_path)

        # ── 加载 Critic ──────────────────────────────────────────────
        self._load_critic(critic_path)
        if v_stop_path is not None:
            self._load_v_stop(v_stop_path)

        # ── 内部状态 ─────────────────────────────────────────────────
        num_envs = env.num_envs if env is not None else 1
        self.num_envs = num_envs

        self.latent = None
        self.rssm_state = None                    # (N, ...) 无时间维的 posterior state
        self.feat = None                          # (N, 1, 1536)
        # Per-env is_first 标志（RSSM reset 依赖）
        self.is_first_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        # Per-env 上一步 cmd（物理量，匹配 WM 训练分布）
        self.last_action = torch.zeros(num_envs, 3, device=device)

        self.stats = CriticFilterStats(num_envs=num_envs, threshold=threshold)
        self._diag_step = 0   # 首步诊断计数器

        print(f"[CriticFilter] ✓ Initialized")
        if self.adaptive_k is not None:
            print(f"  threshold = ADAPTIVE (k={self.adaptive_k}, burnin={self.adaptive_burnin})")
        else:
            print(f"  threshold = {threshold}")
        print(f"  gx_tau    = {self.gx_tau}")
        print(f"  wm device = {self.device}")
        print(f"  cmd space = {CMD_LOW} ~ {CMD_HIGH}")
        print(f"  wm action history = physical cmd (not normalized)")
        if self.v_stop_model is not None:
            print(f"  Q target diagnostic = repeat {self.diagnostic_repeat}, "
                  f"gamma={self.diagnostic_gamma}")

    # ─── WM 加载 ────────────────────────────────────────────────────────
    def _load_world_model(self, path: str):
        import models
        import gymnasium as gym

        config = self._make_wm_config()
        config.device = self.device

        obs_space = gym.spaces.Dict({
            'image':       gym.spaces.Box(0, 255, (64, 64, 1), np.uint8),
            'obs_state':   gym.spaces.Box(-np.inf, np.inf, (self.obs_state_dim,), np.float32),
            'is_first':    gym.spaces.Box(0., 1., (1,)),
            'is_last':     gym.spaces.Box(0., 1., (1,)),
            'is_terminal': gym.spaces.Box(0., 1., (1,)),
        })
        act_space = gym.spaces.Box(-1., 1., (3,), np.float32)

        # num_actions is not in configs.yaml — set from act_space (same as dreamer.py:248)
        config.num_actions = act_space.n if hasattr(act_space, 'n') else act_space.shape[0]
        self.wm = models.WorldModel(obs_space, act_space, 0, config)
        ckpt = torch.load(path, map_location=self.device)
        sd = {k[14:]: v for k, v in ckpt['agent_state_dict'].items() if '_wm' in k}
        self.wm.load_state_dict(sd, strict=False)
        self.wm.eval().to(self.device)
        print(f"[CriticFilter] WM loaded from {path}")

    def _make_wm_config(self):
        """从 configs.yaml 加载 go2_ddpg_wm config"""
        import ruamel.yaml as yaml
        from argparse import Namespace

        yml = yaml.YAML(typ="safe", pure=True)
        cfg_path = _LATENT_SAFETY_ROOT / "configs.yaml"
        all_cfgs = yml.load(cfg_path.read_text())

        merged = {}
        for section in ['defaults', 'go2', 'go2_ddpg_wm']:
            if section in all_cfgs:
                self._deep_update(merged, all_cfgs[section])

        ns = Namespace(**merged)
        ns.device = self.device
        return ns

    @staticmethod
    def _deep_update(base: dict, update: dict):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                CriticSafetyFilter._deep_update(base[k], v)
            else:
                base[k] = v

    # ─── Critic 加载 ────────────────────────────────────────────────────
    def _load_critic(self, path: str):
        from PyHJ.utils.net.common import Net
        from PyHJ.utils.net.continuous import Critic

        feat_size = 1536  # 32*32 + 512
        net = Net(
            (1, 1, feat_size), 3,
            hidden_sizes=[256, 256, 256],
            activation=torch.nn.ReLU,
            concat=True,
            device=self.device,
        )
        self.critic = Critic(net, device=self.device).to(self.device)

        sd = torch.load(path, map_location=self.device)
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        critic_sd = {
            k.replace('critic.', '', 1): v
            for k, v in sd.items()
            if k.startswith('critic.') and 'critic_old' not in k
        }
        self.critic.load_state_dict(critic_sd)
        self.critic.eval()
        print(f"[CriticFilter] Critic loaded from {path}")

    def _load_v_stop(self, path: str):
        from scripts.v_stop_utils import load_v_stop_model

        self.v_stop_model, ckpt = load_v_stop_model(path, self.device)
        self.v_stop_model.eval()
        metadata = ckpt.get("metadata", {})
        convention = metadata.get("label_convention", "unknown")
        if convention != "post":
            print(f"[CriticFilter] WARNING: V_stop label_convention={convention!r}; "
                  "Q brake target diagnostics expect 'post'.")
        print(f"[CriticFilter] V_stop loaded from {path} "
              f"(label_convention={convention})")

    # ─── 公开 API ──────────────────────────────────────────────────────
    def reset(self, env_ids=None):
        """
        重置 RSSM 状态。

        Args:
            env_ids: 要重置的 env 索引（Tensor / list）。
                     None → 重置所有 env。
        """
        if env_ids is None:
            self.is_first_mask[:] = True
            self.last_action[:] = 0.0
            self.rssm_state = None
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu()
            self.is_first_mask[env_ids] = True
            self.last_action[env_ids] = 0.0
        # latent/feat 不需要显式清除——is_first_mask 会在下次 update() 时
        # 通过 RSSM observe 的 is_first 信号完成状态重置

    def finalize_episode(self, env_ids, is_fall):
        """
        Episode 结束时调用：提交轨迹统计并重置对应 env 的 RSSM 状态。

        Args:
            env_ids: 结束的 env 索引（Tensor / list / 单个 int）
            is_fall: 对应每个 env 是否为摔倒结束（Tensor[bool] / list[bool] / 单个 bool）
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.cpu().tolist()
        elif isinstance(env_ids, int):
            env_ids_list = [env_ids]
        else:
            env_ids_list = list(env_ids)

        if isinstance(is_fall, torch.Tensor):
            is_fall_list = is_fall.cpu().tolist()
        elif isinstance(is_fall, (bool, np.bool_)):
            is_fall_list = [bool(is_fall)] * len(env_ids_list)
        else:
            is_fall_list = [bool(f) for f in is_fall]

        self.stats.finalize_episode(env_ids_list, is_fall_list)
        self.reset(env_ids_list)

    @torch.no_grad()
    def update(self, obs_dict: Dict[str, torch.Tensor]):
        """
        编码当前 obs → 更新 latent state z

        Args:
            obs_dict: LeggedLab 环境的观测字典
                包含 'policy' (obs state tensor) + 相机深度图 (从 env 获取)
        """
        data = self._preprocess_obs(obs_dict)
        processed = self.wm.preprocess(data)
        embed = self.wm.encoder(processed)
        prev_state = self.rssm_state
        self.latent, _ = self.wm.dynamics.observe(
            embed, data['action'], processed['is_first'], state=prev_state
        )
        # 清除所有 env 的 is_first（已完成首帧编码）
        self.is_first_mask[:] = False

        # 记录无时间维的 posterior 作为下一个 step 的 RSSM 初始状态
        self.rssm_state = {k: v[:, -1] for k, v in self.latent.items()}

        latent_last = {k: v[:, -1:] for k, v in self.latent.items()}
        self.feat = self.wm.dynamics.get_feat(latent_last)  # (N, 1, 1536)

    @torch.no_grad()
    def get_g(self) -> torch.Tensor:
        """获取 g(x) 单帧安全值  (N,)  —  tanh(margin - gx_tau)"""
        assert self.feat is not None, "call update() first"
        return torch.tanh(self.wm.heads['margin'](self.feat) - self.gx_tau).squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, cmd_physical: torch.Tensor) -> torch.Tensor:
        """
        Q(z, cmd)  — 正 = 安全, 负 = 危险

        Args:
            cmd_physical: (N, 3) 物理 cmd [vx, vy, yaw]
        Returns:
            q: (N,) 安全价值
        """
        assert self.feat is not None, "call update() first"
        # normalize cmd to [-1, 1]
        cmd_np = cmd_physical.cpu().numpy() if isinstance(cmd_physical, torch.Tensor) else cmd_physical
        cmd_norm = cmd_to_normalized(cmd_np)
        cmd_t = torch.tensor(cmd_norm, dtype=torch.float32, device=self.device)
        if cmd_t.ndim == 1:
            cmd_t = cmd_t.unsqueeze(0)

        feat_2d = self.feat.squeeze(1)  # (N, 1536)
        q = self.critic(feat_2d, cmd_t).squeeze(-1)  # (N,)
        return q

    @torch.no_grad()
    def evaluate_brake_target(
        self,
        cmd_physical: torch.Tensor,
        repeat: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Explicit Q_brake target from the current posterior latent.

        Rolls the physical command for R frames in the frozen WM, folds
        g(z_1)..g(z_R), then attaches V_stop_post(z_R). This mirrors the
        DDPG target used with n_step=1 and brake_action_repeat=R.
        """
        if self.v_stop_model is None:
            raise RuntimeError("V_stop is required for Q target diagnostics")
        if self.rssm_state is None:
            raise RuntimeError("call update() before evaluate_brake_target()")

        repeat = self.diagnostic_repeat if repeat is None else max(1, int(repeat))
        gamma = self.diagnostic_gamma if gamma is None else float(gamma)
        cmd = torch.as_tensor(cmd_physical, dtype=torch.float32, device=self.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)

        actions = cmd[:, None, :].repeat(1, repeat, 1)
        prior = self.wm.dynamics.imagine_with_action(actions, self.rssm_state)
        feats = self.wm.dynamics.get_feat(prior)
        g_seq = torch.tanh(
            self.wm.heads["margin"](feats).squeeze(-1) - self.gx_tau
        )
        v_stop_tail = self.v_stop_model(feats[:, -1]).reshape(-1)

        target = v_stop_tail
        for j in range(repeat - 1, -1, -1):
            gj = g_seq[:, j]
            target = (1.0 - gamma) * gj + gamma * torch.minimum(gj, target)

        g_min, g_argmin = g_seq.min(dim=1)
        return {
            "q_target": target,
            "g_seq": g_seq,
            "g_min": g_min,
            "g_argmin": g_argmin + 1,
            "v_stop_tail": v_stop_tail,
            "tail_is_bottleneck": v_stop_tail < g_min,
        }

    @torch.no_grad()
    def evaluate_rand_avg(self, n_rand: int = 20) -> torch.Tensor:
        """
        Q_avg(z) = 对 n_rand 个均匀随机动作取均值  — 与 ablation_hvr_q.py 完全一致。

        ablation 的 tau_opt (Youden) 是在此定义下得到的，想直接套用 ablation 的 tau
        应使用此方法代替 evaluate(actual_cmd)，代价是失去 action-conditioned 的优势。

        二者对比含义：
          evaluate(cmd) 低  而  evaluate_rand_avg() 高
            → 当前这个 cmd 本身危险，但状态还有安全的动作
          evaluate(cmd) 低  且  evaluate_rand_avg() 低
            → 状态本身已经危险，无论什么动作都无法挽救
        """
        assert self.feat is not None, "call update() first"
        feat_2d = self.feat.squeeze(1)  # (N, 1536)
        N = feat_2d.shape[0]
        q_acc = torch.zeros(N, device=self.device)
        for _ in range(n_rand):
            a = torch.FloatTensor(N, 3).uniform_(-1, 1).to(self.device)
            q_out = self.critic(feat_2d, a)
            if isinstance(q_out, tuple):
                q_out = q_out[0]
            q_acc += q_out.squeeze(-1)
        return q_acc / n_rand

    @torch.no_grad()
    def evaluate_multi(self, cmds: torch.Tensor) -> torch.Tensor:
        """
        对同一状态评估多个 cmd 的安全性 (Q 的核心优势)

        Args:
            cmds: (M, 3) 物理 cmd
        Returns:
            q: (M,) 每个 cmd 的安全价值
        """
        assert self.feat is not None, "call update() first"
        M = cmds.shape[0]
        feat_2d = self.feat[:, 0, :].expand(M, -1)  # (M, 1536)
        cmd_np = cmds.cpu().numpy() if isinstance(cmds, torch.Tensor) else cmds
        cmd_norm = cmd_to_normalized(cmd_np)
        cmd_t = torch.tensor(cmd_norm, dtype=torch.float32, device=self.device)
        q = self.critic(feat_2d, cmd_t).squeeze(-1)  # (M,)
        return q

    # ── [LOOK-AHEAD ADD] ────────────────────────────────────────────────────────
    # evaluate_lookahead(): 用 WM transition model 显式向前一步，再评估 Q。
    # 对应 Nakamura 2025 Eq.7/9 的部署方式：Q(ẑ_{t+1}, cmd)。
    #
    # 与 evaluate() 的语义区别：
    #   evaluate()         → Q(z_t, cmd)     评估"当前状态 + 这个 cmd"
    #   evaluate_lookahead → Q(ẑ_{t+1}, cmd) 评估"执行 cmd 之后的状态"
    #
    # 为什么一步 look-ahead 等价于多一次 Bellman backup（DP 视角）：
    #   Q 通过 Bellman backup 压缩了 t+1 → ∞ 的可达性信息。
    #   显式做一步 img_step 相当于：
    #     Q_1(z_t, a) = min{ ℓ(z_t),  Q(img_step(z_t, a), ·) }
    #   这正是对已收敛的 Q 再做一次 Bellman 展开。
    #   在 Q 未完全收敛时，这一步可以补偿训练期间的近似误差。
    #   （详见 docs/WM_SAFETY_FILTER_NOTES.md §8）
    # ────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_lookahead(self, cmd_physical: torch.Tensor) -> torch.Tensor:
        """
        Q(ẑ_{t+1}, cmd) — 执行 cmd 后下一状态的安全价值  (Nakamura 2025 对齐)

        调用前须先调用 update()，使 self.rssm_state 为当前后验状态。

        Args:
            cmd_physical: (N, 3) 物理 cmd [vx, vy, yaw]
        Returns:
            q: (N,) 安全价值（正=安全, 负=危险）
        """
        assert self.rssm_state is not None, "call update() first"

        cmd_np = cmd_physical.cpu().numpy() if isinstance(cmd_physical, torch.Tensor) else cmd_physical
        cmd_norm = cmd_to_normalized(cmd_np)
        cmd_t = torch.tensor(cmd_norm, dtype=torch.float32, device=self.device)
        if cmd_t.ndim == 1:
            cmd_t = cmd_t.unsqueeze(0)

        # [LOOK-AHEAD ADD] 调用 transition model 向前一步（无需真实观测）
        # img_step(prev_state_dict, prev_action) → next_state_dict
        next_state = self.wm.dynamics.img_step(self.rssm_state, cmd_t)

        # get_feat 需要时间维，临时 unsqueeze 后再 squeeze 回来
        next_latent = {k: v.unsqueeze(1) for k, v in next_state.items()}
        next_feat = self.wm.dynamics.get_feat(next_latent).squeeze(1)  # (N, 1536)

        q = self.critic(next_feat, cmd_t).squeeze(-1)  # (N,)
        return q
    # ── [LOOK-AHEAD ADD END] ─────────────────────────────────────────────────

    def filter_cmd(
        self,
        obs_dict: Dict[str, torch.Tensor],
        cmd_physical: torch.Tensor,
        apply_filter: bool = True,
        cmd_for_history: Optional[torch.Tensor] = None,
        compute_q_rand: bool = False,
        use_lookahead: bool = False,  # [LOOK-AHEAD ADD] True → Q(ẑ_{t+1}, cmd)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        核心方法: 安全过滤 cmd  (批量, 每个 env 独立)

        触发信号: Q(z, cmd_physical)  — 直接回答"执行当前指令是否安全"。
        q_rand 仅在 compute_q_rand=True 时计算，用于 diagnostic 对比，不影响触发逻辑。

        Args:
            use_lookahead: True → 用 evaluate_lookahead() 替代 evaluate()，
                           显式调用 WM transition model 前向一步后再评估 Q。
                           对齐 Nakamura 2025 Eq.7/9，激活 WM 的 dynamics 部分。

        Returns:
            cmd_out:  (N, 3) 过滤后的 cmd
            is_safe:  (N,)   True = 安全  [基于 Q(z, cmd)]
            q_values: (N,)   Q(z, cmd_physical) 或 Q(ẑ_{t+1}, cmd) — 触发信号
            g_values: (N,)   g(x) 单帧
            q_rand:   (N,)   Q_avg(z, rand) — diagnostic; 全零当 compute_q_rand=False
        """
        # ── Bug fix: 先设 last_action = a_t（当前 cmd），再调用 update ──────
        # 训练约定: observe(embed[t], action[t]=a_t) — 当前步动作配当前帧。
        # 原实现在 update() 之后才设 last_action，RSSM 拿到 a_{t-1}（错位一帧）。
        hist_cmd = cmd_for_history if cmd_for_history is not None else cmd_physical
        self.last_action = hist_cmd.detach().to(self.device).float().clone()

        self.update(obs_dict)   # _preprocess_obs 此时读到 last_action = a_t ✓
        g_values = self.get_g()
        # [LOOK-AHEAD ADD] 根据 use_lookahead 选择评估方式
        q_values = (self.evaluate_lookahead(cmd_physical) if use_lookahead
                    else self.evaluate(cmd_physical))   # Q(z/ẑ, cmd) — 触发信号
        q_rand    = self.evaluate_rand_avg() if compute_q_rand else q_values.new_zeros(q_values.shape)
        # q_rand 仅供 diagnostic 对比（compute_q_rand=True 时），不参与触发逻辑。

        # ── 确定触发阈值和触发信号 ──────────────────────────────────
        # 触发用 Q(z, cmd)：直接回答"执行当前指令是否安全"。
        # q_rand（随机动作均值）仅在离线 eval_q 没有真实指令时才作触发信号；
        # sim/实机已知指令，用 q_rand 会丢失 action-conditioned 的区分能力。
        # 注意：此处换用 q_values 后，台下离线 tau 需通过 --eval_terrain sim
        # 重标定（sim tau sweep 存的也是 q_vals，因此完全自洽）。
        trigger_q = q_values

        if self.adaptive_k is not None:
            # 收集 Q(z, cmd) 统计，用于自适应阈值
            for v in q_values.cpu().numpy().tolist():
                self._q_history.append(v)
            if len(self._q_history) >= self.adaptive_burnin:
                arr = np.array(self._q_history)
                mu, sigma = float(np.mean(arr)), float(np.std(arr))
                self._adaptive_tau = mu - self.adaptive_k * sigma
                if not self._adaptive_ready:
                    self._adaptive_ready = True
                    print(f"[CriticFilter] Adaptive threshold ready: "
                          f"mu={mu:.4f} std={sigma:.4f} "
                          f"tau={self._adaptive_tau:.4f} (k={self.adaptive_k})")
            current_tau = self._adaptive_tau
        else:
            current_tau = self.threshold

        is_safe = trigger_q >= current_tau  # (N,)

        # fallback: 最低速，不转弯  [vx_min, 0, 0]
        fallback_cmd = torch.tensor(
            [CMD_LOW[0], 0.0, 0.0], dtype=torch.float32, device=self.device
        ).expand_as(cmd_physical)

        if apply_filter:
            cmd_out = torch.where(is_safe.unsqueeze(-1), cmd_physical, fallback_cmd)
        else:
            # monitor-only: 返回原命令，确保 RSSM 历史动作与环境实际执行一致
            cmd_out = cmd_physical

        # last_action already set at function start to a_t
        # stats: 每步 per-env 向量（不做平均）
        self.stats.log_step(
            g_per_env=g_values.cpu().numpy(),
            q_per_env=q_values.cpu().numpy(),
            q_rand_per_env=q_rand.cpu().numpy(),
        )

        return cmd_out, is_safe, q_values, g_values, q_rand

    # ─── 预处理 ────────────────────────────────────────────────────────
    def _preprocess_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        LeggedLab obs → WM 输入格式

        深度图: 从 env.scene.sensors['front_camera'] 获取
        状态:   从 obs_dict['policy'] 取最后一帧 (去掉 history stacking)
        """
        # ── 深度图 ──────────────────────────────────────────
        depth = self._get_depth_image()
        N = depth.shape[0]

        # 统一成 (N,H,W) 后，使用与训练一致的 percentile 预处理
        depth_np = depth.detach().cpu().numpy()
        if depth_np.ndim == 4:
            if depth_np.shape[1] == 1:       # (N,1,H,W)
                depth_np = depth_np[:, 0]
            elif depth_np.shape[-1] == 1:    # (N,H,W,1)
                depth_np = depth_np[..., 0]
            else:
                raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
        elif depth_np.ndim != 3:
            raise ValueError(f"Unexpected depth ndim: {depth_np.ndim}, shape={depth_np.shape}")

        depth_proc = np.stack(
            [preprocess_depth_like_training(depth_np[i]) for i in range(N)],
            axis=0,  # (N,H,W,1)
        )
        depth = torch.from_numpy(depth_proc).unsqueeze(1).to(self.device)  # (N,1,H,W,1)

        # ── 状态 ────────────────────────────────────────────
        # LeggedLab policy obs per-frame layout (45D):
        #   [0:3]  ang_vel
        #   [3:6]  projected_gravity
        #   [6:9]  command   ← NOT in WM training obs (42D), must be stripped
        #   [9:21] joint_pos
        #   [21:33] joint_vel
        #   [33:45] last_action
        # WM was trained with 42D (no command), so frame_dim = obs_state_dim + CMD_DIM
        CMD_DIM = 3           # velocity command dims appended by LeggedLab
        CMD_START = 6         # command starts at index 6 in each raw frame
        frame_dim = self.obs_state_dim + CMD_DIM  # 42 + 3 = 45

        obs_state = obs_dict.get('policy', obs_dict.get('obs'))
        total_dim = obs_state.shape[-1]

        if total_dim % frame_dim == 0 and total_dim > frame_dim:
            # history-stacked with command: (N, hist * 45) → (N, 45) last frame
            hist_len = total_dim // frame_dim
            obs_state = obs_state.view(N, hist_len, frame_dim)[:, -1, :]   # (N, 45)
        elif total_dim == frame_dim:
            # single frame with command: (N, 45)
            pass  # obs_state stays (N, 45)
        elif total_dim % self.obs_state_dim == 0 and total_dim > self.obs_state_dim:
            # history-stacked without command: (N, hist * 42)
            hist_len = total_dim // self.obs_state_dim
            obs_state = obs_state.view(N, hist_len, self.obs_state_dim)[:, -1, :]  # (N, 42)
        # else: already (N, 42) or similar

        # Strip command if frame has 45D
        if obs_state.shape[-1] == frame_dim:
            obs_state = torch.cat([obs_state[:, :CMD_START],
                                   obs_state[:, CMD_START + CMD_DIM:]], dim=-1)  # (N, 42)

        # Final safety clip to obs_state_dim
        if obs_state.shape[-1] != self.obs_state_dim:
            obs_state = obs_state[:, :self.obs_state_dim]

        # ── 首步诊断日志（第1步 + 每500步打印一次）─────────────
        self._diag_step += 1
        if self._diag_step == 1 or self._diag_step % 500 == 0:
            obs_raw = obs_dict.get('policy', obs_dict.get('obs'))
            depth_raw = self._get_depth_image()
            obs_s = obs_state[0].float().cpu()
            print(f"[CriticFilter DIAG step={self._diag_step}]")
            print(f"  obs raw: input_dim={obs_raw.shape[-1]}  after_strip={obs_s.shape[-1]}")
            print(f"    mean={obs_s.mean():.3f}  std={obs_s.std():.3f}  "
                  f"min={obs_s.min():.3f}  max={obs_s.max():.3f}")
            d = depth_raw[0].float().cpu()
            print(f"  depth raw: shape={tuple(depth_raw.shape[1:])}  "
                  f"mean={d.mean():.2f}m  min={d.min():.2f}m  max={d.max():.2f}m  "
                  f"nan={(~d.isfinite()).sum().item()}px")
            act0 = self.last_action[0].cpu().numpy()
            print(f"  last_action (physical): [{act0[0]:.3f}, {act0[1]:.3f}, {act0[2]:.3f}]")

        obs_state = obs_state.unsqueeze(1)  # (N, 1, D)

        # ── action (RSSM 需要上一步的 action) ──────────────
        action = self.last_action.unsqueeze(1)  # (N, 1, 3)

        # ── is_first（per-env 标志）────────────────────────
        is_first = self.is_first_mask.float().unsqueeze(1)  # (N, 1)

        return {
            'image':       depth.float().to(self.device),
            'obs_state':   obs_state.float().to(self.device),
            'action':      action.float().to(self.device),
            'is_first':    is_first,
            'is_last':     torch.zeros(N, 1, device=self.device),
            'is_terminal': torch.zeros(N, 1, device=self.device),
        }

    def _get_depth_image(self) -> torch.Tensor:
        """从 env.scene.sensors 获取深度图"""
        if self.env is not None:
            try:
                camera = self.env.scene.sensors['front_camera']
                depth = camera.data.output['distance_to_image_plane']
                return depth   # (N, H, W)
            except (KeyError, AttributeError):
                pass
        raise RuntimeError(
            "Cannot get depth image. "
            "Pass env= to CriticSafetyFilter() and ensure 'front_camera' sensor exists."
        )

    # ─── 打印统计 ──────────────────────────────────────────────────────
    def print_stats(self, monitor_only: bool = True):
        s = self.stats.summary()
        def _f(v, fmt):
            return (fmt % v) if (isinstance(v, float) and not np.isnan(v)) else "N/A"
        def _pct(v):
            """0-1 fraction → '56.4%' display"""
            return ('%.1f%%' % (v * 100)) if (isinstance(v, float) and not np.isnan(v)) else "N/A"

        print("\n" + "=" * 60)
        print("Critic Safety Filter Statistics")
        print("=" * 60)
        print(f"  Steps:            {s['total_steps']}")
        print(f"  Trajectories:     {s.get('total_trajs', 0)}  "
              f"(摔倒={s.get('fall_trajs', 0)}, 安全/超时={s.get('safe_trajs', 0)})")
        print(f"  Intervention %:   {s['intervention_rate']*100:.2f}%"
              f"  (步级别: 任意 env Q < threshold)")
        if self.adaptive_k is not None:
            if self._q_history:
                arr = np.array(self._q_history)
                mu, sig = float(np.mean(arr)), float(np.std(arr))
                print(f"  Adaptive tau:     {self._adaptive_tau:.4f}  "
                      f"(k={self.adaptive_k}, mu={mu:.4f}, sigma={sig:.4f}, n={len(arr)}, src=Q(z,cmd))")
            else:
                print(f"  Adaptive tau:     pending (burnin={self.adaptive_burnin})")
        print()
        print(f"  --- g(x) (单帧) ---")
        print(f"    mean: {_f(s.get('g_mean', float('nan')), '%.4f')}   "
              f"min: {_f(s.get('g_min', float('nan')), '%.4f')}   "
              f"<0: {_pct(s.get('g_negative_rate', float('nan')))}")
        print(f"  --- Q(z, cmd) (未来, actual cmd) ---")
        print(f"    mean: {_f(s.get('q_mean', float('nan')), '%.4f')}   "
              f"min: {_f(s.get('q_min', float('nan')), '%.4f')}   "
              f"<tau: {_pct(s.get('q_negative_rate', float('nan')))}")
        print(f"  --- Q_rand(z) (≡ablation: avg 20 random actions) ---")
        print(f"    mean: {_f(s.get('q_rand_mean', float('nan')), '%.4f')}   "
              f"<tau: {_pct(s.get('q_rand_negative_rate', float('nan')))}")
        print(f"  ↑ 如果 Q_rand 正常但 Q(cmd) 很负 → 是当前cmd本身危险，非状态危险")
        print()
        print(f"  --- 提前预警 (仅摔倒轨迹, 双方均报警才计入) ---")
        print(f"    Q 比 g 先报警的轨迹比:    {_pct(s.get('lead_q_before_g_rate', float('nan')))}")
        print(f"    平均提前步数 (全部轨迹):  {_f(s.get('lead_avg_steps', float('nan')), '%.1f 步')}")
        print(f"    平均提前步数 (Q先时):     {_f(s.get('lead_avg_steps_when_q_first', float('nan')), '%.1f 步')}")
        print(f"    Q 报警但 g 未报警比例:    {_pct(s.get('q_alarm_only_rate', float('nan')))}")
        print()
        print(f"  --- 假阳性率 ---")
        print(f"    [轨迹级] Q 曾报警的安全轨迹比:  {_pct(s.get('q_fpr_traj', float('nan')))}")
        print(f"    [轨迹级] g 曾报警的安全轨迹比:  {_pct(s.get('g_fpr_traj', float('nan')))}")
        print(f"    [帧级别] Q FPR (≡ablation):     {_pct(s.get('q_fpr_frame', float('nan')))}   "
              f"TPR: {_pct(s.get('q_tpr_frame', float('nan')))}   "
              f"Lead: {_f(s.get('q_lead_frame', float('nan')), '%.1f 帧')}")

        # ── K-filter sweep (post-hoc, 仅 monitor-only 模式有效) ──────────────
        # filter 介入后 Q_rand 序列已被改变（latent state 不同），post-hoc 重放无意义
        if monitor_only:
            kf = self.stats.kfilter_sweep()
            if any(v['n_fall'] > 0 or v['n_safe'] > 0 for v in kf.values()):
                print()
                print(f"  --- K-Filter Sweep (post-hoc, tau={self.threshold}, monitor-only) ---")
                print(f"  {'K':>3}  {'FPR_K':>7}  {'FNR_K':>7}  {'Lead_K':>8}  {'#safe':>5}  {'#fall':>5}")
                print(f"  {'-'*48}")
                for K, r in kf.items():
                    lead_str = ('%.1f f' % r['lead_K']) if not np.isnan(r['lead_K']) else '  N/A '
                    print(f"  {K:>3}  {r['fpr_K']*100:>6.1f}%  {r['fnr_K']*100:>6.1f}%  "
                          f"{lead_str:>8}  {r['n_safe']:>5}  {r['n_fall']:>5}")

        # ── Tau 扫描: 部署域最优阈值校准 ────────────────────────────
        ts = self.stats.tau_sweep(key='q_rand')
        if (
            ts['n_fall_frames'] > 0 and ts['n_safe_frames'] > 0
            and np.isfinite(ts.get('tau_youden', float('nan')))
            and np.isfinite(ts.get('tau_f1', float('nan')))
        ):
            print()
            print(f"  --- Tau Calibration (Q_rand, 帧级, 部署域最优阈值) ---")
            print(f"  基于 {ts['n_fall_frames']} fall帧 / {ts['n_safe_frames']} safe帧")
            print(f"  ★ tau_youden = {ts['tau_youden']:+.4f}  "
                  f"→ FPR={ts['fpr_at_youden']*100:.1f}%  TPR={ts['tpr_at_youden']*100:.1f}%")
            print(f"  ★ tau_F1     = {ts['tau_f1']:+.4f}  "
                  f"→ FPR={ts['fpr_at_f1']*100:.1f}%  TPR={ts['tpr_at_f1']*100:.1f}%")
            print(f"  提示: 下次运行加 --safety_threshold {ts['tau_youden']:.4f}")
        elif ts['n_fall_frames'] > 0 or ts['n_safe_frames'] > 0:
            print()
            print("  --- Tau Calibration (Q_rand) ---")
            print(f"  有效帧不足或全 NaN，跳过自动阈值：fall={ts['n_fall_frames']}, safe={ts['n_safe_frames']}")
        print("=" * 60 + "\n")
