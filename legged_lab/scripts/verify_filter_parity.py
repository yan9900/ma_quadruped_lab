#!/usr/bin/env python3
"""verify_filter_parity.py — 部署前离线对拍，**不需要启动 IsaacSim**

docs/66_Sim闭环部署改动清单.md 的 smoke 清单。三个检查：

  A  深度路径一致性（训练/离线 vs 部署）
     pkl 存的是 v3 采集器的 uint8 量化，但那只是**存储格式**。
     tools.py:474-512 的训练管线是：
         uint8 --depth_to_metres--> 米 --nan_to_num--> if max>1: /3.0 --> *255 --> uint8
     即 WM 实际见到的是**旧的线性 [0,3]→[0,255] 编码**。
     部署侧 preprocess_depth_like_training 对传感器给的**米**做同一串操作。
     ⇒ 两条路必须逐像素相等。这是唯一正确的深度判据。

  A2 误喂保护（tools.py:286 明写的坑）
     「Feeding a uint8 array into the legacy img/3.0 path silently saturates
       every pixel to 255 ... which is white noise, not a crash」
     所以要验：把**存储的 uint8** 当成米直接喂进去会烂成什么样。这是部署侧
     唯一真实的深度风险 —— 忘记 depth_to_metres。

  B  Q 对拍：逐帧 observe（部署路径） vs 批量 observe（离线评估路径）
     两条路数学上等价，但 obs_step 会**采样** stoch，所以必须先量噪声底：
     同一条批量路径换 RNG 跑两遍的 |ΔQ| 就是底噪。只有「逐帧 vs 批量」的 |ΔQ|
     落在底噪量级，才算部署路径 == 评估路径。
     （方法学同 longchain README ③。）

  D  critic 加载完整性：spectral-norm 自动识别 + strict 键匹配。

用法见文件末 main() 或 --help。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

LATENT_SAFETY = os.environ.get("LATENT_SAFETY_ROOT", "/home/lcy/latent-safety")
for p in (LATENT_SAFETY, os.path.join(LATENT_SAFETY, "dreamerv3-torch")):
    if p not in sys.path:
        sys.path.insert(0, p)

EXAM_DIRS = [
    f"/home/lcy/LeggedLab/legged_lab/data/fric_{p}_paired_{b}"
    for b in ("0.23-0.25", "0.39-0.41", "0.79-0.81")
    for p in ("E", "E2")
]


# ─────────────────────────────────────────────────────────────────────────────
# 深度 codec
# ─────────────────────────────────────────────────────────────────────────────
def quantize_depth_new(a, lo, hi):
    """collect_go2_data_v3.py:368-386 的 quantize_depth，逐字。"""
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    q = np.empty(a.shape, dtype=np.uint8)
    q[~finite] = 255                                   # 无回波
    if finite.any():
        v = np.clip(a[finite], lo, hi)
        q[finite] = np.round((v - lo) / max(hi - lo, 1e-9) * 254.0).astype(np.uint8)
    return q


def dequantize_new(q, lo, hi):
    """新编码 → 米。255 → inf（无回波）。"""
    q = np.asarray(q)
    out = np.full(q.shape, np.inf, dtype=np.float64)
    m = q != 255
    out[m] = lo + q[m].astype(np.float64) / 254.0 * (hi - lo)
    return out


def preprocess_depth_old(depth_hw):
    """critic_safety_filter.py:67-87 的 preprocess_depth_like_training，逐字。"""
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
# A — 深度编解码
# ─────────────────────────────────────────────────────────────────────────────
def check_A(trajs, n_stride=7, max_traj=60):
    """A  深度路径一致性 + A2 误喂保护。"""
    print("\n" + "=" * 78)
    print("A  深度路径一致性：训练/离线 prep_img  vs  部署 preprocess_depth_like_training")
    print("=" * 78)
    from scripts.vstop_Qbrake.v_stop_utils import prep_img
    from tools import depth_to_metres

    meta = trajs[0].get("depth_encoding")
    print(f"  depth_encoding（仅存储格式）: {meta}")
    print("  ⚠ 这是**存储**格式。WM 见到的是 depth_to_metres 之后再走旧线性编码的结果。")

    bad = tot = 0
    worst = 0
    sat_frac, sat_frames = [], 0
    for tr in trajs[:max_traj]:
        arr = np.asarray(tr["obs"]["image"])
        for t in range(0, len(arr), n_stride):
            raw = arr[t]
            train = prep_img(raw)                                   # 训练/离线路径
            deploy = preprocess_depth_old(depth_to_metres(raw))     # 部署路径（喂米）
            tot += 1
            d = np.abs(train.astype(np.int32) - deploy.astype(np.int32))
            if d.max() > 0:
                bad += 1
                worst = max(worst, int(d.max()))
            # A2: 忘记 depth_to_metres，把 uint8 当米喂
            wrong = preprocess_depth_old(np.asarray(raw, np.float32))
            sat = float(np.mean(wrong >= 254))
            sat_frac.append(sat)
            sat_frames += int(sat > 0.9)

    print(f"\n  A1  对比帧数 {tot}")
    print(f"      不一致帧数 {bad}   最大灰阶差 {worst}")
    print("      ⇒ " + ("✓ 两条路径逐像素完全一致，深度这一环不用改"
                        if bad == 0 else "✗ 存在不一致 —— 必须修"))

    print(f"\n  A2  误喂保护：把存储的 uint8 当米直接喂进 preprocess（忘了 depth_to_metres）")
    sf = np.array(sat_frac)
    print(f"      饱和到 ≥254 的像素占比: mean {sf.mean():.1%}  p50 {np.median(sf):.1%}")
    print(f"      整帧几乎全白(>90%饱和)的帧: {sat_frames}/{tot} = {sat_frames/max(tot,1):.1%}")
    print("      ⇒ 这是部署侧唯一真实的深度风险（tools.py:286 明写）：不会报错，只会白屏。")
    print("        防御：在 _preprocess_obs 里 assert 深度是浮点米且 max ≤ ~3.5，不是 uint8 码。")

    return {"n_frames": tot, "mismatch_frames": bad, "worst_gray_diff": worst,
            "misfeed_saturated_frac_mean": float(sf.mean()),
            "misfeed_whiteout_frames": sat_frames}


# ─────────────────────────────────────────────────────────────────────────────
# D — critic 加载
# ─────────────────────────────────────────────────────────────────────────────
def check_D(critic_path, device):
    print("\n" + "=" * 78)
    print("D  critic 加载完整性")
    print("=" * 78)
    from torch.nn.utils import spectral_norm as _sn
    from PyHJ.utils.net.common import Net
    from PyHJ.utils.net.continuous import Critic

    pth = torch.load(critic_path, map_location=device, weights_only=False)
    sd = pth.get("model", pth)
    csd = {k.replace("critic.", "", 1): v for k, v in sd.items()
           if k.startswith("critic.") and not k.startswith("critic_old.")}
    use_sn = any("weight_orig" in k for k in csd)
    print(f"  ckpt: {critic_path}")
    print(f"  critic.* 键数 {len(csd)}   spectral_norm: {'是' if use_sn else '否'}")

    lin = (lambda i, o: _sn(torch.nn.Linear(i, o))) if use_sn else None
    kw = {"linear_layer": lin} if use_sn else {}
    net = Net((1, 1, 1536), 3, hidden_sizes=[256, 256, 256],
              activation=torch.nn.ReLU, concat=True, device=device, **kw)
    critic = Critic(net, device=device, **kw).to(device)
    missing, unexpected = [], []
    try:
        critic.load_state_dict(csd, strict=True)
        print("  ✓ strict=True 加载成功（SN 自动识别版）")
    except RuntimeError as e:
        print(f"  ✗ strict=True 失败：{str(e)[:300]}")
        r = critic.load_state_dict(csd, strict=False)
        missing, unexpected = list(r.missing_keys), list(r.unexpected_keys)
    critic.eval()

    if use_sn:
        print("  ⚠ critic_safety_filter._load_critic 写死了**无 SN** 的 Net —— "
              "它加载这个 ckpt 会失败或半加载。M3 必须改。")
    else:
        print("  → 该 ckpt 无 SN，现行 _load_critic 的架构恰好对得上（但 strict 语义仍应修）")

    return {"path": critic_path, "n_keys": len(csd), "spectral_norm": bool(use_sn),
            "missing": missing[:10], "unexpected": unexpected[:10]}, critic


# ─────────────────────────────────────────────────────────────────────────────
# WM 构造 —— 与 CriticSafetyFilter._load_world_model 同构（不碰 gymnasium.make）
# ─────────────────────────────────────────────────────────────────────────────
def build_wm_like_filter(wm_path, config, device, obs_state_dim=42):
    """照抄 critic_safety_filter._load_world_model 的构造方式。

    不用 v_stop_utils.load_wm，因为它走 `gymnasium.make(config.task)` —— 在
    env_isaaclab 下返回的是 TimeLimit 包装器，没有 `observation_space_full`。
    直接手搭 space 既能跨环境跑，也**更忠实于部署路径**（滤波器就是这么建的）。

    ckpt 键的剥离逻辑照抄 v_stop_utils.load_wm，兼容各种前缀。
    """
    import gymnasium as gym
    import models

    config.device = device
    os.environ["DREAMER_DEVICE"] = device
    obs_space = gym.spaces.Dict({
        "image":       gym.spaces.Box(0, 255, (64, 64, 1), np.uint8),
        "obs_state":   gym.spaces.Box(-np.inf, np.inf, (obs_state_dim,), np.float32),
        "is_first":    gym.spaces.Box(0.0, 1.0, (1,)),
        "is_last":     gym.spaces.Box(0.0, 1.0, (1,)),
        "is_terminal": gym.spaces.Box(0.0, 1.0, (1,)),
    })
    act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    config.num_actions = act_space.shape[0]
    wm = models.WorldModel(obs_space, act_space, 0, config)

    ckpt = torch.load(wm_path, map_location=device, weights_only=False)
    state = ckpt.get("agent_state_dict", ckpt)
    own = wm.state_dict()
    wm_state = {}
    for k, v in state.items():
        if k.startswith("_wm."):
            nk = k[len("_wm."):]
        elif "._wm." in k:
            nk = k.split("._wm.", 1)[1]
        elif "_wm." in k:
            nk = k.split("_wm.", 1)[1]
        elif k.startswith("wm."):
            nk = k[len("wm."):]
        elif k in own:
            nk = k
        else:
            continue
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        wm_state[nk] = v
    if not wm_state:
        raise RuntimeError(f"checkpoint 里没有 world-model 权重: {wm_path}")
    missing, unexpected = wm.load_state_dict(wm_state, strict=False)
    print(f"  WM 载入: 匹配 {len(wm_state)-len(unexpected)} 键  "
          f"missing {len(missing)}  unexpected {len(unexpected)}")
    if missing:
        print(f"    (missing 前 5: {missing[:5]})")
    wm.to(device).eval()
    for prm in wm.parameters():
        prm.requires_grad_(False)
    return wm


# ─────────────────────────────────────────────────────────────────────────────
# B / C — Q 对拍
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def feats_batched(wm, traj, device, seed):
    """离线评估路径：整条轨迹一次 observe（encode_trajectories_batched 的等价单条版）。"""
    from scripts.vstop_Qbrake.v_stop_utils import get_image, get_obs_state, get_traj_actions, get_failure
    acts = np.asarray(get_traj_actions(traj), dtype=np.float32)
    T = min(len(traj["obs"]["image"]), len(acts))
    image = np.stack([get_image(traj, t) for t in range(T)])[None]
    obs_state = np.stack([get_obs_state(traj, t) for t in range(T)])[None]
    action = acts[:T][None]
    is_first = np.zeros((1, T), dtype=bool); is_first[0, 0] = True
    failure = get_failure(traj)[:T]
    dones = np.asarray(traj.get("dones", np.zeros(T)), dtype=np.float32)[:T]
    is_terminal = ((failure > 0) | (dones > 0))[None]

    torch.manual_seed(seed)
    data = wm.preprocess({"image": image, "obs_state": obs_state, "action": action,
                          "is_first": is_first, "is_terminal": is_terminal})
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    return wm.dynamics.get_feat(post)[0].cpu().numpy(), action[0]


@torch.no_grad()
def feats_stepwise(wm, traj, device, seed, depth_codec="new", enc_meta=None):
    """部署路径：逐帧 observe，携带 rssm_state。与 CriticSafetyFilter.update 同构。

    depth_codec="old" 时把 pkl 的图反量化回米、再走旧预处理 —— 复现 M1 的 bug。
    """
    from scripts.vstop_Qbrake.v_stop_utils import get_image, get_obs_state, get_traj_actions, get_failure
    acts = np.asarray(get_traj_actions(traj), dtype=np.float32)
    T = min(len(traj["obs"]["image"]), len(acts))
    failure = get_failure(traj)[:T]
    dones = np.asarray(traj.get("dones", np.zeros(T)), dtype=np.float32)[:T]

    torch.manual_seed(seed)
    prev, feats = None, []
    for t in range(T):
        img = get_image(traj, t)                                    # (64,64,1) 新编码
        if depth_codec == "old":
            lo, hi = float(enc_meta["clip_min"]), float(enc_meta["clip_max"])
            img = preprocess_depth_old(dequantize_new(img[..., 0], lo, hi))
        data = {"image": img[None, None],
                "obs_state": get_obs_state(traj, t)[None, None],
                "action": acts[t][None, None],
                "is_first": np.array([[t == 0]], dtype=bool),
                "is_terminal": np.array([[bool(failure[t] > 0 or dones[t] > 0)]], dtype=bool)}
        d = wm.preprocess(data)
        embed = wm.encoder(d)
        post, _ = wm.dynamics.observe(embed, d["action"], d["is_first"], state=prev)
        prev = {k: v[:, -1] for k, v in post.items()}
        feats.append(wm.dynamics.get_feat(post)[:, -1][0].cpu().numpy())
    return np.stack(feats)


def _report(name, dq):
    a = np.abs(dq)
    print(f"    {name:<34} mean |ΔQ| {a.mean():.5f}   p50 {np.median(a):.5f}   "
          f"p95 {np.percentile(a,95):.5f}   max {a.max():.5f}")
    return {"mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def check_B(wm, critic, trajs, device):
    from scripts.vstop_Qbrake.eval_q_mu import q_along, to_norm
    print("\n" + "=" * 78)
    print("B  Q 对拍：逐帧 observe（部署） vs 批量 observe（离线评估）")
    print("=" * 78)
    print("  判据：『逐帧 vs 批量』的 |ΔQ| 必须落在『同路径换 RNG』的底噪量级。")
    print("        obs_step 会采样 stoch，所以不控 RNG 的话两次相同计算本来就有差。\n")

    agg = {"noise": [], "parity": [], "old_codec": []}
    per_traj = []
    for i, tr in enumerate(trajs):
        f_a, act = feats_batched(wm, tr, device, seed=0)
        f_b, _ = feats_batched(wm, tr, device, seed=1)
        f_s = feats_stepwise(wm, tr, device, seed=0)
        cmd = to_norm(act[:, :3])
        q_a = q_along(critic, f_a, cmd, device)
        q_b = q_along(critic, f_b, cmd, device)
        q_s = q_along(critic, f_s, cmd, device)
        agg["noise"].append(q_a - q_b)
        agg["parity"].append(q_a - q_s)
        per_traj.append({"T": int(len(q_a)),
                         "noise_p50": float(np.median(np.abs(q_a - q_b))),
                         "parity_p50": float(np.median(np.abs(q_a - q_s)))})
        print(f"  [{i+1}/{len(trajs)}] T={len(q_a):3d}  "
              f"底噪 p50 {np.median(np.abs(q_a-q_b)):.5f}   "
              f"对拍 p50 {np.median(np.abs(q_a-q_s)):.5f}")

    print("\n  汇总（全部帧池化）")
    out = {}
    out["noise_floor"] = _report("底噪   批量(seed0) vs 批量(seed1)", np.concatenate(agg["noise"]))
    out["parity"] = _report("对拍   批量 vs 逐帧", np.concatenate(agg["parity"]))
    ratio = out["parity"]["p50"] / max(out["noise_floor"]["p50"], 1e-12)
    print(f"\n    对拍/底噪 (p50) = {ratio:.2f}   "
          f"{'✓ 同量级，部署路径 == 评估路径' if ratio < 3.0 else '✗ 显著超出底噪，存在真实不一致'}")
    if ratio >= 3.0:
        print("      偏差形状可以反推病因：")
        print("        整条平移        → gx_tau / config 块（但注意 gx_tau 不进 critic）")
        print("        近距离时差大    → 深度编码 (M1)")
        print("        完全不相关      → critic 加载 (M3)")

        print("=" * 78)
        out["old_codec"] = _report("旧编码引入的 ΔQ", np.concatenate(agg["old_codec"]))
        rc = out["old_codec"]["p50"] / max(out["noise_floor"]["p50"], 1e-12)
        print(f"\n    旧编码ΔQ/底噪 (p50) = {rc:.2f}   "
              f"{'→ 淹没在噪声里，M1 影响可忽略' if rc < 3.0 else '→ 远超底噪，M1 必须修'}")
    out["per_traj"] = per_traj
    out["parity_over_noise_p50"] = float(ratio)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default=f"{LATENT_SAFETY}/logs/go2_fricsweep_wm_privimag/rssm_ckpt.pt")
    ap.add_argument("--critic", default=None, help="epoch_id_*/policy.pth；不给则跳过 B/C/D")
    ap.add_argument("--config", default="go2_fricsweep_eval",
                    help="configs.yaml 块。必须带 task='go2-wm'（load_wm 要 gymnasium.make 它）——"
                         "go2_fricsweep_ddpg 没有 task，会去找未注册的 complex_terr_add_rew。")
    ap.add_argument("--gx_tau", type=float, default=0.0,
                    help="覆盖 config 的 gx_tau。go2_fricsweep_eval 继承 defaults 的 -0.5，"
                         "而 fricsweep 全链是 0.0。⚠ gx_tau **不进 critic Q**，只影响 g 诊断列，"
                         "所以 B/C 的对拍结果与它无关。")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exam_data", nargs="+", default=EXAM_DIRS)
    ap.add_argument("--n_traj", type=int, default=5)
    ap.add_argument("--only", default=None, choices=["A", "D", "B"], help="只跑某一项")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.vstop_Qbrake.v_stop_utils import load_config, load_trajectory_sources

    rng = np.random.default_rng(args.seed)
    print(f"[verify] 加载考卷轨迹 …")
    trajs, _ = load_trajectory_sources(args.exam_data, None)
    idx = rng.choice(len(trajs), size=min(args.n_traj, len(trajs)), replace=False)
    trajs = [trajs[i] for i in idx]
    print(f"[verify] 取 {len(trajs)} 条（共 {len(idx)} / 池中若干）")
    enc_meta = trajs[0].get("depth_encoding")

    results = {"wm": args.wm, "critic": args.critic, "config": args.config,
               "n_traj": len(trajs)}

    if args.only in (None, "A"):
        results["A"] = check_A(trajs)
    if args.only == "A":
        _dump(results, args.out); return

    if args.critic is None:
        print("\n[verify] 未给 --critic，跳过 B/C/D。")
        _dump(results, args.out); return

    if args.only in (None, "D", "B"):
        results["D"], critic = check_D(args.critic, args.device)
    if args.only == "D":
        _dump(results, args.out); return

    print(f"\n[verify] 加载 WM  config={args.config}")
    cfg = load_config(args.config)
    cfg.device = args.device
    cfg_gx = getattr(cfg, "gx_tau", None)
    if args.gx_tau is not None and cfg_gx != args.gx_tau:
        print(f"         gx_tau: config 给的是 {cfg_gx} → 覆盖为 {args.gx_tau}")
        cfg.gx_tau = args.gx_tau
    print(f"         task = {getattr(cfg, 'task', None)}   gx_tau = {getattr(cfg, 'gx_tau', None)}   "
          f"(仅影响 g 诊断列；critic Q 不含 gx_tau)")
    wm = build_wm_like_filter(args.wm, cfg, args.device)
    results["gx_tau"] = float(getattr(cfg, "gx_tau", float("nan")))
    results["B"] = check_B(wm, critic, trajs, args.device)

    _dump(results, args.out)


def _dump(results, out):
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False, default=float)
        print(f"\n[verify] 写入 {out}")
    print("\n[verify] 完成。")


if __name__ == "__main__":
    main()
