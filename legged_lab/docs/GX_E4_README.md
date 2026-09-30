# gx-heightmap-e4 分支：LeggedLab 侧说明

本分支配合 latent-safety 的同名分支 `gx-heightmap-e4`，收录 **E4（纯几何 g）** 和 **静态高程图 g（gx_e4）** 两条路线在仿真侧的代码。
这两条线的训练、V_stop、结果和完整流程写在 latent-safety 的 `docs/gx_e4/README.md` 与 `docs/gx_e4/BRANCH_RECORD.md`。
本文件只讲 LeggedLab 负责的部分：**采集**、**闭环评测环境**和**部署前校验**。

默认目录布局（脚本里写的是绝对路径）：

```
/home/lcy/LeggedLab        # 本仓库，conda env: env_isaaclab（含 Isaac Lab）
/home/lcy/latent-safety    # 世界模型 / V_stop 仓库，conda env: latent-safety
```

---

## 1. 本分支的改动

| 类别 | 文件 |
|---|---|
| RobotLab v1 策略接入 | `utils/robotlab_policy.py`、`assets/unitree/unitree_actuator.py`（GO2HV 执行器）、`envs/go2/go2_robotlab_parity_config.py` + `go2_robotlab_parity_manifest.json`（关节顺序对照）、`envs/__init__.py`、`envs/go2/__init__.py` |
| 地形 | `terrains/terrain_generator_cfg.py`（`gx_e4`、`gx_main_big/small`、`gx_pilot` 等 profile）、`terrains/grid_terrain_generator.py` |
| 采集器 | `scripts/collect_go2_data_v3.py`：`--store_geo`（逐帧位姿/足端/接触 + 地形 mesh，供离线打高程图标签）、`--robotlab_policy`、`--actuator go2hv`、`--camera_mount d435`、μ 无关/μ 相关的起刹尺子等 |
| 闭环评测 | `scripts/play_with_ood_friction_eval.py`（`--eval_terrain`、`--robotlab_policy`、按 env 配额）、`scripts/critic_safety_filter.py`（加载 WM / V_stop，geometric 模式 g 不过 tanh） |
| 环境 | `envs/base/*`、`mdp/events.py`、`mdp/rewards.py`、`utils/env_utils/scene.py` |
| 采集入口 | `scripts/run_gx_e4.sh`、`run_gx_main.sh`、`run_gx_pilot.sh`、`run_flat_brake_ref_v1.sh` |
| 校验 | `scripts/verify_robotlab_policy.py`、`verify_filter_parity.py`、`smoke_test_go2_robotlab_parity_rewards.py` |
| 文档 | 本文件、`GO2_ROBOTLAB_PARITY_PLAN.md`、`ROBOTLAB_POLICY_COLLECTION.md`、`FRICTION_DATASET_COMPOSITION.md`（E4 数据集构成，§9） |

- 移到 `scripts/archived/`、不上传的脚本：`run_flat_brake_ref.sh`（已被 v1 取代）、`run_ood_fric_triplets.sh`（旧的 triplet 路线）。
- `.gitignore` 新增：根目录 `/data/`（采集数据约 8.7 GB，原来没有被忽略）、`*.log`、`/legged_lab/scripts/archived/`。

## 2. 需要的策略文件（GitHub Release `gx-heightmap-e4-v1`，挂在 latent-safety 仓库）

| Release 文件 | 放到（相对 `legged_lab/`） | 用于 |
|---|---|---|
| `robotlab_v1_policy.pt` | `logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt` | gx_e4 / gx_main / pilot 的采集与闭环 |
| `e4_leggedlab_ppo_model_2500.pt` | `logs/go2_flat_robotlab_parity/2026-08-10_01-23-30/model_2500.pt` | E4 闭环（`--load_run 2026-08-10_01-23-30`，按目录名查找，**必须放在这里**） |

`run_gx_e4.sh` 和 latent-safety 的 `run_gxstatic_trace_pair.sh` 可以用 `POLICY=/abs/path/robotlab_v1_policy.pt` 指定别的位置。

## 3. 采集

所有采集脚本都在仓库根目录 `/home/lcy/LeggedLab` 下运行，用 env_isaaclab 的 Python。

### gx_e4（静态高程图主数据，1900 条，约 3 小时）

```bash
for B in B1 B2 B3 B4 B0 A1; do BLOCK=$B bash legged_lab/scripts/run_gx_e4.sh; done
# → data/gx_e4_v1/<B>/gx_e4_<B>.pkl + run_config.json + braking_config.json + terrain_mesh.npz + collect.log

# 试运行（无界面，8 env × 1 条，写到临时目录，约 1 分钟）
TRIAL=1 HEADLESS=1 BLOCK=B1 ROOT=/tmp/gx_trial bash legged_lab/scripts/run_gx_e4.sh
```

| 块 | 内容 | 条数 |
|---|---|---:|
| B1 / B2 / B3 | 刹车，μ 分别 0.22–0.32 / 0.32–0.55 / 0.55–1.00，μ 无关起刹尺子 | 800 / 500 / 200 |
| B4 | 高 μ 边界补采，μ 相关起刹尺子 | 150 |
| B0 | 弱刹车 | 150 |
| A1 | 低 μ 行走 | 100 |
| E | 配对考卷（3 档 μ 同 seed），默认跳过，`E_ENABLE=1` 开启 | — |

各子集的实际采集参数已拷到 latent-safety 的 `docs/gx_e4/manifests/leggedlab_gx_e4_v1/`。

### 其他采集

```bash
SUBSET=A bash legged_lab/scripts/run_gx_main.sh     # gx_main_v1（A/B/C/D 四个子集；对照组，负结果）
bash legged_lab/scripts/run_gx_pilot.sh             # pilot：拟合通过性曲线，标定 h_up / h_down
bash legged_lab/scripts/run_flat_brake_ref_v1.sh    # RobotLab v1 平地刹车参考：拟合 d_brake(v, μ)、d_hold
```

加 `TRIAL=1` 都是 GUI 试运行（8 env × 1 条）。

采集完成后，由 latent-safety 的 `scripts/gxstatic/label_gx_main.py` 离线打标签，
之后的流程见 latent-safety `docs/gx_e4/README.md` §5。

## 4. 闭环评测

闭环评测由 latent-safety 的脚本发起：它们 `cd` 到本仓库，调用 `scripts/play_with_ood_friction_eval.py`。

- gx_e4：`latent-safety/scripts/vstop_Qbrake/run_gxstatic_trace_pair.sh`（`EVAL_TERRAIN=gx_e4 PHW=4.0 CAMERA=d435`）
- E4：`latent-safety/scripts/vstop_Qbrake/run_gxgeo_trace_pair.sh`（需要 env_isaaclab 在 PATH 最前）

两者都在同一个 seed 下各跑一遍低摩擦（lo）和高摩擦（hi），按 env 配对比较。
低摩擦 μ 为 0.24（静）/ 0.17（动），高摩擦为 0.80 / 0.56；在离边 3.0 m 处切换摩擦。
`takeover_duration` 必须保持 120：设成 50 会在低 μ 还在滑时就强制 reset，掉落率会被低估。

## 5. 部署前校验（不需要完整训练）

```bash
cd /home/lcy/LeggedLab/legged_lab
PY=/home/lcy/miniconda3/envs/env_isaaclab/bin/python
$PY scripts/verify_robotlab_policy.py --policy /abs/path/robotlab_v1_policy.pt     # 策略适配器 vs 导出模型，PASS 即可
$PY scripts/verify_filter_parity.py --wm /abs/path/e4_gxgeo_wm.pt \
    --config go2_fricsweep_eval,go2_compmargin,go2_gxgeo --only A                    # 深度预处理：训练路径 vs 部署路径
```

`verify_filter_parity.py` 依赖 latent-safety 的 `scripts/vstop_Qbrake/eval_q_mu.py`；latent-safety 不在默认位置时，设置 `LATENT_SAFETY_ROOT`。

2026-09-30 用 Release 文件实测：以上两项、试运行采集、两条闭环各一次（4 env，lo/hi）全部通过。
记录见 latent-safety `docs/gx_e4/BRANCH_RECORD.md` §6。

## 6. 数据与备份

采集数据（`data/`、`legged_lab/data/`）不进 git。
完整拷贝在 `/media/lcy/My Passport/Backup-PC-status2026-10/Ubuntu2204/LeggedLab`（与本机路径一致）。
