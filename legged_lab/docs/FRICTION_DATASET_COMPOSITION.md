# Friction-adaptive 数据集构成

面向 **μ-adaptive critic** 的 Go2 制动数据集。本文取代
`FRICTION_ADAPTIVE_DATASET_PLAN.md` 第 2 节的参数表——那里的
`d̂_low = 0.59v + 0.144v²` 是在 62% 删失的样本上拟的，已被证伪。

采集脚本：`legged_lab/scripts/collect_go2_data_v3.py`

---

## 1. 已测定的物理常数

```
d_stop = 0.170 · v / μ_dynamic          刹停距离
d_hold = 0.50 m                          刹停后站稳所需的最小离边距离，μ 无关
尺子    d̂_ref(v) = 1.15 · v + 0.50      调度用，必须与 μ 无关
怠速地板 ≈ 0.17 m/s                      零指令下平衡抬腿的残余速度
相机水平视距 ≈ 2.8 m                     名义 clip 3.0 m，俯视安装后的实测有效值
```

**尺子取 `1.15 / 0.50` 而不是 `1.00 / 0.20`**：这是 v3 批次（n=300）实跑并验证过的
唯一一组，产出 safe_stop 54.3% / overrun 24.3% / spawn 重采 0。`d_hold` 的两个候选
估计（联合拟合 0.20、drift_off 剂量–反应上界 0.50）里，0.50 是被端到端结果背书的那个。

| 结论 | 证据 | 样本 |
|---|---|---|
| **`d ∝ v`，不是 `v²`** | 尺子取 p=1 时 r\* 与速度无关（1.00 vs 1.03）；解出指数 p = 1.11 / 0.80 | 100 |
| 库仑模型 `v²/(2μg)` 低估 1.6–10× | 见 §6 | — |
| **起作用的是动摩擦，不是静摩擦** | 同一模型换 μ 通道，Δ logL = **+93.7**（判据 2）；σ 从 0.364 降到 0.232 | 582 |
| `d_hold = 0.50 m` | 刹停时 d_edge < 0.30 m → 100% 翻覆；< 0.40 → 63%；< 0.50 → 39%；> 0.494 → 0% | 128 |
| **怠速地板 0.17 m/s 会伪造 `no_stop`** | v3 的 56 条 `no_stop`，刹车后最低持续速度中位 0.151 m/s（其余 outcome 是 0.059）；阈值提到 0.20 后 86% 重判为已刹停 | 300 |
| **相机水平视距 2.8 m，不是 3.0 m** | 悬崖在深度图里的无回波抬升在 `d_edge_ray > 2.75 m` 处回到远处基线（25.2%） | 48391 帧 |
| 低 μ 刹车距离是高 μ 的 **3.05 倍** | r\*_low = 1.017 vs r\*_high = 0.334（同一把尺子） | 100 |
| 零指令**不是**站住 | 判定刹停后仍平均走 0.26 m；怠速地板 ≈ 0.17 m/s | 300 |

拟合式 `d = 0.170 v/μ_d + 0.20` 在两个摩擦档上的预测误差约 ±4%。**注意：μ 只采过
两大档，`1/μ` 这个形式尚未在连续 μ 上验证**——这是本数据集要回答的问题之一。

### 删失是必须处理的

刹车距离的朴素拟合会系统性偏低，因为**刹得越远越容易被平台边缘截断**。同一批
50 条数据上三个估计量：

| 估计量 | d/v₀ |
|---|---|
| 幸存者（刹停组）朴素拟合 | 0.974 s |
| 被删失组的下界 | **1.066 s** ← 已超过幸存者均值 |
| probit（用全部样本，无偏） | **1.301 s** |

**任何刹车距离的重拟必须用 probit / 删失回归，不能只用刹停的那些。**

---

## 2. 数据集构成

| 块 | μ_static | ratio | → μ_dynamic | v (m/s) | r | vy / ω | 条数 |
|---|---|---|---|---|---|---|---|
| **B1** 低摩擦 | 0.22–0.32 | 0.55–0.90 | 0.12–0.29 | 0.5–2.5 | 0.15–1.15 | ±0.15 / 0 | 800 |
| **B2** 中摩擦 | 0.32–0.55 | 0.55–0.90 | 0.18–0.50 | 0.5–2.5 | 0.15–1.15 | ±0.15 / 0 | 500 |
| **B3** 高摩擦 | 0.55–1.00 | 0.55–0.90 | 0.30–0.90 | 0.5–2.5 | 0.15–1.15 | ±0.15 / 0 | 700 |
| **B0** 弱刹/不刹 | 0.22–1.00 | 0.55–0.90 | 0.12–0.90 | 0.5–2.5 | 0.15–1.15 | ±0.15 / 0 | 400 |
| **A1** 低 μ 走路 | 0.22–0.32 | 0.55–0.90 | 0.12–0.29 | 0.5–2.5 | — | ±0.3 / ±0.3 | 300 |
| **E** 配对评测 | 3 档同 seed | 0.70 固定 | 0.17 / 0.28 / 0.56 | 0.5–2.0 | 0.15–1.15 | ±0.15 / 0 | 180 |

**合计 2880**：B 系 2400（训练）+ A1 300 + E 180（held-out）。地形全部为
`cliff_brake`（`CLIFF_DETECTION_TERRAINS_CFG`，10×10 = 100 格，proportion
55/15/15/15 = dangerous_platform / deep_pit / stairs / slope）。

### 三块共用一个 r 范围

早期方案给三块不同的 r 带（B1 0.50–1.55 / B2 0.30–1.20 / B3 0.20–0.80），已废弃。
按校正后的 r\* 公式铺开整个设计空间：

```
r*(v, μ_d) = 0.86 · (0.170·v/μ_d + 0.50) / (1.15·v + 0.50)     校正系数来自 v3 实测

  v\μ_d    0.12   0.17   0.25   0.35   0.50   0.70   0.90
    0.3    0.94   0.81   0.72   0.66   0.61   0.58   0.57
    1.5    1.01   0.77   0.59   0.47   0.39   0.33   0.30
    2.5    1.03   0.76   0.56   0.44   0.34   0.28   0.25
```

**全空间 r\* ∈ [0.25, 1.03]。** 尺子里的常数项 0.50 把 μ 的影响压缩了——低 μ 与高 μ
的 r\* 只差不到 4 倍，而不是 d_stop 那样的 7 倍，不构成分带的理由。统一取
**[0.15, 1.15]** 覆盖全部 r\* 并留出平台段，同时把设计约束 ② 做到极致：r 带完全重合，
margin **不可能**成为 μ 的代理变量。

v3 实测的剂量–反应（尺子 1.15v+0.50，stop 阈值重判为 0.20/10 帧）：

| r 区间 | n | 失败率 |
|---|---|---|
| 0.50–0.70 | 46 | 80% |
| 0.70–0.85 | 42 | 38% |
| 0.85–1.00 | 28 | 36% |
| 1.00–1.15 | 55 | 13% |
| 1.15–1.30 | 43 | 16% |
| 1.30–1.60 | 81 | 12% ← 平台，纯浪费 |

probit：**r\* = 0.77, σ = 0.47**。曲线在 r ≈ 1.0 就躺平，所以上沿 1.15 足够，
再往上只是烧样本。

### 内存

深度图占总内存 98.3%，是唯一值得优化的项。`--depth_uint8` 后 64×64 深度帧从
16 KB 降到 4 KB：

| 配置 | 均长 | float32 | uint8 |
|---|---|---|---|
| 本方案（`max_steps 500 / post_brake 4.0`） | ≈350 | 17.4 GB | **≈4.4 GB** |
| 附加 `max_steps 300` 硬截 | 273 | 13.1 GB | 3.4 GB ← **不要用** |
| 附加 `early_stop_hold_steps 90` | 257 | 12.3 GB | 3.2 GB ← **不要用** |

后两行省下的 0.2–0.4 GB 要用删失偏差去换，不划算，理由见 §9.1。

### 各块的作用

- **B1/B2/B3** 是主体。按 μ 分三段只为让 **r 带跟着 r\*(μ) 走**——单一 r 带覆盖
  0.30–1.65 的话，任何给定 μ 下大部分样本都远离阈值，浪费采样。
- **B0** 提供**未制动的对照臂**。critic 必须能评估"不刹车会怎样"，而这在只有全力
  刹停的数据里是分布外的。同时补上动作维度：`brake_target_ratio` 均匀落在 [0,1]，
  从完全刹停到完全不刹车的连续谱，Q_brake 在动作轴上才有信号。
- **A1** 提供**长时低摩擦无制动暴露**。B 块里低 μ 行走只有 `fault_window`
  那 0.6–1.2 s，不足以支撑"在冰上继续走 3 秒"这个想象分支。
- **E** 是**留出评测集**，不进训练。同 seed、同 env，`actions[:t_fault]` 逐位相同，
  只有 μ 不同——用来直接检验训练好的 WM 面对相同前缀是否给出不同的制动距离预测。

---

## 3. 三条必须守住的设计约束

**① 尺子必须与 μ 无关。** 触发用 `r · d̂_ref(v)`，`d̂_ref` **不含本条 episode 的 μ**。
若用本条自己的 μ 去算，两类摩擦都会在 r=1 翻车，**outcome 与 μ 结构性独立**，μ 通道
被判定无用。这个 bug 出现过一次，表现为两类摩擦精确 50/50。

**② r 带必须两两重叠。** B1∩B2 = [0.50, 1.20]，B2∩B3 = [0.30, 0.80]。不重叠的话
余量本身就是 μ 的代理变量，模型可以从 margin 反推摩擦而不看观测。

**③ fault 不能在加速段触发。** 由 `--tracking_hold_s` 的 settled 门控保证。否则高速
指令只能采到 87% 的速度，**速度轴会静默塌陷**。

### μ_dynamic 覆盖连续，无空洞

`[0.12, 0.29] ∪ [0.18, 0.50] ∪ [0.30, 0.90] = [0.12, 0.90]`。连续覆盖是为了能在
全体样本上做**联合参数拟合**而不是分箱——统计效率更高，且事后仍可按 μ 分箱检查形式。

---

## 4. spawn 解算

```
D₀(v, r, w, e) = v²/(2a) + v·(T_settle + w) + r·d̂_ref(v) + e
   ↑加速段         ↑稳定+检测窗口          ↑触发余量      ↑额外跑道

x₀ = half − D₀·cos(θ)          θ 为相对目标边法线的朝向
```

**D₀ 是 spawn 点到边缘的距离，不是到停止点的距离。** 最后一项 `r·d̂_ref(v)` 是刹车
开始那一刻手里还剩多少距离，不是刹车阶段本身。实际能不能刹住取决于
`d_stop(v, μ)`，它**不在公式里**——那正是实验要测的量。

`r = 刹车开始时剩余距离 ÷ 刹停所需距离`。触发是**闭环**的（`d_edge_ray` 达到阈值
才点火），所以 `--spawn_extra_runway` 加的跑道**完全不改变 r**，只是让机器人多走一段、
多看几帧不同距离的边缘——用来解除 `(d_edge, v)` 的共线（D₀ ∝ v 会让二者绑死）。

### 可行域

```
r_max(v) = (2·half − lateral − v²/2a − v·(T_settle + w)) / d̂_ref(v)
```

(v, r) 超出可行域时**重采**而不是报错。重采率会在采集结束时打印，**>10% 会告警**——
高重采率意味着 r 的边缘分布依赖 v，正是要避免的混淆。上表各块的实测重采率
≤ 0.024/episode，r 分布畸变 ≤ 2.4%。

---

## 5. 指令为什么这样设

| 指令 | 刹车块 | 走路块 | 依据 |
|---|---|---|---|
| `vy` | **±0.15** | ±0.3 | 实测横移角 7.2°（低速）/ 2.6°（高速），射线度量误差 <1% |
| `ω` | **0** | ±0.3 | ω=±0.3 时 9.7% 从不触发刹车（机器人绕圈走离边缘）；r\* 偏移 0.13；σ 涨 30% |

spawn 解算假设**沿朝向直线前进**，D₀ 是直线距离。ω≠0 时路径是弧线，r 就没按设计
实现。朝向多样性由 `--spawn_heading_range ±45°` 提供，与 ω 作用重复但无副作用。

---

## 6. 为什么不用物理公式

滑块模型 `d = v²/(2μg)` 的三个前提对足式机器人都不成立：

| 前提 | 实际 |
|---|---|
| 减速力是接触面的库仑摩擦 | 减速力由 policy 通过关节力矩**主动产生**，摩擦只是传递上限 |
| 该力已饱和（足端在滑） | 高 μ 下实测减速 3.25 m/s²，而 μg = 7.4 m/s²，**只用了 44%**。卡住它的是俯仰翻越（`g·L/h ≈ 5.8`）和力矩上限 |
| 机器人是质点 | 减速要迈步把足端放到质心前方，需要若干**步态周期**——质点模型没有这个时间尺度 |

数值对比（μ_s≈0.22 / 0.75）：

| v | 低μ实测 | 库仑 | 高μ实测 | 库仑 |
|---|---|---|---|---|
| 1.0 | 1.30 m | 0.23 m | 0.48 m | 0.07 m |
| 2.5 | 3.25 m | 1.45 m | 1.19 m | 0.42 m |

但 μ 的**量级**效应相当库仑：平均减速度比 3.11，μ 比 3.41。错的只是速度依赖。
所以正确的形式是 `d = c·v/μ`（时间常数由摩擦决定，距离是 v 线性放大），
而不是 `d = v²/(2μg)`。

---

## 7. 地形

采集用 `CLIFF_DETECTION_TERRAINS_CFG`（`--terrain_profile cliff_brake`），
**10×10 = 100 格**，四种子地形按 `proportion` 逐格随机采样：

| terrain_key | proportion | hazard_dir | 几何 | 越界后果 |
|---|---|---|---|---|
| `dangerous_platform` | 0.55 | **−1** | origin z > 0，机器人在台面上 | 坠落 |
| `deep_pit` | 0.15 | **+1** | origin z < 0，机器人在坑底 | 撞上升台阶 |
| `stairs` | 0.15 | **+1** | 倒金字塔，机器人在最低平台 | 撞上升台阶 |
| `slope` | 0.15 | **+1** | `inverted=True`，平台在最低处 | 上坡 |

四者的 `platform_width` 全是 8.0，**矩形边界 half = 4.0 对四者都正确**。

余量几何相同，失败机制不同。`hazard_dir` 已逐 episode 记录，**分析时必须分层**，
不要混在一起。实测 r\*：platform 0.923 / pit 0.856 / stairs 0.869（slope 尚无数据）。

> `slope` 的 `hazard_dir` 原先写成 −1，是错的。`HfInvertedPyramidSlopedTerrainCfg`
> 继承自 `HfPyramidSlopedTerrainCfg` 且 `inverted=True`，上游文档原话是
> "the platform is at the bottom" —— 机器人在最低的中央平台上，越界是**上坡**，
> 和 pit / stairs 同类。已修正为 +1。

> `flat` 故意不启用。`MeshPlaneTerrainCfg` 没有平台边缘，`d_edge_ray` 会对着一个
> 不存在的矩形边界计算，机器人越过后既不坠落也不碰撞——制动 episode 全是废片。
> 需要安全基线用 `FLAT_MESH_TERRAINS_CFG` 单独采。

### 不需要手工排 layout

`proportion` 就是配比的来源，`grid_layout` 保持 `None`。之所以仍然走
`GridTerrainGeneratorCfg` 而不是普通 `TerrainGeneratorCfg`，只为解决一个信息丢失：

**上游 `TerrainGenerator` 生成完就把"哪一格是哪种地形"丢掉了。**
`_add_sub_terrain(mesh, origin, row, col, sub_cfg)` 收到了 `sub_cfg`，但只往
`terrain_origins[row, col]` / `terrain_meshes` / `flat_patches` 里写，类型不落盘；
而 `TerrainImporter` 把 generator 当局部变量用完即弃。于是下游拿不到 `terrain_key`，
上面那张 `hazard_dir` 分层表就无从做起。

`GridTerrainGenerator._generate_random_terrains` 因此做两件事：

- `grid_layout=None`：按 `proportion` 逐格采样（与上游同一套逻辑），
  **把实际采到的布局回写**到 `cfg.grid_layout` 和模块级 `LAST_REALIZED_LAYOUT`，
  并在启动日志里打印实际配比。
- `grid_layout` 给定：按固定布局生成，此时 `proportion` 被忽略。

两种模式下 `difficulty` 都逐格随机，所以台面落差在 `(0.40, 0.60)` 内照常变化。

采集器反查 `terrain_key` 的方式不变（`flat_idx = level * num_cols + type`），
cfg 上取不到就退到 `LAST_REALIZED_LAYOUT`。

**残留一点**：随机采样下各次运行的配比会漂移，100 格时次要类型的 95% 区间是
[8, 22]（中位 15），所以 B1/B2/B3 之间地形配比不完全相同。但 `terrain_key` 已逐
episode 记录，**这是被测量到的协变量，分析时分层或重加权即可**，不构成隐性混杂。

**注意**：`GridTerrainGenerator` 只覆写了 `_generate_random_terrains`，
没覆写 `_generate_curriculum_terrains`——`curriculum=True` 时上面这套记录会被
**整个绕过**。cfg 里写死 `curriculum=False`，采集脚本再强制一次。

### spawn 隔离：num_envs 必须小于可 spawn 格数

`enable_random_terrain_spawn=True` 时，`randomize_terrain_spawn` 会
**优先把 reset 的 env 放到空闲格子**（`_sample_terrain_cells_priority`）。但这只是
优先，不是保证：可 spawn 格数 ≤ num_envs 时必然有两台机器人同格。

同一 8 m 平台上的两台 Go2 相距常常 < 2.8 m，**会进入彼此的深度相机**。margin head
本该从图像读地形，读到的却是另一台机器人——静默的观测污染。相邻格子不存在这个问题：
格心间距 12 m，平台边到边 4 m > 2.8 m 视距。

100 格配 20–40 个并行 env 时恒有 60+ 个空闲格，独占与随机性都有富余。采集脚本另有
硬校验：`num_envs >= 可 spawn 格数` 直接报错，`num_envs > 格数 − 3` 打 WARN。

早期用过 3×3（9 格）只是为了省算力，配 `--num_envs 10` 时每次 reset 都至少有一台
被迫同格 —— 已作废的 900 条就是这样采的。

---

## 8. 记录的字段

逐帧：`d_edge`（Chebyshev，标注用）、`d_edge_ray`（沿朝向，触发用）、`base_z`、
`obs.priv_state`（含静/动摩擦真值）。

逐 episode：`outcome`、`t_stop`、`d_edge_at_stop`、`t_brake`、`t_fault`、
`t_tracking_settled`、`brake_margin_ratio`、`d_hat_ref_coeffs`、`spawn_d0`、
`spawn_theta`、`spawn_y_cross`、`spawn_extra_runway`、`fault_window_s`、
`brake_target_vx`、`brake_is_complete`、`max_steps_budget`、`sampled_static_mu`、
`sampled_dynamic_mu`、`terrain_key`、`terrain_type`、`terrain_level`、`hazard_dir`。

**余量以米记录，所以任何下游分析都可以用物理单位重算，不依赖尺子的正确性。**
尺子只决定样本落在边界附近的比例，即统计效率，不影响数据有效性。

### outcome 六分类

| 值 | 含义 | 是不是余量失败 |
|---|---|---|
| `safe_stop` | 刹停并保持在平台内 | — |
| `overrun` | 从未刹停，越过边缘 | **是**，余量不够 |
| `drift_off` | 刹停后才失败 | 否，是 `d_hold` 不够 |
| `no_stop` | 未越界但预算内没完全静止 | 否 |
| `off_before_brake` | 触发前就出界 | 否，**是未制动走出边缘的对照臂** |
| `no_brake` | 触发从未点火且未失败 | 唯一真正的废片 |

`overrun` 和 `drift_off` **需要相反的修复**（前者加 `r`，后者加 `d_hold`），
把它们合成一个 `FAIL_RESET` 会掩盖问题。曾经把 `drift_off` 计入余量失败，
导致 r\*_high 被高估 12%（0.373 → 0.334）。

### `off_before_brake` 不是废片

它不进 r\* 拟合（`t_brake = -1`，没有制动试验），但对 WM 和 critic 是有效轨迹：
机器人在低摩擦上走、没刹车、走出了边缘——正是 critic 必须能评估的"不刹车会怎样"。
冒烟 100 条里的 3 条：

```
T= 77  v_max=1.36  μ_d=0.233  摩擦未跌落(常摩擦)  → 走出边缘
T=140  v_max=1.05  μ_d=0.190  低摩擦已施加        → 走出边缘
T=142  v_max=0.67  μ_d=0.201  低摩擦已施加        → 走出边缘
```

两条低摩擦走出、一条常摩擦走出，共占 1.3% 的帧。

**但不能把它当成设计好的走出边缘臂。** 它只出现在触发阈值够不着的低 r / 低 v 角落
（成因见 §11.4），速度系统性偏低（这 5 条 v_max 中位 0.67，全体 1.76）。真正覆盖
全速度段的未制动臂是 **B0**：`brake_target_ratio ~ U(0,1)`，其中 ratio → 1.0 那一段
就是"完全不刹车"，400 条里约 40 条落在 ratio > 0.9，且 v 铺满 0.5–2.5。

`no_brake` 才是废片：触发没点火、也没失败，机器人低速爬完整个预算。冒烟里 2 条
就占了 808 帧（全部帧的 2.9%），内容与 A1 的低摩擦走路重复。

---

## 9. 采集命令

从 `/home/lcy/LeggedLab/legged_lab` 执行。`--num_envs 20` 让每块的条数都能整除；
20–40 都可以，改的话同步调 `--num_episodes`（总条数 = 两者相乘）。

```bash
COMMON="--task go2_data_collection_robotlab --enable_cameras --num_envs 20 \
 --collection_mode braking --fault_type friction --safe_ratio 0.0 \
 --brake_trigger edge --fault_trigger window --fault_window_range 0.6 1.2 \
 --spawn_mode solve --spawn_accel 2.6 --spawn_settle_s 0.25 \
 --spawn_extra_runway 0.0 1.0 --spawn_lateral_margin 0.2 \
 --spawn_heading_range -0.785398 0.785398 \
 --brake_ref_linear 1.15 --brake_ref_quadratic 0.0 --brake_ref_offset 0.50 \
 --brake_margin_ratio_range 0.15 1.15 \
 --stop_speed_thresh 0.20 --stop_sustain_steps 10 \
 --command_vx_range 0.5 2.5 --command_vy_range -0.15 0.15 \
 --command_yaw_rate_range 0.0 0.0 \
 --ood_dynamic_ratio_range 0.55 0.90 \
 --save_interval 100 --no_final_merge"
TERRAIN="--terrain_profile cliff_brake --platform_half_width 4.0 --terrain_seed 20260901"
LENGTH="--max_steps 500 --dynamic_max_steps --post_brake_budget_s 4.0"
DEPTH="--depth_uint8 --depth_clip_min 0.30 --depth_clip_max 3.00"
BASE="$COMMON $TERRAIN $LENGTH $DEPTH"

# ── 冒烟验证：100 条，跑完先对账再开正式采集 ──────────────────
python scripts/collect_go2_data_v3.py $BASE --seed 20260899 --num_episodes 5 \
  --brake_complete_ratio 1.0 --ood_static_friction_range 0.22 0.32 \
  --output_dir data/fric_smoke

# ── B1  低 μ  800 ────────────────────────────────────────────
python scripts/collect_go2_data_v3.py $BASE --seed 20260901 --num_episodes 40 \
  --brake_complete_ratio 1.0 --ood_static_friction_range 0.22 0.32 \
  --output_dir data/fric_B1_low

# ── B2  中 μ  500 ────────────────────────────────────────────
python scripts/collect_go2_data_v3.py $BASE --seed 20260902 --num_episodes 25 \
  --brake_complete_ratio 1.0 --ood_static_friction_range 0.32 0.55 \
  --output_dir data/fric_B2_mid

# ── B3  高 μ  700 ────────────────────────────────────────────
python scripts/collect_go2_data_v3.py $BASE --seed 20260903 --num_episodes 35 \
  --brake_complete_ratio 1.0 --ood_static_friction_range 0.55 1.00 \
  --output_dir data/fric_B3_high

# ── B0  弱刹车 / 不刹车走出边缘  400 ──────────────────────────
python scripts/collect_go2_data_v3.py $BASE --seed 20260904 --num_episodes 20 \
  --brake_complete_ratio 0.0 --brake_target_ratio_range 0.0 1.0 \
  --ood_static_friction_range 0.22 1.00 \
  --output_dir data/fric_B0_weakbrake

# ── E  配对评测  3 × 60，seed 必须相同 ────────────────────────
for MU in "0.23 0.25" "0.39 0.41" "0.79 0.81"; do
  python scripts/collect_go2_data_v3.py $BASE --seed 20260907 --num_episodes 3 \
    --command_vx_range 0.5 2.0 --brake_complete_ratio 1.0 \
    --ood_static_friction_range $MU --ood_dynamic_ratio_range 0.70 0.70 \
    --output_dir data/fric_E_paired_${MU// /-}
done

# ── A1  低 μ 走路  300（不制动，只复用 COMMON 的一部分）────────
python scripts/collect_go2_data_v3.py $TERRAIN $DEPTH \
  --task go2_data_collection_robotlab --enable_cameras \
  --num_envs 20 --num_episodes 15 --max_steps 300 \
  --collection_mode normal --fault_type friction --lowfric_walk \
  --safe_ratio 0.0 --t_fault_min 30 --t_fault_max 90 \
  --command_vx_range 0.5 2.5 --command_vy_range -0.3 0.3 \
  --command_yaw_rate_range -0.3 0.3 \
  --ood_static_friction_range 0.22 0.32 --ood_dynamic_ratio_range 0.55 0.90 \
  --min_fail_consecutive 5 --seed 20260905 \
  --save_interval 100 --no_final_merge --output_dir data/fric_A1_lowmu_walk
```

`--depth_uint8` **必须出现在全部 7 条命令里**（含 A1），否则分片间 dtype 不一致，
训练侧加载会炸。B0 会打一条 `[WARN] edge schedule 的参考曲线是完全刹停距离` ——
预期行为，分析时按 `brake_is_complete` / `brake_target_vx` 分层。

### 9.0 怎么覆盖 / 去掉 BASE 里已有的参数

**带值的参数：直接在 `$BASE` 之后追加，后者赢。** argparse 对非 append 型
参数是后写覆盖前写：

```bash
python scripts/collect_go2_data_v3.py $BASE --terrain_profile flat_mesh --num_envs 4
#                                                   ↑ 覆盖 BASE 里的 cliff_brake
```

**`store_true` 型开关（`--depth_uint8`、`--dynamic_max_steps`、`--lowfric_walk`、
`--no_final_merge`）没有反向写法**，追加什么都关不掉。两个办法：

```bash
python scripts/collect_go2_data_v3.py ${BASE/--depth_uint8/}     # 字符串剔除
python scripts/collect_go2_data_v3.py $COMMON $TERRAIN $LENGTH   # 或者不拼那一组
```

上面把 BASE 拆成 `COMMON / TERRAIN / LENGTH / DEPTH` 就是为了第二种写法 ——
需要哪几组拼哪几组，比字符串替换稳。

**`$BASE` 千万不要加引号。** 靠的就是 shell 的分词，`"$BASE"` 会当成**一个**参数
传进去，argparse 直接报 `unrecognized arguments`。

### 9.1 为什么不靠截断 episode 省内存

`--max_steps` 与 `--post_brake_budget_s` 都不能当内存旋钮用，两者削掉的都是
**与 r 正相关**的那一头，等于把 `--dynamic_max_steps` 消除掉的信息性删失又请回来：

```
v3 实测 max_steps_budget 与 r：
  budget 250-300: n= 40  r均 0.65
  budget 300-350: n=182  r均 1.07
  budget 350-401: n= 78  r均 1.30
硬截 300 会截短 71% 的 episode，其中 1.7% 在 t_stop 确认之前就被切断。

v3 实测 corr(t_stop − t_brake, r) = +0.306：
  post_brake 2.5s 覆盖 80.7% 的刹停 / 3.0s 88.6% / 3.5s 95.2% / 4.0s 96.4% / 4.5s 98.8%
```

所以取 `post_brake 4.0`（96.4% 覆盖），`max_steps 500` 让硬上限不再咬
（v3 在 400 时已有 4% 撞顶）。**内存只用 uint8 这一个旋钮解决。**

`--early_stop_hold_steps` 同理保持 0：刹停后驻留占 26% 的帧，但 `drift_off` 的
刹停→翻覆延迟（n=33）中位 65 帧、90 分位 131 帧，hold=90 只捕获 67%，
省 0.2 GB 换掉三分之一的 `drift_off` 标签不划算。

## 10. CLI 参数说明

### 调度核心（本数据集的设计变量）

| 参数 | 含义 |
|---|---|
| `--brake_trigger edge` | 刹车由**到边缘的距离**触发，而不是固定帧号。固定帧号会让 r 变成不受控的副产物 |
| `--brake_margin_ratio_range LO HI` | 每 episode 采样 `r`，刹车在 `d_ray ≤ r·d̂_ref(v)` 时点火。**这是整套设计的受控变量** |
| `--brake_ref_linear / _quadratic / _offset` | 尺子 `d̂_ref(v) = a1·v + a2·v² + c`。必须与 μ 无关 |
| `--fault_trigger window` | 摩擦跌落在刹车**之前固定时间**触发，而不是固定帧。固定帧会让 0.3 m/s 有 4 s 冰面暴露而 2.5 m/s 只有 1 s——4 倍差异是混淆不是设计 |
| `--fault_window_range LO HI` | 逐 episode 采样该窗口（秒）。更长的窗口只让 spawn 后退，**不改变 r**，是在刹车块内拿到不同时长低摩擦行走的免费办法 |
| `--spawn_mode solve` | 由 (v, r, θ, extra, window) 反解 spawn，而不是从固定范围采 x₀ |
| `--spawn_extra_runway LO HI` | 独立于 v 的额外跑道，解除 `(d_edge, v)` 共线。因触发闭环，不改变 r |
| `--spawn_heading_range LO HI` | 相对目标边法线的初始朝向（弧度）。刹车块的朝向多样性来源 |
| `--spawn_lateral_margin` | 射线交点与侧边至少保留的距离，影响可行域上界 |
| `--spawn_accel / --spawn_settle_s` | 实测加速度 2.6 m/s² 与加速尾部余量，进 D₀ 的前两项 |
| `--spawn_max_retries` | (v,r,θ) 不可行时的重采上限。耗尽才报错 |

### 摩擦

| 参数 | 含义 |
|---|---|
| `--fault_type friction` | 扰动类型为摩擦跌落 |
| `--safe_ratio 0.0` | 不施加扰动的 episode 比例。刹车块取 0 = 每条都掉摩擦 |
| `--ood_static_friction_range LO HI` | 逐 episode 采样静摩擦 |
| `--ood_dynamic_ratio_range LO HI` | 动摩擦 = 静摩擦 × 该比值。**优先于** `--ood_dynamic_friction_range` |
| `--ood_dynamic_friction_range LO HI` | 独立采样动摩擦。用它可让静/动解耦（本数据集用比值版以保物理合理性） |

### 制动强度

| 参数 | 含义 |
|---|---|
| `--brake_complete_ratio` | 完全刹停（target_vx=0）的 episode 比例 |
| `--brake_target_ratio_range LO HI` | 非完全刹停时 `target_vx = v₀ × ratio`。`ratio=1.0` 即**完全不刹车** |

### episode 长度

| 参数 | 含义 |
|---|---|
| `--max_steps` | 硬上限 |
| `--dynamic_max_steps` | 按 (v, D₀) 逐 episode 解算步数。固定步数会在最需要减速尾段的格子制造删失 |
| `--post_brake_budget_s` | t_brake 之后预留的刹停+驻留时间 |

### outcome 标注

| 参数 | 含义 |
|---|---|
| `--stop_speed_thresh` | 判定刹停的平面速度阈值。**零指令下有 ≈0.17 m/s 的怠速地板**（平衡抬腿），阈值必须在它**之上**：取 0.10 会让站住的机器人永远判不出 `t_stop`，v3 因此伪造了 18.7% 的 `no_stop`。取 **0.20** |
| `--stop_sustain_steps` | 速度需连续低于阈值的帧数。配 0.20 阈值时取 **10**（0.2 s），避免减速末段瞬时穿越被误判 |

### 深度图编码

| 参数 | 含义 |
|---|---|
| `--depth_uint8` | 深度以 uint8 存储，内存/磁盘 4 倍压缩。**必须在所有块上一致开启** |
| `--depth_clip_min / _max` | 量化区间。**不引入新的裁剪**——相机 `CLIP_RANGE=(0.3, 3.0)`，超量程直接返回 `inf`，实测 0 个有限像素落在区间外，`np.clip` 是空操作。这两个参数只是告诉量化器量程在哪，好把 254 个码字铺满。**若日后改相机量程，必须同步改这两个值** |

量化损失（实测）：均值 2.66 mm / 最大 5.32 mm / 相对 0.337% / PSNR 58.9 dB，是
逐帧自然抖动（中位 8.5 mm）的 1/8；而任务尺度（平台落差 0.40–0.60 m、d_hold 0.50 m）
跨 38–56 个量化台阶。码字 **255 保留给无回波**，顺带消掉原始数据里 27.24% 的
非有限像素。demo 里记 `depth_encoding` 字段。

### 稳定判据

| 参数 | 含义 |
|---|---|
| `--tracking_error_ratio / _abs / _hold_s` | 速度跟随稳定判据。fault 只在 settled 之后才允许触发 |

### 环境与输出

| 参数 | 含义 |
|---|---|
| `--terrain_profile cliff_brake` | 覆盖 task 地形为 `CLIFF_DETECTION_TERRAINS_CFG`（10×10 = 100 格，proportion 55/15/15/15）。与 task 默认现在是同一个 cfg，写上更明确 |
| `--platform_half_width` | 触发与标注用的平台半宽，**必须与实际 tile 一致** |
| `--num_envs` | **必须 < 可 spawn 格数**（`cliff_brake` 是 100），否则脚本报错。同格的两台机器人会互相进入深度视野。20–40 都合适 |
| `--command_vx/vy/yaw_rate_range` | 覆盖 task 的指令范围 |
| `--num_envs × --num_episodes` | 总条数 = 两者相乘 |
| `--save_interval` / `--no_final_merge` | 分片大小 / 跳过最终合并 |
| `--lowfric_walk` | 低摩擦走路模式，仅支持 `--collection_mode normal --fault_type friction` |
| `--t_fault_min / _max` | frame 触发模式下的 fault 帧号范围 |
| `--min_fail_consecutive` | failure 标签的连续帧平滑阈值 |

---

## 11. 验收判据

### 11.1 冒烟 100 条（正式采集前必过）

1. `braking_config.json` 里 `depth_uint8: true`、`terrain_profile: cliff_brake`
2. demo 的 `depth_encoding.dtype == "uint8"`，且 `obs['image'].dtype == uint8`
3. **`no_stop` < 5%** —— stop 阈值修好的直接证据（v3 用 0.10 时是 18.7%）
4. **spawn 重采率 < 1%** —— 模拟预测 0.003/条
5. episode 均长 ≈ 350，**无 episode 撞到 `max_steps 500`**
6. 启动日志里没有 spawn 隔离的 WARN

### 11.2 每个 B 块跑完

1. **`no_brake` < 3%**（`off_before_brake` 不设上限，它是有用的未制动轨迹，见 §8）
2. `spawn 重采率 < 5%`
3. **各速度档的 overrun 率互相接近** —— 这是尺子正确性的判据。
   参考：尺子错时是 31/60/50/49/63%，尺子对时是 26/23/18/28/26%
4. B0 的 `brake_target_vx` 应均匀铺满 `[0, v₀]`
5. `corr(r, μ_d)` 与 `corr(r, v)` 的 |值| < 0.05（模拟预测 0.013）

### 11.3 全部跑完

用 2400 条一次性联合拟合：

```
P(overrun) = Φ( (r*(μ_d, v) − r) / σ ),   r* = (c·v/μ_d + h) / (1.15v + 0.50)
```

若 `c ≈ 0.17`、`h ≈ 0.50` 在连续 μ_d 上稳定成立，尺子就从拟合升格为物理定律，
可外推到未采过的 μ（包括 OOD 的 0.05–0.10）。事后按 μ_d 分箱检查形式有无系统偏离。

E 块跑之前先验 RNG 对齐：三次运行同 seed、同 env 的 `actions[:t_fault]` 应逐位相同。
μ 不影响可行性（尺子 μ 无关），重采次数一致，理论上不会错位，但值得实际对一次。

---

### 11.4 触发阈值的物理下限（冒烟 100 条实测）

Go2 的机体中心靠不到边缘 0.30 m 以内——再近前脚就悬空了。所以
`r · d̂_ref(v) < 约 0.45 m` 的组合**物理上不可达**，触发永远不点火：

```
outcome            触发阈值   d_ray 全程最小   d_edge 全程最小
no_brake            0.23 m      0.36            0.30
off_before_brake    0.28 m      0.37            0.30
off_before_brake    0.30 m      0.34            0.34
no_brake            0.37 m      0.40            0.38
```

按当前区间蒙特卡洛，**9.7% 的 (v, r) 落在这个死区**，低速端最严重
（v=0.3 要 r ≥ 0.53 才够得着，v=2.5 只要 r ≥ 0.13）。

**结论是不改 r 下界。** 死区的产物里 3/5 是有用的 `off_before_brake`，只有 2/5 是
`no_brake` 废片；把 r 下界抬到 0.30 能把死区压到 2.8%，但同时砍掉高速端
r ∈ [0.15, 0.30] 那段可达的、位于 probit 下 10% 分位附近的失败尾。评估过的选项：

| 方案 | 死区 | corr(r,v) | 代价 |
|---|---|---|---|
| A 保持 v 0.3–2.5, r 0.15–1.15 | 9.7% | +0.001 | 约 2% 的 `no_brake` 废片 |
| B `r·d̂_ref<0.45` 时重采 | 0% | **−0.158** | 破坏 r ⊥ v，不可接受 |
| C v 0.3→0.8, r 0.15–1.15 | — | +0.001 | 砍掉 13% 的 episode |
| D v 0.3–2.5, r 0.30–1.15 | 2.8% | +0.000 | 砍掉高速端失败尾 |
| **采用：v 0.5–2.5, r 0.15–1.15** | **7.5%** | **+0.002** | 见下 |

**最终取 v 下界 0.5、r 下界不动。** 死区只从 9.7% 降到 7.5%（死区本来就在低 r 而非
低 v），真正的收益是帧数：低速档单条最长（v<0.8 均长 382 帧 vs 全体 277）而信息量
最少——v=0.55 时 `d_stop` 在整个 μ 区间上只变 0.46 m，还不到 `d_hold = 0.50 m`
这个与摩擦无关的常数，**μ 自适应在那里近乎不可辨识**。按"每帧携带的 μ 信息"折算，
最低速档只有最高速档的 1/6.4：

```
v_max      n    T均   占总帧   d_stop 随 μ 的跨度   每帧 μ 信息
0.3-0.8   13   382   18.0%        0.46 m           1.00x
0.8-1.5   28   295   29.8%        0.96 m           2.71x
1.5-2.0   16   289   16.7%        1.45 m           4.20x
2.0-2.6   33   250   29.8%        1.91 m           6.39x
```

新区间下重新验算：`r*` 全空间仍是 [0.25, 1.03]，r 上界 1.15 照旧覆盖；
触发点出视距 6.6% → 7.3%（略升，因为低速端的短触发距离没了）；
预测 spawn 重采率 0.006 次/episode；corr(r, v) = +0.002。

另有 1/5 是**侧边先出界**：θ = −36.3° 时前向射线还剩 2.44 m（阈值 2.22 m），
而 `d_edge` 已经是 −0.22。`--spawn_lateral_margin` 只约束射线交点，管不住途中横漂。
占比 1/100，暂不处理。

### 11.5 slope 对 r\* 拟合贡献为零

冒烟里 slope 12/12 全是 `safe_stop`，与早先"slope 在该策略下 0/50 失败"一致。

```
dangerous_platform   n=57  overrun=30, safe_stop=23, drift_off=3, off_before_brake=1
deep_pit             n=18  overrun=11, safe_stop=4,  off_before_brake=2, drift_off=1
stairs               n=13  safe_stop=9,  no_brake=2, overrun=2
slope                n=12  safe_stop=12                      ← 零失败
```

**probit 拟合必须按 `terrain_key` 排除 slope**，否则这些恒为 0 的响应会把 σ 撑大。
地形本身保留：WM 需要"看起来不同但实际安全"的负样本。`hazard_dir=+1` 在几何上
正确（平台在最低处，越界是上坡），但经验上更接近 0，等 n 够了再定。

---

## 12. 仍然开着的问题

| 问题 | 状态 |
|---|---|
| `1/μ` 的形式只有两个 μ 档支撑 | 本数据集的连续 μ 覆盖就是为了回答它 |
| r\* 在不同批次间有 0.78 / 0.90 / 1.02 的差异 | σ 已收到 0.24，但均值差约 4 个标准误，需一次单变量对照 |
| 地形间 r\* 差 0.07（platform / pit / stairs） | `hazard_dir` 已记录，分层分析即可；n 还不够定论 |
| **5.2% 的 episode 触发点落在 2.8 m 视距外** | r 上沿从 1.55 收到 1.15 后从 23% 降到 5.2%。这部分刹车指令在图像和 `priv_state` 里都没有可观测起因（`priv_state` 是 R^10，**不含 `d_edge`**），对 margin head 是不可学样本。备选方案：把 `d_edge_ray` 加进 `priv_state`（R^10 → R^11，训练时监督 g(x)、推理时不用），需同步改 latent-safety 侧维度。**未决** |
| 高速 + 极低 μ 下物理上必须盲刹 | `0.170v/μ_d > 2.60` 时即便策略完美也看不见边缘。占 B 系 0.7%，只在 μ_d < 0.163 且 v > 2.0 出现。规模小，暂不处理，但那一格若 critic 表现差，未必是 μ 通道的问题而是时序记忆的问题 |
| 训练侧图像加载仍按 float32 | 需改成 `img = q.astype(np.float32) / 255.0`（在 latent-safety 仓库）。**未改** |
