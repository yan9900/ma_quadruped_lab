# 用 RobotLab MoE-CTS policy 在 LeggedLab 采集数据

本文说明如何把 `go2_rl_robotlab` 训练的 MoE-CTS student 直接放进 LeggedLab 的
`collect_go2_data_v3.py` 使用。**只替换 policy，物理仿真上的差异一律不改**
（执行器模型、摩擦合成方式、机器人模型、vx 指令上限）。目的是先确认这条路可不可行，
并量化这些偏差造成的退化。

## 1. 结论先行

- 采集只用 student。它的输入是 45 维 × 10 帧 = 450 维本体观测，不含任何 privileged
  信息；teacher（275 维，含 187 个高程点）只在 RobotLab 训练时使用。因此 WM 记录什么
  priv（包括全高程图）和这个 policy 无关，两边可以独立设计。
- LeggedLab 的 `go2_data_collection_robotlab` 任务在观测维度、缩放、单帧顺序、历史初始化
  以及动作映射上已经和 RobotLab 一致。剩下三个接口差异（加载方式、历史排列、关节顺序）
  都由 `utils/robotlab_policy.py` 处理，env 和采集主流程不需要改。

## 2. 文件位置

| 路径 | 内容 |
|---|---|
| `logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/` | 从 HF 下载的 v1 policy，附 `SOURCE.md`（来源、sha256、网络接口） |
| `utils/robotlab_policy.py` | `RobotLabStudentPolicy`（批量推理 adapter）和 `check_env_compat`（启动时检查） |
| `scripts/verify_robotlab_policy.py` | 离线逐位校验 adapter 与导出模型，不需要 Isaac |
| `scripts/collect_go2_data_v3.py` | 新增 `--robotlab_policy PATH` 参数 |

`logs/` 被 gitignore，policy 文件不进 git。`/home/lcy/LeggedLab/logs` 是指向
`legged_lab/logs` 的符号链接，所以从仓库根目录启动（如 `run_flat_brake_ref.sh`）时路径同样有效。

### 为什么选 v1（symmetry_v1_77k_0.7006_bb4a078）

HF 上四个 RobotLab checkpoint（v3.3 / v4.2 / symmetry v5.1 / symmetry_v1）的网络接口
完全一样，所以只按训练设定来选。v1 相比 v4.2 有两处和刹车数据直接相关的改动
（见 go2_rl_robotlab 的 CHANGELOG 2026.7.6）：

| | v4.2（main 分支） | v1（bb4a078） |
|---|---|---|
| 零指令采样 | 只把 vx、vy 置零，yaw 保留原采样值（多数是"原地转"） | `[0, 0, 0]`，真正的零 |
| 站立时关节偏离默认姿态的惩罚 | ×1，和行走时一样 | 指令为 0 时 ×10 |

采集器的完全刹车指令正是 `[0, 0, 0]`，对应 v1 专门训练过的情形。

## 3. adapter 做了什么

```
obs["policy"]  [N, 10*45]   LeggedLab：按帧展开，关节按类型排
   │ ① 每帧的 joint_pos / joint_vel / last_action（各 12 维）置换到 RobotLab 关节顺序
   │ ② 按帧 [N,10,45] → 按项 [ang×10 | grav×10 | cmd×10 | jp×10 | jv×10 | act×10]
   ▼
student_moe_encoder(历史) → latent [N,32]
actor(cat(latent, 当前帧45)) → action [N,12]（RobotLab 关节顺序）
   │ ③ 逆置换
   ▼
action [N,12]（LeggedLab 关节顺序）→ env.step
```

- 关节顺序：RobotLab 按腿排（FL 的 hip/thigh/calf 连在一起），LeggedLab 的 USD 按类型排
  （先 4 个 hip，再 4 个 thigh，再 4 个 calf）。置换表由 `robot.data.joint_names` 按名字
  计算，不写死。当前的值是 `robotlab_from_env = [0,4,8,1,5,9,2,6,10,3,7,11]`。
- 为什么不直接调用导出模型的 `forward`：它只支持 batch = 1，内部自带历史，
  `reset()` 会把历史清零。而 RobotLab 训练时和 LeggedLab 的 `CircularBuffer` 一样，
  reset 后用第一帧填满历史。adapter 直接调用 `student_moe_encoder` 和 `actor` 两个
  子模块（它们支持任意 batch），历史完全交给 env 维护，adapter 自身无状态。
- 按帧展开的布局保留在 env 里，所以 `apply_command_override_to_obs` 按帧修改 command
  的逻辑不受影响。

## 4. 对齐状态

### 4.1 已对齐（`check_env_compat` 启动时自动检查，不一致直接报错）

| 项 | 值 |
|---|---|
| 单帧顺序 | ang_vel(3), gravity(3), cmd(3), joint_pos(12), joint_vel(12), last_action(12) |
| obs 缩放 | ang 0.25，jv 0.05，其余 1.0 |
| 历史长度及初始化 | 10 帧；reset 后第一帧填满 |
| 动作映射 | q = default + 0.25·a，0.005 s × 4 = 50 Hz |
| 默认站姿 | hip ±0.1，前 thigh 0.8 / 后 thigh 1.0，calf −1.5 |
| last_action 的含义 | policy 原始输出（没有 clip，阈值 ±100 实际不起作用） |
| obs 噪声 | 采集器第 845 行强制 `add_noise=False`（RobotLab play/部署时也不加噪声） |

### 4.2 有意保留的偏差（本阶段不改，只记录）

| 项 | RobotLab 训练 | LeggedLab 采集 | 影响 |
|---|---|---|---|
| vx 指令范围 | 平地 ±2.0；坡面 ±1.5；台阶、障碍 ±1.0 | 任务默认 `(1.0, 2.5)` | 2.0–2.5 超出训练范围 |
| 摩擦合成方式 | average；足端 μ 在 [0, 2] 随机，地面 1.0，有效 μ 约 0.5–1.5 | multiply（为了让低 μ 故障生效） | 有效 μ < 0.5 对 policy 是 OOD；低 μ 下的行为里含 policy 外推成分 |
| 执行器 | Unitree GO2HV（扭矩-速度曲线，friction 0.01，0–4 个物理步延迟） | DCMotor 23.5 Nm / 30 rad/s，无延迟 | 动态响应不同 |
| 机器人模型 | Unitree URDF | 本地 USD | 惯量不同 |
| 基座附加质量 | ±1 kg | `base_config.py` `EventCfg.add_base_mass` 为 **±5 kg** | 整机约 15 kg，+5 kg 超出训练范围很多；很可能是冒烟测试里蹲伏行走的原因（见 §7） |
| 足端摩擦随机 | [0, 2]，average 合成 | static 0.6–1.0 / dynamic 0.4–0.8，multiply 合成 | 有效 μ 在训练范围内 |

## 5. 地形怎么配

**这个 policy 是盲的**：student 不读高程图。地形不需要在 policy 侧做任何配置，
它只决定物理接触和失败标签。需要考虑的只有两件事：地形是否在 policy 训练见过的范围内，
以及采集器的 spawn 约束。

### 5.1 RobotLab 训练见过的地形

| 类型 | 占比 | 参数范围 | 训练时的 vx 上限 |
|---|---|---|---|
| flat | 15% | — | ±2.0 |
| 坡面上/下、粗糙坡、wave | 30% | slope 0.1–0.568 | ±1.5 |
| 台阶上/下 | 35% | 台阶高 0.05–0.257 m | ±1.0 |
| 障碍 | 20% | 高 0.05–0.275 m | ±1.0 |
| gap、梅花桩 | 0% | — | — |

对照 LeggedLab 的地形：
- 平地（`FLAT_MESH`）：在训练范围内。
- 倒金字塔坡 slope 0.15–0.25：在训练范围内，但训练时坡面速度只到 1.5。
- 0.4–0.6 m 的平台落差（`dangerous_platform`）、0.3–0.6 m 的坑壁（`deep_pit`）：超出训练见过的
  最大台阶高度 0.257 m。这正是要采的失败，不是问题。

### 5.2 采集器的 `--terrain_profile`

| 取值 | 实际地形 | 格数 | 用途 |
|---|---|---|---|
| `task`（默认） | 任务配置里的 `CLIFF_DETECTION_TERRAINS_CFG` | 5×5 = 25 | 正式采集 |
| `cliff_brake` | 同上，但由采集器覆盖（开随机 spawn，`--terrain_seed` 生效） | 25 | 正式采集，需要固定地形时用 |
| `flat_mesh` | `FLAT_MESH_TERRAINS_CFG`，10 m × 10 m 平地 mesh | 25 | 可行性验证、平地刹车参考；没有边缘，不能配 `--brake_trigger edge` |

约束和注意事项：
- `--num_envs` 必须小于可 spawn 的格数，两个 profile 都是 ≤ 24。否则两台机器人会进到
  同一格，出现在彼此的深度图里，采集器会直接报错。
- 当前工作区的 `CLIFF_DETECTION_TERRAINS_CFG` 只启用了 `slope`（platform_width=10.0），
  `dangerous_platform`、`deep_pit`、`stairs` 都被注释掉了。要采落差或碰撞类失败，需要在
  `terrains/terrain_generator_cfg.py` 里重新启用。
- 该配置上方注释写着"所有子地形 platform_width 都是 8.0"，但当前 slope 是 10.0，而
  `--platform_half_width` 默认 4.0。用 edge 触发或 `--spawn_mode solve` 之前请核对这个值。
- 地面材质：static / dynamic 都是 1.0，multiply 合成。有效摩擦就是足端的 μ。

### 5.3 推荐的分阶段地形

| 阶段 | 地形 | 为什么 |
|---|---|---|
| G1、G2 | `flat_mesh` | 排除地形因素，只看 policy 移植本身，以及执行器/摩擦差异造成的退化 |
| G3 | `task` / `cliff_brake`，先只开 slope（训练范围内），再开平台/坑 | 逐步引入超出训练范围的地形 |

## 6. 验证步骤

所有命令都从 `/home/lcy/LeggedLab` 启动，解释器用
`/home/lcy/miniconda3/envs/env_isaaclab/bin/python`（下文记作 `$PY`）。
下文 `$POL` 指 `logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt`。

**G0a 离线逐位校验（不需要 Isaac）**
```bash
python legged_lab/scripts/verify_robotlab_policy.py --policy legged_lab/$POL
```
通过标准：`max |adapter - export| < 1e-4`（实测 2.1e-5）。这一步同时验证了关节置换、
历史重排和动作逆置换。

**G0b 启动检查**：带 `--robotlab_policy` 启动时，`check_env_compat` 会校验 obs 缩放、
action_scale、控制周期和默认站姿，并打印 env 关节顺序和置换表。

**G1 平地行走**：vx 分训练范围内和超出范围两档，分开报告。
```bash
$PY legged_lab/scripts/collect_go2_data_v3.py --task go2_data_collection_robotlab --headless \
  --robotlab_policy $POL --terrain_profile flat_mesh --spawn_mode range \
  --collection_mode normal --fault_type none --safe_ratio 1.0 \
  --num_envs 24 --num_episodes 4 --max_steps 250 --seed 1 --terrain_seed 1 \
  --command_vx_range 1.0 2.0 --output_dir ./data/robotlab_v1_smoke/g1_in --output_name g1_in
# 超出范围的一档：--command_vx_range 2.0 2.5，其余不变
```
看：起步 1 秒内和整个 episode 的摔倒率、vx 跟踪误差。

**G2 平地刹车**：参照 `run_flat_brake_ref.sh`，加上 `--robotlab_policy $POL`。
看：停车率（`v_eps=0.20`，和采集器口径一致）、停车距离、停稳后能否站住。

**G3 原采集场景**：`--terrain_profile task`，按 5.3 节逐步开启地形。

G1、G2 都应该用你现在的 LeggedLab policy 在相同 seed 和指令下配对跑一遍作对照，
用差值来判断偏差造成的影响，不要只看绝对值。

## 7. 冒烟测试结果（2026-09-29）

配置：`flat_mesh` 地形，`normal` 模式，无故障，8 env × 2 episode，每个 episode 200 步，
vx 在 1.0–2.0 之间随机，vy、yaw 用任务默认范围（±0.3），seed 1。所有物理偏差都没改。

| 指标 | 结果 |
|---|---|
| 完整跑完 200 步 | 15 / 16 |
| vx 跟踪（第 1 秒之后，15 条完整 episode） | 平均误差 −0.06 m/s，RMSE 0.077（稳定地略慢于指令） |
| 稳态基座高度 | **0.21–0.36 m**（RobotLab 训练目标 0.38 m） |
| 失败 1 条（ep8） | 起步加速时基座从 0.40 m 降到 0.21 m，第 37 步因接触被终止 |

结论：加载链路是通的，行走能跟上指令。但机器人在蹲伏行走，而且同样的指令下不同
episode 的高度差别很大（例如 1.60 m/s 对应 0.33 m，1.57 m/s 对应 0.25 m），这说明高度
取决于每个 env 的随机参数，而不是指令本身。

首要怀疑：±5 kg 的基座附加质量（训练时只有 ±1 kg）。已经排除的原因：扭矩上限。
RobotLab 的 GO2HV 上限是 20.2–23.4 Nm，比 LeggedLab 的 23.5 Nm 还略低。

支持这个怀疑的两条证据：
- `add_base_mass` 的采样模式是 `startup`，也就是每个 env 抽一次、整个运行期间不变。
  v1 的结果在同一个 env 内高度很稳定（env3：0.236 / 0.220；env5：0.358 / 0.353），
  不同 env 之间差异很大，和这个模式吻合。
- 同一个任务、同样的 ±5 kg 设置下，旧 LeggedLab policy（`data/flat_brake_ref`，μ = 0.80，
  从第 1 秒到故障触发前）的稳定高度也分散在 0.24–0.37 m，中位数 0.32–0.35 m。
  所以高度分散是这个 env 设置本身带来的，不是 v1 特有的问题。但 v1 整体更低
  （中位数约 0.29 m，最低 0.21 m）。

**验证（同日，`--base_mass_range -1 1`，其余设置完全相同）**

| | ±5 kg（任务默认） | ±1 kg（RobotLab 训练范围） |
|---|---|---|
| 跑满 200 步 | 15 / 16 | **16 / 16** |
| 稳态基座高度 | 0.207–0.358 m，中位数 0.289 | **0.280–0.343 m，中位数 0.311** |
| 同一 env 两次高度之差 | 最大 0.04 m | 最大 0.04 m |
| vx 跟踪 RMSE | 0.077 | 0.080 |

采样到的 base 质量是 6.62–7.96 kg。高度的分散范围从 0.15 m 缩小到 0.06 m，剩下的差异
主要来自速度：vx ≥ 1.8 时约 0.28–0.31 m，vx ≤ 1.3 时约 0.33–0.34 m。确认蹲伏的主要原因是
±5 kg 的附加质量。

高度仍然低于训练目标 0.38 m，原因尚未确定。可能是高速步态本身就会更低，也可能是
URDF 和 USD 的 base 坐标原点不同，这次没有深究。

## 8. 已知问题（与 policy 来源无关，本次没有修改）

`apply_command_override_to_obs` 里的 `env.actor_obs_buffer.buffer[:, -1, ...] = ...`
写在一个副本上：IsaacLab 的 `CircularBuffer.buffer` 返回的是 `clone()`，所以写入不会保存。
真正生效的只有同一行对 `obs_dict["policy"]` 当前帧的修改。结果是，指令切换那一步
policy 看到的是新指令，但缓存里那一帧仍是旧指令，之后 9 步的历史里会带着这一帧旧值。
新旧两种 policy 都受这个影响。
