# Go2 RobotLab → LeggedLab 对齐与复现实验方案

## 1. 文档用途

本文档用于在 LeggedLab 中新增一套与 `go2_rl_robotlab` 尽量可比的 Go2 训练配置。它应当能够作为一个独立任务说明直接交给新的开发对话使用。

当前优先目标是复现稳定的平地 velocity-tracking policy，而不是立即复现完整的 MoE-CTS。平地基线通过后，再以相同 observation schema 和评测协议扩展到多地形，最后决定是否移植 concurrent teacher-student 训练结构。

参考仓库：

- `/home/lcy/go2_rl_robotlab`
- 参考环境：`source/robot_lab/robot_lab/tasks/go2/env_cfg.py`
- 参考 reward：`source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py`
- 参考训练配置：`source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py`
- 参考 MoE-CTS：`source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py`、`source/rsl_rl/rsl_rl/algorithms/moe_cts.py`

目标仓库：

- `/home/lcy/LeggedLab`
- Go2 配置：`legged_lab/envs/go2/go2_config.py`
- reward 实现：`legged_lab/mdp/rewards.py`
- 基础环境：`legged_lab/envs/base/base_env.py`
- 基础配置：`legged_lab/envs/base/base_env_config.py`

## 2. 核心原则

不要把“reward 权重一致”当作“任务已经对齐”。一个 locomotion MDP 至少包括：

1. robot asset 和物理参数；
2. action 定义和控制频率；
3. command 分布和课程；
4. observation 定义、顺序、尺度、噪声和历史初始化；
5. reward 函数的数学语义、参数、权重和时间尺度；
6. reset 分布和 domain randomization；
7. termination/truncation；
8. terrain 分布和 curriculum；
9. PPO/网络结构；
10. 导出和部署端的预处理、关节顺序及历史 buffer。

对齐顺序应为：

```text
physics/action
    → command/reset/history semantics
    → observation
    → reward
    → randomization/termination
    → PPO hyperparameters
    → teacher-student architecture
```

每次实验只改变一个轴；不允许同时修改 reward、随机化、地形和网络结构后仅根据最终 reward 判断原因。

## 3. 建议新增的配置，不覆盖现有 Go2-Flat-v0

新增并注册独立任务：

```text
Go2-Flat-RobotLab-Parity-v0
Go2-Rough-RobotLab-Parity-v0       # 平地通过后再创建
```

建议类名：

```python
Go2RobotLabRewardCfg
Go2RobotLabFlatEnvCfg
Go2RobotLabFlatAgentCfg
Go2RobotLabRoughEnvCfg
Go2RobotLabRoughAgentCfg
```

`Go2RobotLabFlatEnvCfg` 最好直接继承 `BaseEnvCfg`，不要继承当前 `Go2FlatEnvCfg` 后依赖零散覆盖。原因是当前 flat 配置已经混入 gait、mirror、upward、camera、较宽 command 和特定 reset 设置，后续容易发生隐式继承。

现有 `Go2-Flat-v0` 和 WM 数据收集配置保持不变，确保旧实验可复现。

## 4. 第一阶段：补齐并核对 reward 实现

### 4.1 需要补充或重写的函数

以 RobotLab 的数学实现为基准，将以下函数适配到 LeggedLab 的 `BaseEnv` API：

- `joint_power`
- `action_smoothness_l2`
- `joint_pos_penalty_l1`
- `hip_pos_penalty_l1`
- `feet_regulation`
- RobotLab 语义的 terrain-relative `base_height_l2` 辅助函数

不要仅按名称判断已有函数等价。当前已知差异：

- LeggedLab `energy` 使用向量 norm；RobotLab `joint_power` 使用各关节 `abs(tau * qdot)` 之和。
- 两边二阶 action smoothness 的 reset 初始帧屏蔽语义需要核对。
- LeggedLab 当前 `base_height_l2` 的无效 ray 判断是跨 batch 的全局判断；RobotLab 按 env 判断并逐 env fallback。
- reward 是否由 manager 乘以 control-step `dt` 必须通过运行日志确认，不能只比较配置 weight。

新增函数应有 shape、frame、单位、reset 边界行为的注释，并至少做一次小规模数值 smoke test：输出必须为 `[num_envs]`、有限值且不存在跨 env 污染。

### 4.2 Parity reward 集合

第一版只启用 RobotLab 的 reward 集合及权重：

| term | weight | 关键参数 |
|---|---:|---|
| `track_lin_vel_xy_exp` | 2.0 | `std=0.5` |
| `track_ang_vel_z_exp` | 1.0 | `std=0.5` |
| `lin_vel_z_l2` | -2.0，后续 curriculum 到 0 | body frame |
| `ang_vel_xy_l2` | -0.05 | body frame |
| `joint_acc_l2` | -1e-7 | 指定 12 个 Go2 joints |
| `joint_power` | -2e-5 | sum of `abs(tau*qdot)` |
| `joint_torques_l2` | -1e-4 | applied torque |
| `base_height_l2` | -1.0，后续 curriculum 到 -10 | target 0.38 m |
| `action_rate_l2` | -0.01 | 一阶差分 |
| `action_smoothness_l2` | -0.01 | 二阶差分 |
| `undesired_contacts` | -1.0 | thigh/calf，threshold 5 N |
| `joint_pos_limits` | -2.0 | soft limits |
| `feet_regulation` | -0.05 | target base height 0.38 m |
| `hip_pos_penalty_l1` | -0.05 | command-aware |
| `joint_pos_penalty_l1` | -0.01 | thigh/calf，command-aware |

第一版关闭当前 LeggedLab 特有的：

- `upward`
- `feet_gait`
- `joint_mirror`
- `feet_contact_without_cmd`
- 当前版本的 `feet_slide`、`feet_height_body`

这些项不是错误；关闭它们只是为了减少行为塑形差异。完成 parity 后可以通过消融实验逐项加回。

### 4.3 Reward 对比不能只看总和

训练日志必须按 term 记录：

- raw term mean/std/quantiles；
- weighted term mean；
- `mean(abs(weighted term)) / mean(total positive reward)`；
- episode 内累计贡献；
- reset 前 0.5 秒和普通行走阶段分别统计。

如果某个 penalty 的量级比参考实现大 10 倍，应先查数学实现、单位、频率和传感器值，不要立刻把 weight 除以 10。

## 5. 第二阶段：平地 parity 配置

### 5.1 Physics/action 必须显式记录

至少对齐并输出 manifest：

- simulation dt：0.005 s；
- decimation：4，即 policy 50 Hz；
- action：`q_target = q_default + 0.25 * action`；
- stiffness：25；
- damping：0.5；
- effort/velocity limit；
- self-collision；
- contact/rest offsets；
- friction combine mode；
- robot body/joint 顺序；
- default joint pose 和 base initial height；
- actuator delay。

注意：当前两个仓库的 Go2 asset、默认站姿和 base 高度并不完全一致。必须选择一个明确策略：

1. 严格 parity：让 LeggedLab 使用 RobotLab 同源 asset/default pose；或
2. 近似 parity：保留 LeggedLab asset，但把差异写进 manifest，不能把行为差异全部归因于 reward。

第一轮关闭 actuator/action delay；稳定后再把延迟作为 curriculum 加入。

### 5.2 Observation schema

平地标准 PPO 阶段建议：

- policy：45 维单帧普通感知 × 10 帧 = 450；
- critic：当前帧 privileged observation，不使用 10 帧 critic history；
- 后续若准备无缝切换 MoE-CTS，额外暴露 `single_obs` 45 维；
- rough 阶段不要把 actor history 改回 1。

单帧 policy 顺序固定为：

```text
base angular velocity (3)
projected gravity (3)
velocity command (3)
joint position relative to default (12)
joint velocity relative/default semantics (12)
previous action (12)
```

需要逐项对齐：

- body/world/yaw frame；
- scale；
- noise；
- clipping；
- joint order；
- previous action 是 raw、clipped 还是 delayed action；
- reset 时 history buffer 是补零还是复制第一帧。

历史初始化必须在训练、Isaac Sim 推理和 MuJoCo 推理中一致。建议 reset 后用第一帧填满 history；如果保留补零冷启动，则必须在训练中完整暴露相同冷启动。

### 5.3 Commands

初始平地实验采用小 command range：

```text
lin_vel_x: [-0.5, 0.5]
lin_vel_y: [-0.3, 0.3]
ang_vel_z: [-0.5, 0.5]
resampling: 5 s
```

稳定后再实现 RobotLab 的 command-range curriculum、zero-command curriculum、极值 command 采样和 terrain-dependent limits。

训练中保留一定比例的 command step，使策略学习加速瞬态；部署和 WM 数据收集可以使用 0.3–1.0 s command ramp。评测必须同时包含 step 与 ramp，避免用 ramp 掩盖 policy 本身的起步问题。

## 6. Reset、randomization 与数据收集分离

不要让一个配置同时承担 policy 训练和 WM 数据收集。

### 6.1 Policy 训练 curriculum

按以下顺序逐层加入，每层通过固定评测后再进入下一层：

1. nominal reset、固定摩擦、无 delay、无 push；
2. joint reset scale 0.9–1.1；
3. root linear/angular velocity 小扰动；
4. friction randomization；
5. base/other mass 和 COM；
6. actuator gains 和 motor zero offset；
7. action delay 0–4；
8. periodic push；
9. 最后扩大到 RobotLab 的完整 reset/randomization 范围。

每个 randomization 必须记录：

- 采样分布而不只是 min/max；
- startup/reset/interval 模式；
- 是否 per-env；
- 是否 recompute inertia；
- 各随机变量是否相关；
- 实际采样后的均值、标准差和 quantiles。

### 6.2 WM 数据收集配置

使用独立 eval/data 配置：

- 关闭 reset pose/velocity randomization；
- 可保留明确设计的 OOD 场景变量；
- 使用 command ramp；
- 标记 reset 后 history warmup；
- critic filter 冷启动期间不输出有效安全裁决；
- 保存 `termination_reason`，不要只保存 done；
- policy 训练配置不因数据收集便利性而修改。

## 7. Termination 设计：跌倒与撞墙必须分开

### 7.1 推荐结论

平地 walk-only policy：

- base/chassis 倒地接触：立即 terminal；
- head 碰撞：记录并可加 penalty，但不作为即时 terminal；
- thigh/calf：reward penalty，不 terminal；
- timeout：truncation，可 bootstrap，不是失败 terminal。

带墙或导航场景：

- base 倒地仍立即 terminal；
- head-wall 接触使用较高阈值和 dwell/debounce，例如连续 3–5 个 policy step；
- 极大冲击可以单独设置 immediate impulse terminal；
- 日志中区分 `fall_base_contact`、`persistent_head_collision`、`timeout` 等原因。

不要使用“任意历史帧 head force > 1 N 就立即 reset”作为通用撞墙规则。它会把擦碰、落地振动和真正持续撞墙混在一起，也会阻止策略学习碰撞后的恢复或退让。

### 7.2 为什么 walk policy 不应默认用 head-wall terminal

如果训练场景没有墙，这个条件不会提供有效学习信号，只会在跌倒时比 base contact 更早截断。如果训练场景有墙但 actor 没有视觉/距离等 exteroception，policy 在碰撞前无法知道墙的位置；termination 不能教会它主动绕墙，只能影响碰撞后的动作。

因此需先定义职责：

- 导航层负责不向墙发出危险命令；
- locomotion policy 负责 tracking、平衡和扰动恢复；
- safety filter/WM 负责部署时风险判断；
- 环境的 persistent head collision terminal 负责结束无意义或危险 rollout。

若确实希望 locomotion policy 学会接触后后退，应给它可观测的接触信号或足够的本体反馈，并配合 head-contact penalty、恢复命令和持续碰撞终止，而不是仅添加即时 done。

### 7.3 实现要求

当前 `BaseEnv.check_reset()` 通过单个 body list、固定 1 N 阈值和 contact history maximum 产生 reset。为了支持上述语义，后续可扩展为配置化条件：

```text
fall_contact_body_names
fall_force_threshold
head_contact_body_names
head_force_threshold
head_contact_dwell_steps
head_impulse_threshold
```

在第一版 parity config 中可以先只把 base 放入现有 termination list，避免为了平地实验提前扩大 BaseEnv 改动范围。

## 8. 训练结构

### 8.1 平地和初期 rough baseline

不修改 PPO 核心，继续使用 LeggedLab 的标准 asymmetric actor-critic：

- actor 读取普通感知历史；
- critic 读取当前 privileged observation；
- PPO 超参数先对齐 RobotLab：entropy 0.01、其余 clip/gamma/lambda/epochs/minibatches/network dims 保持参考值。

先证明环境定义能够训练出稳定策略，再讨论 teacher-student。否则 MoE、privileged latent 和 reward 差异会纠缠在一起。

### 8.2 完整 MoE-CTS（后续阶段）

如果目标升级为算法复现，不直接改写原 `ppo.py`。新增：

```text
ActorCriticMoECTS
MoECTS algorithm
OnPolicyRunnerCTS
CTS-compatible observation groups/storage
```

PPO 的 GAE、clipped surrogate 和 value loss仍可复用，但需增加：

- teacher/student env partition；
- teacher privileged encoder；
- student history MoE encoder；
- shared actor；
- student latent MSE；
- load-balance loss；
- student 独立 optimizer。

LeggedLab 现有 Distillation 是“固定 teacher 的 action imitation”，不是 concurrent latent distillation，不能直接视为 RobotLab MoE-CTS。

## 9. 多地形扩展顺序

保持 flat 已验证的 action、obs、reward 和 PPO 不变，只切换 terrain/curriculum：

```text
flat mesh
  → wave / gentle slope
  → random rough / rough slope
  → stairs
  → obstacles
  → gap / stepping stones
```

从平地阶段就保持未来 rough 所需的 critic schema；平地 height scan 可以是平坦值，这样 checkpoint 切换地形时不改变网络输入维度。

多地形阶段：

- actor 保持 10 帧 proprioception；
- height scan 只给 privileged critic/teacher；
- terrain level curriculum 单独启用；
- terrain-dependent command limits 单独启用；
- 不因 rough 更难而同时改一批 reward 权重，除非消融结果支持。

## 10. 比较方法论

### 10.1 配置 manifest 比较

训练开始时把以下内容保存为机器可读 JSON/YAML：

- git commit 和 dirty status；
- asset 路径/hash；
- joint/body order；
- physics/action/control 参数；
- obs term/order/scale/noise/history；
- reward function/params/weights；
- randomization mode/range/distribution；
- termination 条件；
- commands/terrain/curriculum；
- PPO/network config。

建立 RobotLab ↔ LeggedLab parity table，每项状态标记为：

```text
exact / semantically equivalent / intentionally different / unknown
```

禁止只标“已对齐”而不注明数学语义和证据。

### 10.2 三层比较

#### 层 1：静态和单步测试

- 检查 observation shape、顺序和有限值；
- 检查零动作对应的 target joint pose；
- 对可构造状态比较每个 reward term 数值；
- 检查 reset 后 12 个控制步的 history、previous action、二阶 action penalty；
- 检查 terminal 和 timeout 的 bootstrap 标志。

#### 层 2：固定 action rollout

使用 standing action、正弦 action 或已导出 policy，在固定 seed 和固定 command 下运行短 rollout，比较：

- base pose/velocity；
- joint q/dq/torque；
- contact force；
- 每项 raw/weighted reward；
- termination reason。

两个 simulator 的轨迹不会逐帧完全相同，比较重点是初值、方向、量级和统计分布。

#### 层 3：训练与评测

每个关键版本至少 3 个 seed。使用冻结的 evaluation config，不在评测时启用训练 curriculum。报告均值和离散度，而不是只选择最好 checkpoint。

### 10.3 One-factor-at-a-time 消融序列

建议实验编号：

| ID | 相对上一实验的唯一变化 | 目的 |
|---|---|---|
| E0 | nominal plane + 标准 PPO | 控制链路 smoke test |
| E1 | RobotLab reward | reward 行为基线 |
| E2 | actor 10 / critic 1 obs 对齐 | observation 影响 |
| E3 | reset curriculum | 起步恢复能力 |
| E4 | physics randomization | sim2sim robustness |
| E5 | delay + push | 动态扰动 robustness |
| E6 | flat mesh | 接触模型迁移 |
| E7 | terrain curriculum | 多地形基线 |
| E8 | MoE-CTS | 算法增益 |

如果怀疑两个因素存在交互，再补充小型 2×2 实验，例如：

```text
history init: zero vs repeat-first
command onset: step vs ramp
```

## 11. 固定评测场景与指标

固定 command cases：

```text
(0.0, 0.0, 0.0)
(0.3, 0.0, 0.0)
(0.8, 0.0, 0.0)
(0.0, 0.3, 0.0)
(0.0, 0.0, 0.5)
0 → 0.8 step
0 → 0.8 ramp over 0.5 s
```

至少报告：

- reset 后 0.5 s / 1.0 s fall rate；
- episode success rate 和 termination reason 分布；
- velocity tracking RMSE；
- max/percentile roll、pitch；
- minimum base height；
- action rate 和 action second difference；
- torque/power 峰值及均值；
- thigh/calf/head contact rate；
- Isaac Sim → MuJoCo 性能下降比例。

针对当前“初始加速快摔倒”，单独保存 reset 后前 50 个 policy step 的 command、history fill level、q、dq、target q、torque、base pitch、base height 和 action。

## 12. 阶段验收门槛

### Gate A：环境正确性

- observation/action shape、order、scale 有自动检查；
- reset/history 没有 NaN 和跨 env 污染；
- reward term 均为有限值；
- timeout 与 failure terminal 可区分；
- nominal 零动作能保持物理上合理的初始站姿。

### Gate B：平地训练

- 至少 3 seeds 能稳定收敛；
- 固定慢速命令下起步 1 秒 fall rate 接近 0；
- step 和 ramp 均通过，且差距有记录；
- 主要 reward term 的数量级合理，无单项意外支配。

### Gate C：sim2sim

- Isaac Sim 与 MuJoCo 的 observation preprocessing、joint order、default pose、PD、action scale、history init 完全核对；
- 在无随机化 nominal 条件下先通过，再评估 OOD；
- 不用 command ramp 掩盖明显的接口错误。

### Gate D：多地形

- terrain curriculum 可控；
- actor 输入维度不因 terrain 改变；
- 各 terrain type 单独报告 success，而不是只报告混合平均值。

## 13. 给执行对话的工作清单

1. 阅读本文件和列出的 RobotLab/LeggedLab 源文件。
2. 检查 LeggedLab worktree，保留用户已有修改。
3. 在 `mdp/rewards.py` 中补齐 RobotLab 语义的 reward，并添加最小测试或 smoke check。
4. 新增 `Go2RobotLabRewardCfg`、`Go2RobotLabFlatEnvCfg`、`Go2RobotLabFlatAgentCfg`，不要修改现有 `Go2FlatEnvCfg` 的行为。
5. 注册 `Go2-Flat-RobotLab-Parity-v0`。
6. 第一版 termination 仅使用 base fall contact；head 只记录，不即时 terminal。
7. 第一版关闭强随机化、delay 和 push，保留后续 curriculum 的明确入口。
8. 设置 actor history=10、critic history=1，并验证 reset history 初始化。
9. 对齐 PPO 超参数，但不修改 PPO 核心。
10. 运行 import/config construction/reward shape smoke test。
11. 生成 parity manifest 和差异表。
12. 在实现报告中列出所有 intentionally-different 项目，不宣称无法证明的 exact parity。

## 14. 当前明确不做的工作

- 不覆盖或删除现有 Go2 Flat/Rough/WM 配置；
- 不在第一版移植 MoE-CTS；
- 不把 WM critic filter 与 PPO value critic 混为同一组件；
- 不在第一版同时加入全地形、全部 randomization 和完整 command curriculum；
- 不通过单次最好 seed 判断对齐成功；
- 不把 head 碰撞即时 termination 当作主动避障能力。

