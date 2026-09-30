#!/usr/bin/env bash
# 静态高程图 g_x 正式采集（latent-safety/docs/82、docs/83）。四个子集：
#   A 跨越      ：normal、不刹车、高 μ；小平台（GX_MAIN_SMALL）          —— 失败/成功跨越，细化 p_fail
#   B 高 μ 刹车 ：braking、edge 起刹、spawn solve；大平台（GX_MAIN_BIG）  —— V_stop 标签
#   C 低摩擦刹车：B + 在起刹前窗口内注入低摩擦                             —— μ 对刹车距离的影响
#   D 低摩擦行走：normal + 低摩擦、不刹车（lowfric_walk）；小平台          —— latent 的 μ 表示
# 共同：RobotLab v1 + GO2HV、载荷 ±1 kg、multiply 摩擦、失败 = base 或头部接触 >1 N（任务默认终止列表）、
#       --store_geo（逐帧位姿/足端/接触 + 地形 mesh + run_config.json）、vx 0.5–2.0。
# 起刹调度用 v1 平地拟合（latent-safety/results/d_brake_fit_v1.json）：
#   d_brake = 0.0394·v²/μ + 0.0478·v/μ − 0.0623·v，d_hold = 0.11，spawn_accel = 1.1（p10，偏保守）。
#
# 用法：
#   TRIAL=1 SUBSET=B bash run_gx_main.sh     # 试运行：GUI、8 env × 1 条
#   SUBSET=A bash run_gx_main.sh             # 正式：headless
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/lcy/miniconda3/envs/env_isaaclab/bin/python}"
"$PY" -c "import isaaclab" || { echo "FATAL: $PY 没有 isaaclab"; exit 1; }
SUBSET="${SUBSET:?SUBSET=A|B|C|D}"
TRIAL="${TRIAL:-0}"
SEED_BASE="${SEED_BASE:-20261002}"
MU_LO="${MU_LO:-0.22}"; MU_HI="${MU_HI:-1.00}"          # C/D 的静摩擦范围，同 E4 fricsweep（动摩擦 = 0.7×）
BRAKE_MODE="${BRAKE_MODE:-mu_aware}"

POLICY=logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt
COMMON=(--task go2_data_collection_robotlab --enable_cameras
        --robotlab_policy "$POLICY" --actuator go2hv --base_mass_range -1 1 --store_geo
        --depth_uint8 --depth_clip_min 0.30 --depth_clip_max 3.00
        --command_vx_range 0.5 2.0)
BRAKE=(--collection_mode braking --brake_trigger edge --spawn_mode solve --platform_half_width 5.0
       --brake_complete_ratio 1.0 --brake_margin_ratio_range 0.4 1.6
       --brake_ref_mode "$BRAKE_MODE" --brake_ref_dyn_a 0.0394 --brake_ref_dyn_b 0.0478 --brake_ref_dyn_c -0.0623
       --brake_ref_hold 0.11 --spawn_accel 1.1
       --command_vy_range 0.0 0.0 --command_yaw_rate_range 0.0 0.0
       --dynamic_max_steps --post_brake_budget_s 3.0 --max_steps 500
       --terrain_profile gx_main_big)
LOWMU=(--ood_static_friction_range "$MU_LO" "$MU_HI" --ood_dynamic_ratio_range 0.7 0.7)

# 各子集用不同 seed（否则 A 与 D 的地形/出生/指令完全相同，试运行已出现）
case "$SUBSET" in A) OFF=0 ;; B) OFF=1 ;; C) OFF=2 ;; D) OFF=3 ;; *) OFF=0 ;; esac
SEED="${SEED:-$((SEED_BASE + OFF))}"

case "$SUBSET" in
  A) ARGS=(--terrain_profile gx_main_small --platform_half_width 1.5 --spawn_mode range
           --collection_mode normal --fault_type none --safe_ratio 1.0 --max_steps 250); EPS_FULL=10 ;;
  B) ARGS=("${BRAKE[@]}" --fault_type none --safe_ratio 1.0); EPS_FULL=15 ;;
  C) ARGS=("${BRAKE[@]}" --fault_type friction --fault_trigger window --fault_window_s 0.6
           --safe_ratio 0.0 "${LOWMU[@]}"); EPS_FULL=15 ;;
  D) ARGS=(--terrain_profile gx_main_small --platform_half_width 1.5 --spawn_mode range
           --collection_mode normal --fault_type friction --lowfric_walk --t_fault_min 30 --t_fault_max 60
           --safe_ratio 0.0 --max_steps 250 "${LOWMU[@]}"); EPS_FULL=5 ;;
  *) echo "SUBSET 必须是 A|B|C|D"; exit 1 ;;
esac

if [ "$TRIAL" = "1" ]; then
  OUT="./data/gx_main_trial/${SUBSET}"; ENVS=8; EPS="${TRIAL_EPS:-1}"; VIS=()
else
  OUT="./data/gx_main_v1/${SUBSET}"; ENVS=60; EPS=$EPS_FULL; VIS=(--headless)   # 8x8=64 格，num_envs 须 < 格数
fi
NAME="gx_main_${SUBSET}"
PKL="${OUT}/${NAME}.pkl"
if [ -s "$PKL" ]; then echo "已存在 $PKL，跳过"; exit 0; fi
mkdir -p "$OUT"

"$PY" legged_lab/scripts/collect_go2_data_v3.py "${COMMON[@]}" "${ARGS[@]}" "${VIS[@]}" \
  --num_envs "$ENVS" --num_episodes "$EPS" --seed "$SEED" --terrain_seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" > "${OUT}/collect.log" 2>&1 &
CPID=$!
for _ in $(seq 1 1440); do            # 最多 2 小时
  [ -s "$PKL" ] && break
  kill -0 "$CPID" 2>/dev/null || break
  sleep 5
done
if [ -s "$PKL" ]; then
  sleep 15
  kill -0 "$CPID" 2>/dev/null && { echo "[teardown] ${SUBSET} pkl 已落地，强制收工"; kill -9 "$CPID"; }
fi
wait "$CPID" 2>/dev/null || true
[ -s "$PKL" ] || { echo "FATAL: ${SUBSET} 没有产出 pkl，看 ${OUT}/collect.log"; exit 1; }
echo "done: $PKL"
