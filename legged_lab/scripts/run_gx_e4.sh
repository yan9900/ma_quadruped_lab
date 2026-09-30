#!/usr/bin/env bash
# 按 E4（go2_fricsweep_2700，LeggedLab/legged_lab/docs/FRICTION_DATASET_COMPOSITION.md §9）方案重采，2026-09-30 用户定。
# 与 E4 逐项相同：刹车块全部注入随机 μ（B1 0.22–0.32 / B2 0.32–0.55 / B3 0.55–1.00）、μ 无关的起刹尺
# d̂_ref = 1.15·v + 0.5 × r∈[0.15, 1.15]、摩擦在起刹前 0.6–1.2 s 注入、spawn solve + extra runway 0–1 m +
# 朝向 ±45°、动摩擦比 0.55–0.90、B0 弱刹车 400、A1 低 μ 行走 300、E 配对考卷 3 档 μ。
# 与 E4 不同（均为用户指定或为换 policy 所必需）：
#   policy   RobotLab v1 + GO2HV + ±1 kg（与 gx_main_v1 一致）；指令上限 2.0 m/s（v1 能力），E4 为 2.5
#   spawn_accel 1.1（v1 平地实测 p10），E4 的 2.6 是 LeggedLab 自训 policy 的
#   相机     实体 D435 挂载（--camera_mount d435，与 E4 数据相同）
#   地形     gx_e4：E4 的 CLIFF_DETECTION 去掉 stairs，换成 box_low 0.15–0.40（12 m 格、平台 8 m）
#   --store_geo：逐帧位姿/足端/接触 + 地形 mesh，供静态高程图 g 离线打标签
#
#   TRIAL=1 BLOCK=B1 bash run_gx_e4.sh      # 试运行：GUI、8 env × 1 条
#   BLOCK=B1 bash run_gx_e4.sh              # 正式：headless
#   BLOCK=E  bash run_gx_e4.sh              # 考卷：3 档 μ 同 seed（不进训练）
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/lcy/miniconda3/envs/env_isaaclab/bin/python}"
"$PY" -c "import isaaclab" || { echo "FATAL: $PY 没有 isaaclab"; exit 1; }
BLOCK="${BLOCK:?BLOCK=B1|B2|B3|B0|B4|A1|E}"
TRIAL="${TRIAL:-0}"
TERRAIN_SEED="${TERRAIN_SEED:-20261100}"          # 全部块同一张地形（与 E4 相同做法）
POLICY="${POLICY:-logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt}"

BASE=(--task go2_data_collection_robotlab --enable_cameras
      --robotlab_policy "$POLICY" --actuator go2hv --base_mass_range -1 1 --store_geo --camera_mount d435
      --depth_uint8 --depth_clip_min 0.30 --depth_clip_max 3.00
      --terrain_profile gx_e4 --platform_half_width 4.0 --terrain_seed "$TERRAIN_SEED")
BRAKE=(--collection_mode braking --fault_type friction --safe_ratio 0.0
       --brake_trigger edge --fault_trigger window --fault_window_range 0.6 1.2
       --spawn_mode solve --spawn_accel 1.1 --spawn_settle_s 0.25
       --spawn_extra_runway 0.0 1.0 --spawn_lateral_margin 0.2 --spawn_heading_range -0.785398 0.785398
       --brake_ref_mode mu_blind --brake_ref_linear 1.15 --brake_ref_quadratic 0.0 --brake_ref_offset 0.50
       --brake_margin_ratio_range 0.15 1.15
       --stop_speed_thresh 0.20 --stop_sustain_steps 10
       --command_vx_range 0.5 2.0 --command_vy_range -0.15 0.15 --command_yaw_rate_range 0.0 0.0
       --ood_dynamic_ratio_range 0.55 0.90
       --max_steps 500 --dynamic_max_steps --post_brake_budget_s 4.0)

case "$BLOCK" in
  B1) SEED=20261101; ARGS=("${BRAKE[@]}" --brake_complete_ratio 1.0 --ood_static_friction_range 0.22 0.32); EPS_FULL=16 ;;
  B2) SEED=20261102; ARGS=("${BRAKE[@]}" --brake_complete_ratio 1.0 --ood_static_friction_range 0.32 0.55); EPS_FULL=10 ;;
  B3) SEED=20261103; ARGS=("${BRAKE[@]}" --brake_complete_ratio 1.0 --ood_static_friction_range 0.55 1.00); EPS_FULL=4 ;;   # 原 14（700），18:00 减量
  # B4：E4 当年的高 μ 边界补采（fric_B4_highmu_muaware，480 条）——μ 无关的尺子让高 μ 几乎不失败，
  # 改用按 μ 调节的起刹（v1 平地拟合 d_brake + d_hold 0.11），r∈[0.4, 1.6]，使高 μ 也有刹不住的边界样本。
  B4) SEED=20261106; ARGS=("${BRAKE[@]}" --brake_complete_ratio 1.0 --ood_static_friction_range 0.55 1.00
                            --brake_ref_mode mu_aware --brake_ref_dyn_a 0.0394 --brake_ref_dyn_b 0.0478 --brake_ref_dyn_c -0.0623
                            --brake_ref_hold 0.11 --brake_margin_ratio_range 0.4 1.6); EPS_FULL=3 ;;   # 原 10（500）
  B0) SEED=20261104; ARGS=("${BRAKE[@]}" --brake_complete_ratio 0.0 --brake_target_ratio_range 0.0 1.0
                            --ood_static_friction_range 0.22 1.00); EPS_FULL=3 ;;   # 原 8（400）
  A1) SEED=20261105; ARGS=(--collection_mode normal --fault_type friction --lowfric_walk --safe_ratio 0.0
                            --t_fault_min 30 --t_fault_max 90 --max_steps 300
                            --command_vx_range 0.5 2.0 --command_vy_range -0.3 0.3 --command_yaw_rate_range -0.3 0.3
                            --ood_static_friction_range 0.22 0.32 --ood_dynamic_ratio_range 0.55 0.90
                            --min_fail_consecutive 5); EPS_FULL=2 ;;   # 原 6（300）
  E)  SEED=20261107 ;;
  *) echo "BLOCK 必须是 B1|B2|B3|B0|B4|A1|E"; exit 1 ;;
esac

run_one() {  # $1 = 输出目录，$2 = 名字，$3 = env 数，$4 = 每 env 条数，其余 = 额外参数
  local OUT="$1" NAME="$2" ENVS="$3" EPS="$4"; shift 4
  local PKL="${OUT}/${NAME}.pkl"
  if [ -s "$PKL" ]; then echo "已存在 $PKL，跳过"; return 0; fi
  mkdir -p "$OUT"
  "$PY" legged_lab/scripts/collect_go2_data_v3.py "${BASE[@]}" "$@" "${VIS[@]}" \
    --num_envs "$ENVS" --num_episodes "$EPS" --seed "$SEED" \
    --output_dir "$OUT" --output_name "$NAME" > "${OUT}/collect.log" 2>&1 &
  local CPID=$!
  for _ in $(seq 1 2160); do            # 最多 3 小时
    [ -s "$PKL" ] && break
    kill -0 "$CPID" 2>/dev/null || break
    sleep 5
  done
  if [ -s "$PKL" ]; then
    sleep 15
    kill -0 "$CPID" 2>/dev/null && { echo "[teardown] ${NAME} pkl 已落地，强制收工"; kill -9 "$CPID"; }
  fi
  wait "$CPID" 2>/dev/null || true
  [ -s "$PKL" ] || { echo "FATAL: ${NAME} 没有产出 pkl，看 ${OUT}/collect.log"; exit 1; }
  echo "done: $PKL"
}

if [ "$TRIAL" = "1" ]; then
  ROOT="${ROOT:-./data/gx_e4_trial}"; VIS=(); [ "${HEADLESS:-0}" = "1" ] && VIS=(--headless); ENVS=8; EPS="${TRIAL_EPS:-1}"
else
  ROOT="${ROOT:-./data/gx_e4_v1}"; VIS=(--headless); ENVS=50       # 10×10=100 格
fi

if [ "$BLOCK" = "E" ] && [ "${E_ENABLE:-0}" != "1" ]; then
  echo "E 考卷暂缓（2026-09-30 为赶时间跳过；补采用 E_ENABLE=1 BLOCK=E）"; exit 0
fi
if [ "$BLOCK" = "E" ]; then
  # 配对考卷：三档 μ 同 seed、同 env 数 ⇒ 同一 env 的第 k 条 episode 出生/指令/r 完全相同，只差摩擦
  [ "$TRIAL" = "1" ] && E_ENVS=8 || E_ENVS=20
  [ "$TRIAL" = "1" ] && E_EPS=1 || E_EPS=3
  for MU in 0.24 0.40 0.80; do
    run_one "${ROOT}/E_mu${MU}" "gx_e4_E_mu${MU}" "$E_ENVS" "$E_EPS" "${BRAKE[@]}" \
      --brake_complete_ratio 1.0 --ood_static_friction_range "$MU" "$MU" --ood_dynamic_ratio_range 0.70 0.70
  done
else
  [ "$TRIAL" = "1" ] || EPS=$EPS_FULL
  run_one "${ROOT}/${BLOCK}" "gx_e4_${BLOCK}" "$ENVS" "$EPS" "${ARGS[@]}"
fi
