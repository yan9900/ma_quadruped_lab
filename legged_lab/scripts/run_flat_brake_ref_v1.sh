#!/usr/bin/env bash
# RobotLab v1 的平地制动参考：拟合 d_brake(v, μ_dyn)、d_hold、起步加速度（spawn solve 用）。
# 旧尺子（docs/68，0.0388·v²/μ + 0.0848·v/μ − 0.0109·v）是用 LeggedLab 自训 policy 拟合的，v1 不能套用。
#
# 每档 μ 一次运行，速度在 0.5–2.0 m/s 连续随机（拟合用实测起刹速度）；
# 先在 t_fault 注入目标摩擦，再在 t_fault+delay 按帧起刹；刹停后多驻留 2 s 量漂移。
# 失败 = base 或头部接触 >1 N（任务默认终止列表），与 gx 主线口径一致。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/lcy/miniconda3/envs/env_isaaclab/bin/python}"
"$PY" -c "import isaaclab" || { echo "FATAL: $PY 没有 isaaclab"; exit 1; }

OUT_ROOT="${OUT_ROOT:-./data/flat_brake_ref_v1}"
SEED=20261001
ENVS=24      # FLAT_MESH 是 5x5=25 格
EPS=5
CAP=360      # 单档最多等 30 分钟

for MU in 0.12 0.24 0.40 0.80; do
  TAG="mu${MU}"
  PKL="${OUT_ROOT}/${TAG}/${TAG}.pkl"
  if [ -s "$PKL" ]; then echo "=== ${TAG}（已完成，跳过）==="; continue; fi
  echo "=== ${TAG} ==="
  mkdir -p "${OUT_ROOT}/${TAG}"
  "$PY" legged_lab/scripts/collect_go2_data_v3.py \
    --task go2_data_collection_robotlab --headless \
    --robotlab_policy logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt \
    --actuator go2hv --base_mass_range -1 1 --store_geo \
    --num_envs ${ENVS} --num_episodes ${EPS} --max_steps 500 \
    --seed ${SEED} --terrain_seed ${SEED} \
    --terrain_profile flat_mesh --spawn_mode range \
    --collection_mode braking --fault_type friction --safe_ratio 0.0 \
    --brake_complete_ratio 1.0 \
    --fault_trigger frame --t_fault_min 30 --t_fault_max 60 \
    --brake_trigger frame --post_fault_brake_delay_range 40 80 \
    --ood_static_friction_range ${MU} ${MU} --ood_dynamic_ratio_range 0.7 0.7 \
    --command_vx_range 0.5 2.0 --command_vy_range 0.0 0.0 --command_yaw_rate_range 0.0 0.0 \
    --stop_speed_thresh 0.20 --stop_sustain_steps 10 \
    --early_stop_hold_steps 100 \
    --output_dir "${OUT_ROOT}/${TAG}" --output_name "${TAG}" > "${OUT_ROOT}/${TAG}/collect.log" 2>&1 &
  CPID=$!
  for _ in $(seq 1 ${CAP}); do
    [ -s "$PKL" ] && break
    kill -0 "$CPID" 2>/dev/null || break
    sleep 5
  done
  if [ -s "$PKL" ]; then
    sleep 15
    kill -0 "$CPID" 2>/dev/null && { echo "[teardown] ${TAG} pkl 已落地，强制收工"; kill -9 "$CPID"; }
  fi
  wait "$CPID" 2>/dev/null || true
  [ -s "$PKL" ] || { echo "FATAL: ${TAG} 没有产出 pkl，看 ${OUT_ROOT}/${TAG}/collect.log"; exit 1; }
done
echo "done: ${OUT_ROOT}"
