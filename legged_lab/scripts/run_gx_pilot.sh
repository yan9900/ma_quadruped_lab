#!/usr/bin/env bash
# 静态高程图 g_x 的 pilot 采集（docs/82 §4、§7；latent-safety/docs/82_静态高程图gx_设计定稿.md）。
#
# 目的：拟合下行/上行通过性曲线 p_fail(Δh)，并验证离线 g 标签（d_fall / d_col）。
#   - 策略：RobotLab v1 盲走 student，GO2HV 执行器（与其训练一致），载荷 ±1 kg
#   - 地形：GX_PILOT_TERRAINS_CFG（drop_box 0.05–0.60 m、step_pit 0.05–0.45 m、上/下坡、平地；不含楼梯）
#   - 模式：normal、不刹车（每条 episode 是一次跨越尝试），vx 0.5–2.0 m/s，高 μ（DR 0.6–1.0，multiply）
#   - 终止：只按 base 接触（--terminate_base_only）；头部接触逐帧存下，失败在离线定义
#     （可选"头碰即失败"或 docs/81 的持续接触），以便以后补楼梯数据时口径一致
#   - 存储：--store_geo（逐帧位姿/足端/接触快照 + 地形 mesh + run_config.json）
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/lcy/miniconda3/envs/env_isaaclab/bin/python}"
"$PY" -c "import isaaclab" || { echo "FATAL: $PY 没有 isaaclab"; exit 1; }

OUT="${OUT:-./data/gxstatic_pilot_v1}"
NAME="${NAME:-gxstatic_pilot_v1}"
SEED="${SEED:-20260930}"
ENVS="${ENVS:-32}"   # GX_PILOT 是 6x6=36 格；采集器要求 num_envs < 格数
EPS="${EPS:-10}"
PKL="${OUT}/${NAME}.pkl"
if [ -s "$PKL" ]; then echo "已存在 $PKL，跳过"; exit 0; fi
mkdir -p "$OUT"

"$PY" legged_lab/scripts/collect_go2_data_v3.py \
  --task go2_data_collection_robotlab --headless --enable_cameras \
  --robotlab_policy logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt \
  --actuator go2hv --base_mass_range -1 1 --terminate_base_only --store_geo \
  --num_envs "${ENVS}" --num_episodes "${EPS}" --max_steps 250 \
  --seed "${SEED}" --terrain_seed "${SEED}" \
  --terrain_profile gx_pilot --platform_half_width 1.5 --spawn_mode range \
  --collection_mode normal --fault_type none --safe_ratio 1.0 \
  --command_vx_range 0.5 2.0 \
  --depth_uint8 --depth_clip_min 0.30 --depth_clip_max 3.00 \
  --output_dir "${OUT}" --output_name "${NAME}" > "${OUT}/collect.log" 2>&1 &
CPID=$!

# Isaac 的 simulation_app 关闭经常挂死：pkl 落地 + 宽限期后直接 KILL（同 run_flat_brake_ref.sh）
for _ in $(seq 1 720); do
  [ -s "$PKL" ] && break
  kill -0 "$CPID" 2>/dev/null || break
  sleep 5
done
if [ -s "$PKL" ]; then
  sleep 15
  kill -0 "$CPID" 2>/dev/null && { echo "[teardown] pkl 已落地，强制收工"; kill -9 "$CPID"; }
fi
wait "$CPID" 2>/dev/null || true
[ -s "$PKL" ] || { echo "FATAL: 没有产出 pkl，看 ${OUT}/collect.log"; exit 1; }
echo "done: $PKL"
