#!/usr/bin/env bash
# 文件功能：
#   调度 PatchTST frozen prediction + fixed visual embedding 双分支 65k 剩余轻量变体。
# 关键约束：
#   本脚本继承 robust multiseed 口径，只训练新增融合头和 residual_scale sweep；
#   不生成图像、不运行 ViT、不训练 PatchTST、不覆盖 1b20e72 历史输出。

set -u

OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/syh/Time/run_outputs/2026-06-23_patchtst_visual_dual_branch_65k_remaining_variants}"
VISUAL_CACHE="${VISUAL_CACHE:-/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/features/spatial_panel_3view}"
PATCHTST_CACHE="${PATCHTST_CACHE:-/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/inputs/patchtst_frozen_cache_from_round2_expanded.npz}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k_robust_multiseed/patchtst_visual/spatial_panel_3view}"
PY="${PY:-/home/shiyuhong/application/miniconda3/envs/quito/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

RESULT_ROOT="$OUTPUT_ROOT/patchtst_visual/spatial_panel_3view"
LOGDIR="$OUTPUT_ROOT/launcher_logs"
mkdir -p "$RESULT_ROOT" "$LOGDIR"

echo "start_time=$(date '+%F %T %Z')"
echo "output_root=$OUTPUT_ROOT"
echo "visual_cache=$VISUAL_CACHE"
echo "patchtst_cache=$PATCHTST_CACHE"
echo "historical_root=$HISTORICAL_ROOT"
echo "max_parallel=$MAX_PARALLEL"
printf 'pid\tgpu\tmode\tseed\tresidual_scale\trun_dir\n' > "$LOGDIR/pids.tsv"

active_pids=()
active_labels=()
failures=0
idx=0

wait_oldest() {
  local wait_pid="${active_pids[0]}"
  local wait_label="${active_labels[0]}"
  wait "$wait_pid"
  local rc=$?
  echo "finish label=$wait_label pid=$wait_pid rc=$rc time=$(date '+%F %T %Z')"
  if [ "$rc" -ne 0 ]; then
    failures=$((failures + 1))
  fi
  active_pids=("${active_pids[@]:1}")
  active_labels=("${active_labels[@]:1}")
}

launch_run() {
  local mode="$1"
  local seed="$2"
  local residual_scale="$3"
  local run_name="$4"
  local gpu=$((idx % MAX_PARALLEL))
  local run_dir="$RESULT_ROOT/$run_name"
  local run_log="$LOGDIR/${run_name}.stdout.log"
  echo "launch mode=$mode seed=$seed residual_scale=$residual_scale gpu=$gpu run_dir=$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m visual_router_experiments.dual_branch_fusion.train_patchtst_visual_65k \
    --data_subset 65k \
    --ts_model patchtst \
    --visual_embedding_cache "$VISUAL_CACHE" \
    --patchtst_cache "$PATCHTST_CACHE" \
    --fusion_mode "$mode" \
    --train_split round2_train_expanded \
    --val_split round2_selection_expanded \
    --test_split round2_test_expanded \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --residual_scale "$residual_scale" \
    --seed "$seed" \
    --device cuda \
    --output_dir "$run_dir" \
    --overwrite > "$run_log" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$mode" "$seed" "$residual_scale" "$run_dir" >> "$LOGDIR/pids.tsv"
  active_pids+=("$pid")
  active_labels+=("$run_name")
  idx=$((idx + 1))
  if [ "${#active_pids[@]}" -ge "$MAX_PARALLEL" ]; then
    wait_oldest
  fi
}

for mode in pred_gate feature_gate gated_residual_feature; do
  for seed in 1 2 3; do
    launch_run "$mode" "$seed" "0.1" "${mode}_seed${seed}"
  done
done

for scale in 0.01 0.03 0.05 0.1; do
  scale_tag="${scale/./p}"
  for seed in 1 2 3; do
    launch_run "patchtst_residual_visual" "$seed" "$scale" "patchtst_residual_visual_scale${scale_tag}_seed${seed}"
  done
done

while [ "${#active_pids[@]}" -gt 0 ]; do
  wait_oldest
done

echo "training_failures=$failures"
if [ "$failures" -eq 0 ]; then
  "$PY" -m visual_router_experiments.dual_branch_fusion.summarize_full_comparison \
    --remaining_root "$RESULT_ROOT" \
    --historical_root "$HISTORICAL_ROOT" \
    --output_dir "$RESULT_ROOT/summary"
  summary_rc=$?
else
  summary_rc=99
fi
echo "summary_rc=$summary_rc"
echo "end_time=$(date '+%F %T %Z')"
exit $((failures + summary_rc))
