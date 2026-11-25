#!/usr/bin/env bash
# Run stage-wise GraphMoE controls after the matched comparison releases this GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=$1
backbone=$2
dataset=$3
out="results/partition_social/graph_moe_ablation30_shards10/$dataset"
lock="${GRAPH_MOE_GPU_LOCK:-results/runtime_acceleration/gpu${gpu}.lock}"
mkdir -p "$out" "$(dirname "$lock")"
export CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec 8>"$out/queue.lock"
flock -n 8 || exit 0
exec 9>"$lock"
flock -x 9

run_case() {
  local seed=$1 label=$2 shards=$3
  shift 3
  local result="$out/seed${seed}_${label}.json"
  if [[ -f "$result" ]] && python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("completion") != "complete")' "$result"; then
    return
  fi
  python -u scripts/run_partition_social.py --backbone "$backbone" --dataset "$dataset" \
    --seed "$seed" --device cuda:0 --epochs 30 --validation-every 2 --shards "$shards" \
    --methods sorecremover_graph_moe --output "$result" "$@" >"${result%.json}.log" 2>&1
}
for seed in 42 43 44; do
  run_case "$seed" one_hop_partition 10 --ablation one_hop_partition
  run_case "$seed" user_balance_only 10 --ablation user_balance_only
done
for shards in 20 50; do
  run_case 42 "shards${shards}" "$shards"
done
