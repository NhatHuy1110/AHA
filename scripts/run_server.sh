#!/usr/bin/env bash
# Usage: bash scripts/run_server.sh features | abstract | ablation | vlm
#   features : frozen caches (label-free encoder for the model, SurgFormer for comparators)
#   abstract : proposed model + matched trained baselines, 3 seeds, and deterministic baselines on val
#   ablation : remaining ablations; run after the abstract matrix
#   vlm      : multimodal-LLM comparators: zero-shot (two input protocols) and LoRA on the same train QA
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
stage="${1:?features|abstract|ablation|vlm}"
ENCODER="${ENCODER:-dinov2_vitl14}"

train_matrix() {
  for variant in "$@"; do
    for seed in 17 29 43; do
      run="runs/${variant}/seed_${seed}"
      if [ -f "$run/last.pt" ]; then resume=--resume; else resume=; fi
      python -m aha train --config "configs/${variant}.json" --out "$run" --seed "$seed" --device cuda $resume
    done
  done
}

case "$stage" in
  features)
    python -m aha verify
    python -m aha text --device cpu
    python -m pytest -q
    python -m aha audit
    test -f "assets/encoders/${ENCODER}/source.json" || { echo "Stage the encoder first: scripts/fetch_encoder.py"; exit 1; }
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python -m aha encode --name "$ENCODER" --device cuda
    python -m aha extract --device cuda
    ;;
  abstract)
    train_matrix full independent structured_only residual_only
    # Intentionally validation only: freeze the run matrix before scripts/evaluate_test.sh.
    for qa in original derived; do
      for method in surgformer_index prior; do
        python -m aha baseline --method "$method" --split val --qa "$qa" --out "runs/${method}_val_${qa}.json"
        python -m aha evaluate --split val --qa "$qa" --predictions "runs/${method}_val_${qa}.json" --out "runs/${method}_val_${qa}"
      done
    done
    ;;
  ablation)
    train_matrix no_workflow no_derived no_hard_negative seen_backbone
    ;;
  vlm)
    # Comparators are evaluated on the retrospective test with the same evaluator and gold.
    # Zero-shot decoding is greedy, so one run per (model, input protocol) is deterministic.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    vlm_eval() {  # name model frames budget [adapter]
      for qa in original derived; do
        out="runs/vlm/$1/test_${qa}"
        [ -f "${out}_metrics/metrics.json" ] && continue
        python -m aha vlm-predict --model "$2" --frames "$3" --budget "$4" --split test --qa "$qa" --out "${out}.json" ${5:+--adapter "$5"}
        python -m aha evaluate --split test --qa "$qa" --predictions "${out}.json" --out "${out}_metrics" \
          --paired "runs/full/seed_17/test_${qa}.json"
      done
    }
    for model in ${VLM_MODELS:-qwen3vl_8b hulumed_7b qwen3vl_4b}; do
      vlm_eval "${model}_uniform64" "$model" uniform 64
      vlm_eval "${model}_retrieval48" "$model" retrieval 48
    done
    for seed in ${VLM_SEEDS:-17 29 43}; do
      run="runs/vlm/qwen3vl_4b_lora/seed_${seed}"
      [ -f "$run/history.json" ] || python -m aha vlm-train --model qwen3vl_4b --frames retrieval --budget "${VLM_LORA_BUDGET:-48}" --out "$run" --seed "$seed"
      vlm_eval "qwen3vl_4b_lora/seed_${seed}" qwen3vl_4b retrieval "${VLM_LORA_BUDGET:-48}" "$run/best"
    done
    python scripts/summarize_seeds.py
    ;;
  *) echo "unknown stage $stage"; exit 1;;
esac
