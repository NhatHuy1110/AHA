#!/usr/bin/env bash
# Usage: bash scripts/run_server.sh features | abstract | ablation
#   features : frozen caches (label-free encoder for the model, SurgFormer for comparators)
#   abstract : proposed model + matched trained baselines, 3 seeds, and deterministic baselines on val
#   ablation : remaining ablations; run after the abstract matrix
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
stage="${1:?features|abstract|ablation}"
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
    python -m pytest -q
    python -m aha audit
    test -f "assets/encoders/${ENCODER}/source.json" || { echo "Stage the encoder first: scripts/fetch_encoder.py"; exit 1; }
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python -m aha text --device cpu
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
  *) echo "unknown stage $stage"; exit 1;;
esac
