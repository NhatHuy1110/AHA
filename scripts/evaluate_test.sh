#!/usr/bin/env bash
# Retrospective test. Run once, after the validation-selected run matrix is written to the study log.
# Evaluates every variant that has a run directory; both QA sets; paired against the SurgFormer index.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
for qa in original derived; do
  for method in surgformer_index prior; do
    python -m aha baseline --method "$method" --split test --qa "$qa" --out "runs/${method}_test_${qa}.json"
    python -m aha evaluate --split test --qa "$qa" --predictions "runs/${method}_test_${qa}.json" --out "runs/${method}_test_${qa}"
  done
done
for dir in runs/*/; do
  variant="$(basename "$dir")"
  [ -f "configs/${variant}.json" ] || continue
  for seed in 17 29 43; do
    run="runs/${variant}/seed_${seed}"
    [ -f "$run/best.pt" ] || { echo "missing $run/best.pt"; exit 1; }
    for qa in original derived; do
      python -m aha predict --checkpoint "$run/best.pt" --split test --qa "$qa" --out "$run/test_${qa}.json"
      python -m aha evaluate --split test --qa "$qa" --predictions "$run/test_${qa}.json" \
        --paired "runs/surgformer_index_test_${qa}.json" --out "$run/test_${qa}_metrics"
    done
    python scripts/compare_historical.py --rows "$run/test_original_metrics/per_question.json" --out "$run/test_original_metrics/versus_historical_pcjd.json"
    if [ "$variant" = full ]; then
      # Same weights, decoupled decoding: localize by onset evidence, then answer at that second.
      for qa in original derived; do
        python -m aha predict --checkpoint "$run/best.pt" --split test --qa "$qa" --decode cascade --out "$run/test_${qa}_cascade.json"
        python -m aha evaluate --split test --qa "$qa" --predictions "$run/test_${qa}_cascade.json" \
          --paired "$run/test_${qa}.json" --out "$run/test_${qa}_cascade_metrics"
      done
      for diagnostic in zero_visual shuffle_visual; do
        python -m aha predict --checkpoint "$run/best.pt" --split test --diagnostic "$diagnostic" --out "$run/test_original_${diagnostic}.json"
        python -m aha evaluate --split test --predictions "$run/test_original_${diagnostic}.json" --out "$run/test_original_${diagnostic}_metrics"
      done
    fi
  done
done
python scripts/summarize_seeds.py
