# Reproduction commands

Run from the repository root after installation and dependency/data preparation. Full training requires CUDA. These commands create new results; they do not download trained checkpoints.

## ODSR: one request

```bash
python scripts/train_original.py --backbone ODSR --dataset epinions --seed 42 --device cuda:0
python scripts/run_edge_user_odsr.py --dataset epinions --seed 42 --fraction .01 --method retrain --device cuda:0
python scripts/run_odsr_native_topology.py --dataset epinions --seed 42 --fraction .01 --variant full --device cuda:0
```

Then run `graph_only`, `retain_finetune`, `scif`, `gnndelete`, or `da2` using `run_edge_user_odsr.py --method METHOD`. Always produce the matching retrain checkpoint first. Use the exact same fraction and seed. To reproduce all three seeds and both datasets, use `scripts/reproduce.py` as shown in the README.

`train_original.py` invokes the shared original training routine with no deletion requests. Directly executing the older `run_unlearning.py` also runs historical compact-row deletion experiments; use the new wrapper for the current study.

## EGCL: one request

```bash
python scripts/train_original.py --backbone EGCL --dataset epinions --seed 42 --device cuda:0
python scripts/run_egcl_edge_controls.py --dataset epinions --seed 42 --method retrain --device cuda:0
python scripts/run_egcl_unlearning_adapters.py --dataset epinions --seed 42 --method native_topology --device cuda:0
python scripts/run_egcl_unlearning_adapters.py --dataset epinions --seed 42 --method gnndelete_native --device cuda:0
```

EGCL edge controls are fixed to 1% deletion. The original cap is 2,000 epochs with validation selection and early stopping, rather than 2,000 guaranteed epochs. `train_original.py --epochs N` is only for a separate pilot checkout: a changed epoch budget intentionally invalidates existing provenance-compatible caches.

## Controlled membership inference

The optional `--bank` argument on the main ODSR method reports positive-only membership scores; it does not supply target nonmembers and must not be labeled target AUC. Use the independent controlled suite below for AUC.

1. Train a seed-42 original for the selected backbone/dataset.
2. Build a frozen shadow/attacker bank:

```bash
python scripts/run_benchmark_mio.py --backbone ODSR --dataset epinions --device cuda:0
```

3. Locate the bank produced under `checkpoints/benchmark_mio/ODSR/epinions/<source-hash>/`. It contains `manifest.json` and `attacker.pt`. Set the shell variable to this actual directory; hashes are deliberately not hard-coded in documentation:

```bash
BANK=checkpoints/benchmark_mio/ODSR/epinions/REPLACE_WITH_SOURCE_HASH
python scripts/run_controlled_target_mio.py --dataset epinions --seed 42 --bank "$BANK" \
  --methods original retrain graph_only retain_finetune scif gnndelete da2 --device cuda:0
python scripts/run_tuned_benchmark_mio.py --dataset epinions --bank "$BANK" --device cuda:0
```

4. The target directory is `checkpoints/controlled_target_mio/ODSR/epinions/seed_42/<experiment-hash>/`; the tuned bank is under `checkpoints/tuned_benchmark_mio/ODSR/epinions/<source-hash>/`. Use the directories produced by the preceding commands:

```bash
TARGET=checkpoints/controlled_target_mio/ODSR/epinions/seed_42/REPLACE_WITH_EXPERIMENT_HASH
TUNED_BANK=checkpoints/tuned_benchmark_mio/ODSR/epinions/REPLACE_WITH_SOURCE_HASH
python scripts/run_controlled_odsr_native_topology.py --dataset epinions \
  --target "$TARGET" --bank "$BANK" --tuned-bank "$TUNED_BANK" --device cuda:0
```

Repeat target generation/evaluation for seeds 43 and 44 while reusing the compatible frozen attack bank. Repeat the dataset-specific sequence for Yelp2018. The corrected controlled evaluator currently requires both attack-bank variants and records both for diagnostics. The study's primary choices are the shadow-validation-tuned attack for Epinions and the default attack for Yelp2018; do not choose the primary attack based on which target result is more favorable.

For EGCL, build a separate EGCL bank and pass it to its native controlled runner:

```bash
python scripts/run_benchmark_mio.py --backbone EGCL --dataset epinions --device cuda:0
EGCL_BANK=checkpoints/benchmark_mio/EGCL/epinions/REPLACE_WITH_SOURCE_HASH
python scripts/run_controlled_egcl_mio.py --dataset epinions --seed 42 --bank "$EGCL_BANK" --device cuda:0
```

These controlled target suites train separate targets with genuine never-history-member controls. Their utility values must not be silently pooled with main-comparison utility runs. Preserve manifests, attacker hashes, and membership IDs with each result.

## Reading outputs

Each completed run writes JSON metrics and provenance beside its checkpoint. Corrected ODSR results include native correction/reset metadata and direct forgetting comparisons. Control results include utility/fidelity and measured timing. `scripts/audit_corrected_forgetting.py` provides additional control diagnostics; its historical path registry is not a replacement for the corrected method's own report.

Aggregate only identical protocols, datasets, fractions, and method versions. Report sample mean/standard deviation across the declared seeds and expose missing/failed seeds. Do not interpret a queued job or a saved intermediate checkpoint as a completed result.
