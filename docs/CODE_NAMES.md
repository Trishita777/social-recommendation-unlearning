# Public code names

Use these names in commands, documentation, and paper method labels.

| Paper label | Public command | Backbones | Frozen implementation |
| --- | --- | --- | --- |
| ODSR | `train_odsr.py` | ODSR | Model: `external/ODSR/ODSR.py`; original training: `run_unlearning.py` |
| EGCL | `train_egcl.py` | EGCL | Model: `src/social_unlearning/egcl.py`; training: `run_egcl_backbone.py` |
| **SoRecRemover** | **`run_sorecremover.py`** | ODSR, EGCL | ODSR: `run_odsr_native_topology.py --variant full`; EGCL: `run_egcl_unlearning_adapters.py --method native_topology` |
| SCIF | `run_scif.py` | ODSR | `scif_edges.py` and the SCIF runner in `run_edge_user_odsr.py` |
| GNNDelete | `run_gnndelete.py` | ODSR, EGCL | `gnndelete_edges.py` / `egcl_unlearning_adapters.py`, using the upstream deletion operator |
| DA2Unlearn | `run_da2unlearn.py` | ODSR | `baselines.py` and `edge_deletion.py`, using the upstream difficulty estimator |
| Scratch retraining | `run_retraining.py` | ODSR, EGCL | Edge-control runners with `method=retrain` |
| Graph pruning | `run_graph_pruning.py` | ODSR, EGCL | Edge-control runners with `method=graph_only` |
| Retained fine-tuning | `run_retain_finetuning.py` | ODSR, EGCL | Edge-control runners with `method=retain_finetune` |

All public command files are under `scripts/`. Method implementation modules are under `src/social_unlearning/` unless another path is given.

Example for the current method:

```bash
python scripts/train_odsr.py --dataset epinions --seed 42 --device cuda:0
python scripts/run_retraining.py --backbone ODSR --dataset epinions --seed 42 --fraction .01 --device cuda:0
python scripts/run_sorecremover.py --backbone ODSR --dataset epinions --seed 42 --fraction .01 --device cuda:0
```

The batch runner uses these canonical method IDs:

```text
retraining  graph_pruning  retain_finetuning  scif  gnndelete  da2unlearn  sorecremover
```

Older batch arguments `retrain`, `graph_only`, `retain_finetune`, and `da2` remain accepted aliases. Result files and checkpoint namespaces retain their original internal IDs so existing provenance and comparisons remain valid. The new names do not change model parameters, losses, hyperparameters, or deletion semantics.

`ours`, reverse-loss variants, and the older guarded method are **historical methods**, not aliases for SoRecRemover. The public SoRecRemover command always dispatches to the frozen corrected implementation. ODSR ablations are explicitly selected with `--variant no_topology`, `uniform_weights`, or `no_reset`; the default is `full`.
