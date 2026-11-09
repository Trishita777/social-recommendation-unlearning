# Baselines and backbones

All current comparisons retain the full user/item universe, remove requested users' incident interaction/social edges, rebuild normalization, and evaluate retained utility against the matching scratch reference. Older compact-row functions remain only where imported; the reproduction runner selects the current fixed-ID entry points.

| Component | Entry point / implementation | Source and adaptation |
| --- | --- | --- |
| ODSR | `external/ODSR/ODSR.py`, `scripts/train_original.py` | Bundled upstream model; common validation split/checkpoint selection is study-specific |
| EGCL | `src/social_unlearning/egcl.py`, `scripts/run_egcl_backbone.py` | PyTorch equation port of the upstream TensorFlow model; full-candidate chunked contrastive computation and alternating optimizers |
| Graph-only | `run_edge_user_odsr.py --method graph_only`, `run_egcl_edge_controls.py --method graph_only` | Original parameters on rebuilt altered graphs; no optimizer updates |
| Retained fine-tuning | Same control runners, `--method retain_finetune` | Native retained objectives and validation selection |
| SCIF | `run_edge_user_odsr.py --method scif`, `scif_edges.py` | User-embedding gradient change, fixed-probe damped HVP solve, validation-selected damping, norm control; task/backbone adaptation |
| GNNDelete | `gnndelete_edges.py`, `egcl_unlearning_adapters.py` | Loads official `DeletionLayer` from pinned source; extends masks and layer losses to the native backbone and joint social/interaction request |
| DA2Unlearn | `run_edge_user_odsr.py --method da2`, `baselines.py`, `edge_deletion.py` | Loads official difficulty estimator; native retained objective, difficulty-scaled forgetting and adaptive mixing for the social-backbone setting |

SCIF trials record conjugate-gradient status and residuals; failed solves must not be reported as successful unlearning. GNNDelete freezes original backbone parameters and learns deletion operators, using its declared request-side witnesses. DA2's optimized predictions use the altered graph. These adaptations do not inherit certifications or unchanged reproduction claims from upstream methods.

EGCL currently includes scratch, graph-only, retained fine-tuning, native SoRecRemover adaptation, and GNNDelete adaptation. This release does not expose EGCL SCIF or DA2 as supported experiments. ODSR sweeps and ablations do not imply equivalent EGCL sweep support.

The upstream revisions are pinned in `configs/upstream_sources.json`; original source URLs and licensing notes are in `THIRD_PARTY_NOTICES.md`.
