# Baselines and backbones

All current comparisons retain the full user/item universe, remove requested users' incident interaction/social edges, rebuild normalization, and evaluate retained utility against the matching scratch reference. Older compact-row functions remain only where imported; the reproduction runner selects the current fixed-ID entry points.

| Component | Entry point / implementation | Source and adaptation |
| --- | --- | --- |
| ODSR | `scripts/train_odsr.py`; model: `external/ODSR/ODSR.py` | Bundled upstream model; common validation split/checkpoint selection is study-specific |
| EGCL | `scripts/train_egcl.py`; model: `src/social_unlearning/egcl.py` | PyTorch equation port of the upstream TensorFlow model; full-candidate chunked contrastive computation and alternating optimizers |
| Graph pruning | `scripts/run_graph_pruning.py` | Original parameters on rebuilt altered graphs; no optimizer updates |
| Retained fine-tuning | `scripts/run_retain_finetuning.py` | Native retained objectives and validation selection |
| SCIF | `scripts/run_scif.py`; implementation: `scif_edges.py` | User-embedding gradient change, fixed-probe damped HVP solve, validation-selected damping, norm control; task/backbone adaptation |
| GNNDelete | `scripts/run_gnndelete.py`; implementations: `gnndelete_edges.py`, `egcl_unlearning_adapters.py` | Loads official `DeletionLayer` from pinned source; extends masks and layer losses to the native backbone and joint social/interaction request |
| DA2Unlearn | `scripts/run_da2unlearn.py`; implementations: `baselines.py`, `edge_deletion.py` | Loads official difficulty estimator; native retained objective, difficulty-scaled forgetting and adaptive mixing for the social-backbone setting |

SCIF trials record conjugate-gradient status and residuals; failed solves must not be reported as successful unlearning. GNNDelete freezes original backbone parameters and learns deletion operators, using its declared request-side witnesses. DA2's optimized predictions use the altered graph. These adaptations do not inherit certifications or unchanged reproduction claims from upstream methods.

EGCL currently includes scratch, graph-only, retained fine-tuning, native SoRecRemover adaptation, and GNNDelete adaptation. This release does not expose EGCL SCIF or DA2 as supported experiments. ODSR sweeps and ablations do not imply equivalent EGCL sweep support.

The upstream revisions are pinned in `configs/upstream_sources.json`; original source URLs and licensing notes are in `THIRD_PARTY_NOTICES.md`.
