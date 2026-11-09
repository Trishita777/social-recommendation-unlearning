# SoRecRemover

Research code for **SoRecRemover**, an empirical unlearning method for social recommendation, with ODSR and EGCL backbones. The repository name is `social-recommendation-unlearning`.

**Release status:** the corrected method is frozen in [configs/frozen_corrected_method.json](configs/frozen_corrected_method.json). Experiments are ongoing. This release supplies executable code and reproduction instructions; it does not claim that every paper comparison has finished or that the method provides certified erasure.

## Method and task

A deletion request removes every requested user's interaction edges and incoming/outgoing social edges. User/item IDs and embedding-table dimensions stay fixed. Adjacency matrices and normalization are rebuilt from the remaining edges; requested users become isolated. Utility is evaluated on retained users.

The frozen ODSR method performs:

1. A normalized first-order correction from paired native-loss gradients on the original and altered graphs, using a fixed interaction probe.
2. An independent reset of isolated requested-user embedding rows.
3. Five epochs of sensitivity-weighted native retained training, selecting the checkpoint by retained validation Recall@20.

The EGCL adaptation uses its native objective and alternating optimization. EGCL's isolated-user output is structurally zero; this alone is not evidence that shared-model influence was erased. See [method and evaluation](docs/METHOD_AND_EVALUATION.md).

## Installation

Use Linux, Git, Python 3.10 or newer, and a CUDA-capable PyTorch installation for full experiments. CPU is sufficient for the unit tests. The experiment environment used Python 3.12.3 and PyTorch 2.13.0+cu130; exact observed versions are recorded in [environment](docs/ENVIRONMENT.md).

```bash
git clone https://github.com/Trishita777/social-recommendation-unlearning.git
cd social-recommendation-unlearning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Select a PyTorch build compatible with your GPU driver before running experiments. Run the commands below from the repository root; the scripts and bundled backbone are part of the source checkout.

Fetch the exact upstream baseline revisions and the official datasets:

```bash
python scripts/prepare_dependencies.py --datasets epinions yelp2018
python scripts/verify_source.py
python -m pytest -q
```

The setup script fetches GNNDelete, DA2Unlearn, and EGCL at the commits in [upstream_sources.json](configs/upstream_sources.json). It fetches Epinions/Yelp2018 from the pinned ODSR revision, resolves Git LFS pointers, and verifies file size and SHA-256. Existing data with different hashes is rejected. Downloaded data, checkpoints, and upstream checkouts are excluded from Git. See [data and dependencies](docs/DATA_AND_DEPENDENCIES.md).

## Reproduce the main comparisons

Inspect the commands without starting training:

```bash
python scripts/reproduce.py --backbone ODSR --dry-run
```

Run both datasets with seeds 42, 43, and 44 at 1% deletion:

```bash
python scripts/reproduce.py --backbone ODSR --device cuda:0
python scripts/reproduce.py --backbone EGCL --device cuda:0
```

These are substantial GPU jobs, executed sequentially per invocation. The runner trains/reuses original models, creates a matched scratch reference, and then runs the supported methods. It stops on the first failed command. Completed artifacts are reused only when the underlying script's provenance checks pass. Run different datasets/seeds in separate processes only if they write to different output directories.

For one dataset and seed:

```bash
python scripts/reproduce.py --backbone ODSR --datasets epinions --seeds 42 \
  --methods retrain graph_only retain_finetune sorecremover --device cuda:0
```

`sorecremover` maps to `run_odsr_native_topology.py --variant full` for ODSR and `run_egcl_unlearning_adapters.py --method native_topology` for EGCL. Internal filenames are preserved to retain source hashes. The historical `--method ours` in `run_edge_user_odsr.py` is an earlier method and **must not be labeled SoRecRemover**.

## Deletion sweeps and ablations

The paper sweep is scoped to ODSR on Epinions. Each fraction uses all three seeds, with matched deletion IDs across methods:

```bash
python scripts/reproduce.py --backbone ODSR --datasets epinions \
  --seeds 42 43 44 --fractions 0.005 0.01 0.02 0.05 --device cuda:0

python scripts/reproduce.py --backbone ODSR --datasets epinions \
  --seeds 42 43 44 --methods sorecremover \
  --variants full no_topology uniform_weights no_reset --device cuda:0
```

The runner includes the scratch reference even if only `sorecremover` is selected. The EGCL comparison currently supports 1% deletion; its control runner does not implement a multi-fraction sweep. See [reproduction details](docs/REPRODUCING.md) for individual commands and membership-inference evaluation.

## Included comparisons

| Method | ODSR | EGCL | Interpretation |
| --- | --- | --- | --- |
| Scratch retraining | Yes | Yes | Fresh parameters on altered graphs |
| Graph-only | Yes | Yes | Edge removal and graph rebuilding, without learning |
| Retained fine-tuning | Yes | Yes | Native retained objective |
| SoRecRemover | Yes | Yes | Frozen corrected method / explicit EGCL adaptation |
| SCIF | Yes | — | Selective influence approximation adapted to this task |
| GNNDelete | Yes | Yes | Official deletion operator with backbone-specific objectives |
| DA2Unlearn | Yes | — | Official difficulty estimator with social-backbone adaptation |

Published baselines are adaptations to the shared social-user deletion protocol, not unchanged reproductions of their original experimental setups. See [baseline notes](docs/BASELINES.md) for the differences and source attribution.

## Outputs and validation

Models, manifests, metrics, timing, and logs are written under `checkpoints/` and `results/`. Their paths encode the backbone, dataset, seed, split, request, and method version. The corrected ODSR method uses `checkpoints/odsr_native_topology_v1/`; edge controls use `checkpoints/edge_user_v1/`. Controlled MIA has separate target/checkpoint namespaces.

The test suite covers graph rebuilding, isolation/reset, correction gradients, checkpoint recovery, native losses, and MIA identity separation. [Validation notes](docs/VALIDATION.md) state what has actually been run. A passing test suite is not an empirical unlearning guarantee. Shared-GPU timings must not be used to claim an isolated speed advantage.

## Repository map

```text
configs/       Frozen method hashes and pinned upstream revisions
docs/          Setup, protocol, baseline, and reproduction notes
external/ODSR/ Bundled upstream ODSR model and utility files, with original license
scripts/       Training, unlearning, controlled MIA, and reproduction entry points
src/           Native objectives, graph utilities, adapters, and checkpoint code
tests/         Small deterministic correctness and integration tests
```

## Attribution and citation

Please acknowledge the original ODSR, EGCL, SCIF, GNNDelete, DA2Unlearn, and CURE4Rec work when using the corresponding components. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Formal paper citation metadata will be added when the title, author list, and publication identifier are finalized; none is invented in this release.

Bundled third-party code retains its original license. A license grant for the newly authored research code has not yet been selected by the authors.
