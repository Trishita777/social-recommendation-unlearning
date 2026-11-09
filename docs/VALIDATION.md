# Release validation

Checked on 9 November 2026:

- **34 CPU tests passed, zero skipped**, after fetching the pinned GNNDelete and DA2Unlearn source checkouts. Tests exercise graph isolation/rebuilding, native correction gradients, resets, checkpoint recovery, baseline adapters, and controlled-MIA separation.
- **10 synthetic GPU commands passed** in a temporary clean copy: original training, scratch retraining, graph-only, retained fine-tuning, and the corrected method for each of ODSR and EGCL. They produced nine result files and 18 finite Recall@20/NDCG@20 values. Original training was deliberately limited to two epochs for this integration check. These are not paper results.
- All experiment CLI help entry points completed successfully. Default ODSR and EGCL reproduction plans expanded to 48 and 36 sequential commands, respectively.
- Editable package installation and importing the package from outside the source directory succeeded.
- Every file listed in the frozen method manifest matched its recorded SHA-256. Release helpers leave those frozen implementation files unchanged.
- Both official datasets were downloaded from the pinned revision and passed size/SHA-256 verification. The bundled ODSR model, utilities, and license matched their upstream bytes exactly.
- Source publishing excludes datasets, checkpoints, result directories, upstream Git checkouts, and machine-specific connection details.

Reproduce the checks:

```bash
python scripts/prepare_dependencies.py
python scripts/verify_source.py
python -m pytest -q
python scripts/smoke_gpu.py --device cuda:0
```

The GPU smoke test requires CUDA and writes synthetic data only to a temporary copy. The GitHub workflow runs CPU tests and command-plan checks; it does not run full-data experiments or the GPU smoke test.

The tests establish executable integration and selected invariants. They do not establish full-data convergence, completion of every experiment, superiority over baselines, exact/certified unlearning, or GPU timing reproducibility. Full dataset experiments remain a separate evidence record. Deprecation warnings from upstream sparse tensor construction and a test-only tensor-to-scalar conversion are currently expected.
