# Third-party sources

- **ODSR:** https://github.com/Rongfeng-Guo/ODSR-KDD-2026 — `external/ODSR/ODSR.py` and `common.py` are bundled with the original MIT license at `external/ODSR/LICENSE`. Copyright (c) 2026 Rongfeng-Guo.
- **EGCL:** https://github.com/wubinzzu/EGCL — reference for the local PyTorch equation port. The original TensorFlow repository can be fetched separately for inspection; it is not redistributed here.
- **GNNDelete:** https://github.com/mims-harvard/GNNDelete — the pinned upstream `DeletionLayer` is loaded from a separately fetched checkout. Upstream license and notices remain in that checkout.
- **DA2Unlearn:** https://gitlab.com/hcdou/da2unl — the pinned upstream difficulty estimator is loaded from a separately fetched checkout. Consult that repository for its license and citation.
- **SCIF / ERASE:** https://github.com/deem-data/erase-bench — reference for selective collaborative influence and the damped HVP solver adaptation. This repository's social-backbone adaptation does not constitute an unchanged reproduction of the benchmark.
- **CURE4Rec:** https://github.com/xiye7lai/CURE4Rec — reference for fixed-ID edge-removal semantics and benchmark-style membership features. Our social-edge extension and neural attack implementation are explicit adaptations.

Exact fetched revisions are recorded in `configs/upstream_sources.json`. Cite the corresponding original works when using their methods. Each upstream project's original licensing terms apply to its files. This notice does not relicense upstream code or grant a new license for the newly authored research code.
