# Data and dependencies

`scripts/prepare_dependencies.py` reads the pinned revisions in `configs/upstream_sources.json`. GNNDelete's `DeletionLayer` and DA2Unlearn's difficulty estimator are loaded directly from these checkouts. Their entire training frameworks are not imported. EGCL's original TensorFlow source is required for the training runner's provenance hash; experiments execute the local PyTorch equation port and do not require TensorFlow.

```bash
# All source dependencies (also the default):
python scripts/prepare_dependencies.py

# Data only:
python scripts/prepare_dependencies.py --sources --datasets epinions yelp2018

# Fetch only the required original EGCL source:
python scripts/prepare_dependencies.py --sources EGCL
```

The data layout is:

```text
external/ODSR/dataset/epinions/{train.txt,test.txt,trust.txt}
external/ODSR/dataset/yelp2018/{train.txt,test.txt,trust.txt}
```

`links.txt` is supported as a social-file alternative when `trust.txt` is absent upstream. The loader uses the supplied train/test files, removes duplicate pairs and train/test overlap, maps users/items from training IDs, and creates the validation split from training interactions using the requested seed. The same data/split and request construction are used across methods. Social edges outside the mapped universe are excluded.

The downloader resolves and verifies the Git LFS objects referenced by the pinned revision. Manifests are saved to `results/data_manifests/`. Download from the original distribution and observe the dataset owners' terms; data is not redistributed in this repository.

Checkpoint caches store source and data provenance. Do not mix artifacts from a different code revision, data mapping, deletion protocol, or seed. A provenance mismatch is an error to investigate, not a check to disable. Legacy compact-row deletion artifacts are incompatible with the current fixed-ID protocol.
