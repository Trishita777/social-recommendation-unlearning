# GraphMoE SoRecRemover

This repository implements partition-based user withdrawal for social recommendation with ODSR and DiffNet++. It includes full retraining, RecEraser-social, UltraRE-social, GraphEraser-social, GUIDE-Fast-social, the EEMU partition control, and the SoRecRemover partition and aggregation variants.

GraphMoE refits affected experts from scratch and rebuilds the catalogs and router using retained data under the original shard assignment. A fresh partition of the retained graph can differ from that assignment.

## Run an experiment

Install dependencies with `python -m pip install -e .` in the release checkout. Place the Douban-Book and Yelp2018 `train.txt`, `test.txt`, and `trust.txt` or `links.txt` files from the [ODSR dataset](https://github.com/Rongfeng-Guo/ODSR-KDD-2026/tree/main/dataset) in `external/ODSR/dataset/<dataset>/`. Resolve Git LFS pointers before running.

```bash
python scripts/run_partition_social.py --backbone ODSR --dataset douban-book \
  --seed 42 --device cuda:0 --fraction .01 --shards 10 --epochs 30 \
  --validation-every 2 \
  --methods retrain receraser_social ultrare_social grapheraser_social guide_social \
    eemu_partition sorecremover_partition sorecremover_graph_moe \
  --output results/douban-book-odsr-seed42.json
```

Repeat with seeds 43 and 44, datasets `douban-book` and `yelp2018`, and backbones `ODSR` and `DiffNet++`. Run `python scripts/run_partition_epoch_sweep.py --help` for the resumable benchmark runner. Run GraphMoE as the sole method for ablation runs. The paired ablations use `--ablation one_hop_partition`, `--ablation user_balance_only`, and `--router-ablations`. Shard-count sensitivity runs use 10, 20, and 50 shards.

See [the study protocol](docs/PARTITION_SOCIAL_STUDY.md) for design and experimental details and [the verified benchmark snapshot](artifacts/graph-moe-results.json) for completed runs and ablations. The planned benchmark contains 96 runs, 16 additional ablation runs, and six router cases.

Regenerate the tables with `python scripts/export_graph_moe_overleaf.py --allow-partial --source artifacts/graph-moe-results.json --output graph-moe-results.tex`.
