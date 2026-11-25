# Partition-based social recommendation unlearning

This study measures retained-user ranking utility and deletion cost when ODSR or DiffNet++ is trained as an ensemble of user shards. The benchmark uses adaptations of RecEraser, UltraRE, GraphEraser, GUIDE, and the EEMU partition control because their released experiments use different backbones or prediction tasks.

| Adaptation | Partitioning | Prediction |
| --- | --- | --- |
| RecEraser-social | Capacity-constrained clustering on 32-dimensional truncated-SVD representations of training interactions | The user's own shard supplies their user vector. A learned item-wise attention combines item vectors across shards. |
| UltraRE-social | Capacity-constrained clustering with an entropic optimal-transport assignment on the same features | The user's own shard supplies their user vector. Mean scores across shard item embeddings implement the simplified ensemble. |
| GraphEraser-social | Capacity-constrained label propagation on the symmetrized social graph | The user's own shard supplies their user vector. Validation learns one global convex combination of the item scores. |
| GUIDE-Fast-social | Sparse guided spectral partitioning with balanced training-activity strata | Mirror repair for cut social links and pyramid-match similarity weights over shard item predictions. |
| EEMU partition control | Random per-user interaction shards and independently assigned social pairs | Unweighted mean of shard scores. |
| Full retraining | The complete retained graph | One backbone model trained from scratch. |

Each user belongs to exactly one shard. Interactions follow the user into their shard. The item catalog is shared, and each model trains its own item embeddings from a common, data-independent initialization. Shard graphs retain only within-shard social links. For a deletion batch, the code removes requested users and their incident edges from each affected shard, fits those shards from scratch, and refits the learned aggregation on retained validation interactions. Shard assignments are fixed after original training. The protocol measures model update cost conditional on those assignments and does not establish erasure of the influence of deleted records on the partition.

The upstream method designs and repositories used for the adaptations are [RecEraser](https://arxiv.org/abs/2201.06820) ([source](https://github.com/chenchongthu/Recommendation-Unlearning/tree/e23f6432a7fd8dd72899dec165ac3a215f8e0bec)), [UltraRE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29a0ea49a103a233b17c0705cdeccb66-Abstract-Conference.html) ([source](https://github.com/ZhangYizhao/UltraRE/tree/d81c767678d2aa5fbf456f0aa079f9a32198ce8c)), and [GraphEraser](https://arxiv.org/abs/2103.14991) ([source](https://github.com/MinChen00/Graph-Unlearning/tree/3da805e7ef5a68d1fc7fe1b82d9dfd775f599526)).

## Partition-based SoRecRemover

Partition-based SoRecRemover extends the UltraRE-social adaptation in two places. The design aims to preserve social context during partitioning and preserve reliable local item representations during recommendation.

**Socially informed transport.** For user $u$, let $x_u$ be their L2-normalized training-interaction SVD vector and $m_u$ the normalized mean of their social neighbors' vectors. The clustering feature is $z_u=[x_u,\sqrt{0.5}m_u]$. Each clustering iteration computes squared distances from $z_u$ to shard centers. After the first assignment, the cost of assigning $u$ to shard $k$ also includes $0.5(\max_j p_{uj}-p_{uk})$, where $p_{uk}$ is the fraction of $u$'s social neighbors assigned to $k$. The entropic transport step uses unit user mass, shard capacities as column marginals, and an entropy scale of 0.1 times the median assignment cost with a numerical floor. We run 50 Sinkhorn scaling iterations per clustering step. A greedy hard assignment enforces the exact user quotas, and the centers update for up to 20 clustering steps. Users without social neighbors contribute zero social context and zero neighbor penalty.

**Item aggregation by local evidence.** Each shard trains an independent model of the selected backbone. For item $i$, shard $k$ contributes to the shared item vector in proportion to $\log(1+n_{ki})$, where $n_{ki}$ is the number of retained training interactions with $i$ in that shard. The item vector used to score users in shard $k$ interpolates between the local vector and the shared vector, with local weight $n_{ki}/(n_{ki}+4)$. Users are scored using their own shard's user vectors. An item absent from every shard uses the mean item vector. Aggregation requires interaction counts and a weighted sum, with no fitted aggregation parameters.

**Deletion and scope.** The partition remains fixed. The update rebuilds and refits every shard that contains a requested user, removes those users from the deployed catalogs, and recomputes the item weights from retained interactions. Unaffected shard embeddings can be reused. With random batch withdrawals, all shards may be affected. The original partition was derived from training interactions and social links, so this experiment measures update cost and ranking conditional on that partition.

The comparison uses ODSR, Epinions and Yelp2018, five shards, uniform 1% withdrawal, seeds 42–44, and five epochs per fitted model with validation at epochs 1, 2, 4, and 5. It reuses the prior seed-42 baseline runs under the same protocol. Run just the proposed method with `--methods sorecremover_partition`. Use the loop below to reproduce the full comparison on one GPU.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for dataset in epinions yelp2018; do
  for seed in 42 43 44; do
    python scripts/run_partition_social.py --dataset "$dataset" --seed "$seed" \
      --device cuda:0 --epochs 5 --validation-every 2 \
      --output "results/partition_social/three_seed/${dataset}_seed${seed}_reproduction.json"
  done
done
```

## Evaluation

- Use the existing Epinions and Yelp2018 splits and the same uniform 1% withdrawal request across methods within seed.
- Measure full-catalog Recall and NDCG at 10 and 20 on retained test users, masking their training positives. Validation Recall@20 selects each shard checkpoint.
- Measure sequential, synchronized GPU wall time for graph reconstruction, fitting, validation, and aggregation. Report original ensemble setup, partitioning, and final test time separately. The update starts with original shard embeddings in memory. All methods use the same per-model epoch budget and validation schedule.
- Report the number of affected shards and pre-deletion within-shard social edge fraction. A random request spanning all shards removes the selective-retraining advantage, even though training on smaller graphs can still change runtime.
- Use the short, five-epoch budget for the initial paired comparison. The default runner budget is 30 epochs. A single seed and shared hardware limit claims about utility at convergence and controlled speedup.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python scripts/run_partition_social.py --dataset epinions --seed 42 --device cuda:0 \
  --epochs 5 --validation-every 2 --output results/partition_social/epinions_seed42.json
python scripts/run_partition_social.py --dataset yelp2018 --seed 42 --device cuda:1 \
  --epochs 5 --validation-every 2 --output results/partition_social/yelp2018_seed42.json
```

## Measured results

| Dataset | Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Update (s) | Original ensemble setup (s) | Models refit | Within-shard social links |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epinions | Full retraining | 0.0483 | 0.0347 | 0.0742 | 0.0425 | 31.6 | — | 1 | 100.0% |
| Epinions | RecEraser-social | 0.0210 | 0.0171 | 0.0325 | 0.0207 | 27.6 | 28.7 | 5 | 31.7% |
| Epinions | UltraRE-social | 0.0241 | 0.0188 | 0.0376 | 0.0229 | 27.3 | 29.0 | 5 | 35.5% |
| Epinions | GraphEraser-social | 0.0280 | 0.0202 | 0.0422 | 0.0244 | 27.2 | 27.3 | 5 | 48.6% |
| Yelp2018 | Full retraining | 0.0464 | 0.0395 | 0.0771 | 0.0505 | 366.6 | — | 1 | 100.0% |
| Yelp2018 | RecEraser-social | 0.0232 | 0.0210 | 0.0388 | 0.0267 | 231.1 | 232.4 | 5 | 48.7% |
| Yelp2018 | UltraRE-social | 0.0296 | 0.0255 | 0.0485 | 0.0324 | 239.9 | 235.5 | 5 | 52.5% |
| Yelp2018 | GraphEraser-social | 0.0301 | 0.0250 | 0.0494 | 0.0321 | 93.1 | 101.6 | 5 | 46.0% |

Each row is one run at seed 42 with five training epochs per model. The JSON files give partition and test times separately, as well as peak allocated GPU memory. Epinions results and Yelp2018 results.

Yelp2018 GraphEraser-social ran on GPU 0, and the other Yelp2018 methods ran on GPU 1. Both GPUs are H100 NVL devices. A second full-retraining run on GPU 0 took 150.5 seconds with Recall@20 of 0.0771, compared with 93.1 seconds for GraphEraser-social on the same device. The times describe observed wall-clock cost on shared hardware. GPU 0 full-retraining control.

### Longer Epinions run

| Dataset | Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Update (s) | Original ensemble setup (s) | Models refit | Within-shard social links |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epinions | Full retraining | 0.0483 | 0.0347 | 0.0742 | 0.0425 | 88.0 | — | 1 | 100.0% |
| Epinions | RecEraser-social | 0.0196 | 0.0158 | 0.0318 | 0.0196 | 73.9 | 79.1 | 5 | 31.7% |
| Epinions | UltraRE-social | 0.0258 | 0.0194 | 0.0400 | 0.0237 | 75.4 | 76.1 | 5 | 35.5% |
| Epinions | GraphEraser-social | 0.0292 | 0.0206 | 0.0440 | 0.0250 | 78.6 | 78.9 | 5 | 48.6% |

This additional run uses the same split and withdrawal request and raises the per-model training budget to 15 epochs. Epinions longer-run results.

## Three-seed comparison

| Dataset | Method | Recall@20 | NDCG@20 | Update (s) | Update / same-device retrain | Setup (s) | Social links within shard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epinions | Full retraining | 0.0746 ± 0.0011 | 0.0429 ± 0.0006 | 62.6 ± 27.0 | 1.00 ± 0.00 | — | 100.0 ± 0.0% |
| Epinions | RecEraser-social | 0.0315 ± 0.0013 | 0.0201 ± 0.0007 | 53.4 ± 22.3 | 0.86 ± 0.02 | 55.2 ± 22.9 | 31.6 ± 0.1% |
| Epinions | UltraRE-social | 0.0388 ± 0.0015 | 0.0236 ± 0.0014 | 52.4 ± 21.7 | 0.84 ± 0.03 | 55.4 ± 22.9 | 35.5 ± 0.1% |
| Epinions | GraphEraser-social | 0.0417 ± 0.0010 | 0.0236 ± 0.0010 | 52.7 ± 22.2 | 0.85 ± 0.03 | 54.0 ± 23.3 | 41.5 ± 8.7% |
| Epinions | SoRecRemover (partition-based) | 0.0525 ± 0.0018 | 0.0321 ± 0.0010 | 65.0 ± 3.0 | 0.83 ± 0.04 | 69.6 ± 4.3 | 48.9 ± 2.3% |
| Yelp2018 | Full retraining | 0.0768 ± 0.0003 | 0.0504 ± 0.0003 | 225.5 ± 122.5 | 1.00 ± 0.00 | — | 100.0 ± 0.0% |
| Yelp2018 | RecEraser-social | 0.0378 ± 0.0009 | 0.0261 ± 0.0005 | 141.3 ± 77.8 | 0.63 ± 0.05 | 145.1 ± 75.7 | 48.3 ± 0.4% |
| Yelp2018 | UltraRE-social | 0.0488 ± 0.0010 | 0.0324 ± 0.0006 | 140.5 ± 86.1 | 0.61 ± 0.06 | 144.6 ± 78.8 | 52.6 ± 0.3% |
| Yelp2018 | GraphEraser-social | 0.0488 ± 0.0012 | 0.0318 ± 0.0004 | 96.1 ± 2.6 | 0.63 ± 0.04 | 98.4 ± 4.6 | 43.2 ± 2.4% |
| Yelp2018 | SoRecRemover (partition-based) | 0.0689 ± 0.0003 | 0.0459 ± 0.0003 | 130.4 ± 55.0 | 0.60 ± 0.06 | 139.1 ± 54.9 | 81.4 ± 0.9% |

Reported values are means and sample standard deviations over seeds 42–44. The update ratio divides each run by its same-seed, same-device full-retraining time before averaging. The H100 NVL GPUs were shared, so even same-device comparisons reflect variable workload. Setup includes partitioning, pre-deletion shard training, and any learned aggregation. Per-seed metrics and summary.

SoRecRemover improves Recall@20 over UltraRE-social on every paired seed, by 0.0138 on Epinions and 0.0201 on Yelp2018 on average. It retains more social links within shards, and its mean update-to-retraining ratios are close to UltraRE-social's on both datasets. The experiment changes partitioning and aggregation together, so it measures their combined effect. Each sampled withdrawal touches all five shards. The measured update cost therefore includes a refit of the entire ensemble. Full retraining has the highest ranking utility on both datasets.

## DiffNet++ protocol

The same partitions, requested users, aggregation rules, and full-catalog ranking evaluator are used with DiffNet++. Each shard builds its own retained interaction and social graph and trains the released two-layer, no-attribute model. Training uses 64-dimensional embeddings, Adam at 0.0005, batches of 512 users, eight negative draws per positive, and the native sigmoid squared-error objective. All shards share data-independent item initialization within a seed. As in the ODSR partition benchmark, full-catalog validation Recall@20 selects the checkpoint under a five-epoch budget. The epoch budget controls passes over each shard. Optimizer-step counts vary with shard size. All methods use the same full-catalog evaluation protocol.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for dataset in epinions yelp2018; do
  for seed in 42 43 44; do
    python scripts/run_partition_social.py --backbone DiffNet++ --dataset "$dataset" --seed "$seed" \
      --device cuda:0 --epochs 5 --validation-every 2 \
      --output "results/partition_social/diffnet_three_seed/${dataset}_seed${seed}.json"
  done
done
```

## DiffNet++ three-seed results

| Dataset | Method | Recall@20 | NDCG@20 | Update (s) | Update / retrain | Original setup (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Epinions | Full retraining | 0.0322 ± 0.0002 | 0.0170 ± 0.0004 | 41.8 ± 1.9 | 1.00 ± 0.00 | — |
| Epinions | RecEraser-social | 0.0016 ± 0.0002 | 0.0009 ± 0.0001 | 37.5 ± 0.4 | 0.90 ± 0.05 | 38.6 ± 0.5 |
| Epinions | UltraRE-social | 0.0008 ± 0.0002 | 0.0004 ± 0.0000 | 35.9 ± 1.3 | 0.86 ± 0.01 | 38.5 ± 0.5 |
| Epinions | GraphEraser-social | 0.0011 ± 0.0002 | 0.0006 ± 0.0001 | 36.3 ± 1.7 | 0.87 ± 0.04 | 37.7 ± 1.9 |
| Epinions | SoRecRemover (partition-based) | 0.0012 ± 0.0001 | 0.0006 ± 0.0000 | 35.6 ± 2.1 | 0.85 ± 0.07 | 40.0 ± 1.3 |
| Yelp2018 | Full retraining | 0.0355 ± 0.0016 | 0.0231 ± 0.0012 | 69.1 ± 2.4 | 1.00 ± 0.00 | — |
| Yelp2018 | RecEraser-social | 0.0164 ± 0.0002 | 0.0114 ± 0.0002 | 61.3 ± 1.9 | 0.89 ± 0.02 | 66.3 ± 3.8 |
| Yelp2018 | UltraRE-social | 0.0141 ± 0.0013 | 0.0092 ± 0.0008 | 57.1 ± 1.8 | 0.83 ± 0.01 | 66.0 ± 1.0 |
| Yelp2018 | GraphEraser-social | 0.0114 ± 0.0023 | 0.0072 ± 0.0015 | 61.6 ± 1.3 | 0.89 ± 0.04 | 63.1 ± 1.0 |
| Yelp2018 | SoRecRemover (partition-based) | 0.0345 ± 0.0011 | 0.0222 ± 0.0007 | 58.3 ± 2.2 | 0.84 ± 0.03 | 66.8 ± 2.7 |

Values are means and sample standard deviations over seeds 42–44. Update ratios pair each method with full retraining on the same seed and same shared H100 device. Original setup is measured separately. Each case matches the ODSR split, withdrawal request, test-user count, and shard statistics. Full DiffNet++ per-seed measurements.

On Yelp2018, partition-based SoRecRemover had the highest mean Recall@20 and NDCG@20 among the partitioned methods. It retained approximately 97% of full retraining's mean Recall@20 and used 84% of its update time. On Epinions, all partitioned methods averaged Recall@20 below 0.002, compared with 0.0322 for full retraining. These results describe the five-epoch protocol. Each withdrawal affected all five shards on both datasets, so the measured update time includes refitting the full ensemble.

## GUIDE-Fast-social protocol

GUIDE is an inductive graph unlearning framework with guided partitioning, subgraph repair, and similarity-based aggregation. This adaptation uses the GPFB-Fast variant and transfers those three components to retained-user recommendation on ODSR and DiffNet++. [USENIX Security ’23 paper](https://www.usenix.org/conference/usenixsecurity23/presentation/wang-cheng-long) and [upstream implementation](https://github.com/Happy2Git/GUIDE/tree/548753a4e7858940a2c17ec1bd1a08b0b56b9c9c).

- **Partitioning.** Apply the sparse generalized power iteration to the degree-normalized user–user adjacency, using the number of training interactions to define activity strata. The activity strata substitute for classification labels. Cluster the spectral rows, then apply capacity-constrained rounding within each stratum. Shard sizes and per-stratum counts differ by at most one. The relaxation uses ten iterations and a guidance coefficient of 0.01 divided by the largest stratum size.
- **Repair.** For each cut edge with a surviving endpoint in another shard, add one mirror contribution from the retained user's own embedding. ODSR folds mirror neighbors into a weighted self-message. DiffNet++ adds self-edges with independent edge attention and a shared source embedding. A withdrawal invalidates the deleted user's shard and any shard whose retained users lose a mirror contribution.
- **Aggregation.** Users use their own shard's user embedding. A normalized pyramid-match kernel compares each retained shard's real social graph with the full retained social graph, which serves as the query graph for this known-user ranking task. The graph signature uses six sparse spectral coordinates and four histogram levels, with per-graph normalization for graph size. The normalized similarities weight item scores across shards. The aggregation uses retained social structure and requires no fitted combiner.

The five-epoch, five-shard, 1% deletion protocol reuses the splits, seed-specific requests, and full-catalog ranking evaluator from the existing benchmark. The adaptation evaluates ranking and wall time under fixed partitions. It does not test inductive node classification, membership leakage, or the original GUIDE datasets.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for backbone in ODSR DiffNet++; do
  for dataset in epinions yelp2018; do
    for seed in 42 43 44; do
      python scripts/run_partition_social.py --backbone "$backbone" --dataset "$dataset" \
        --seed "$seed" --device cuda:0 --epochs 5 --validation-every 2 --methods guide_social \
        --output "results/partition_social/guide_three_seed/${backbone}_${dataset}_seed${seed}.json"
    done
  done
done
```

## GUIDE-Fast-social three-seed results

| Backbone | Dataset | Method | Recall@20 | NDCG@20 | Update (s) | Update / retrain |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ODSR | Epinions | Full retraining | 0.0746 ± 0.0011 | 0.0429 ± 0.0006 | 62.6 ± 27.0 | 1.00 ± 0.00 |
| ODSR | Epinions | GUIDE-Fast-social | 0.0413 ± 0.0002 | 0.0233 ± 0.0004 | 66.4 ± 0.5 | 0.84 ± 0.02 |
| ODSR | Epinions | Partition-based SoRecRemover | 0.0525 ± 0.0018 | 0.0321 ± 0.0010 | 65.0 ± 3.0 | 0.83 ± 0.04 |
| ODSR | Yelp2018 | Full retraining | 0.0768 ± 0.0003 | 0.0504 ± 0.0003 | 225.5 ± 122.5 | 1.00 ± 0.00 |
| ODSR | Yelp2018 | GUIDE-Fast-social | 0.0544 ± 0.0008 | 0.0357 ± 0.0004 | 99.7 ± 3.5 | 0.65 ± 0.04 |
| ODSR | Yelp2018 | Partition-based SoRecRemover | 0.0689 ± 0.0003 | 0.0459 ± 0.0003 | 130.4 ± 55.0 | 0.60 ± 0.06 |
| DiffNet++ | Epinions | Full retraining | 0.0322 ± 0.0002 | 0.0170 ± 0.0004 | 41.8 ± 1.9 | 1.00 ± 0.00 |
| DiffNet++ | Epinions | GUIDE-Fast-social | 0.0006 ± 0.0000 | 0.0003 ± 0.0000 | 35.4 ± 2.3 | 0.85 ± 0.09 |
| DiffNet++ | Epinions | Partition-based SoRecRemover | 0.0012 ± 0.0001 | 0.0006 ± 0.0000 | 35.6 ± 2.1 | 0.85 ± 0.07 |
| DiffNet++ | Yelp2018 | Full retraining | 0.0355 ± 0.0016 | 0.0231 ± 0.0012 | 69.1 ± 2.4 | 1.00 ± 0.00 |
| DiffNet++ | Yelp2018 | GUIDE-Fast-social | 0.0082 ± 0.0005 | 0.0054 ± 0.0002 | 60.4 ± 2.0 | 0.87 ± 0.00 |
| DiffNet++ | Yelp2018 | Partition-based SoRecRemover | 0.0345 ± 0.0011 | 0.0222 ± 0.0007 | 58.3 ± 2.2 | 0.84 ± 0.03 |

Values are means and sample standard deviations across seeds 42–44. Timing ratios pair each run with same-device, same-seed retraining on shared H100 NVLs. The update times exclude original ensemble setup. GUIDE per-seed measurements and aggregates for all baselines.

GUIDE-Fast-social has the highest ODSR/Yelp2018 ranking utility among the adapted literature baselines. Partition-based SoRecRemover has higher mean Recall@20 and NDCG@20 in all four settings. GUIDE-Fast-social has the lowest mean Recall@20 among the partitioned variants in both DiffNet++ settings under the five-epoch budget. All twelve withdrawal requests affected all five GUIDE shards.

## EEMU partition control

The PoPETs 2025 paper *Meta-Learn to Unlearn* introduces EEMU, which uses per-user random interaction partitioning, model-agnostic meta-learning, few-shot recommendation specialization, and static score averaging. The benchmark transfers the paper's partitioning and averaging components to the ODSR and DiffNet++ trainers. The row therefore measures an EEMU partition control with the shared native training objectives. The EEMU meta-learning and inference specialization stages are outside this control. [PoPETs paper](https://www.petsymposium.org/popets/2025/popets-2025-0152.php) and [upstream implementation](https://github.com/alshabae/EEMU/tree/7d68bbd84309a4103b2b6f0ad99309299ae71909).

- Shuffle each user's training interactions and divide them evenly across the five shards. A seeded random rotation spreads the remainder interactions across shard indices. Each training interaction belongs to one shard, while a user can appear in several shards.
- Assign each unordered social pair to one shard using a separate random assignment. Reciprocal edges stay together. A shard contains the users incident to its assigned interactions or social links. The item catalog remains shared.
- When a user withdraws, refit each shard containing any of that user's training interactions or incident social links. Compact each affected graph using the remaining edges. At ranking time, average the independent shard scores. An absent user contributes a zero score for that shard, and the final full-catalog evaluator masks every retained training positive.
- Use the GUIDE comparison's splits, 1% requests, five epochs, and seeds 42–44. Update time includes refitting affected shards and reconstructing the scoring ensemble. The comparison holds the partition fixed after a request and measures ranking utility and cost.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for backbone in ODSR DiffNet++; do
  for dataset in epinions yelp2018; do
    for seed in 42 43 44; do
      python scripts/run_partition_social.py --backbone "$backbone" --dataset "$dataset" \
        --seed "$seed" --device cuda:0 --epochs 5 --validation-every 2 --methods guide_social eemu_partition \
        --output "results/partition_social/eemu_three_seed/${backbone}_${dataset}_seed${seed}_reproduction.json"
    done
  done
done
```

## Matched EEMU and GUIDE results

| Backbone | Dataset | Method | Recall@20 | NDCG@20 | Update (s) | Update / retrain |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ODSR | Epinions | Full retraining | 0.0746 ± 0.0011 | 0.0429 ± 0.0006 | 62.6 ± 27.0 | 1.00 ± 0.00 |
| ODSR | Epinions | GUIDE-Fast-social | 0.0413 ± 0.0002 | 0.0233 ± 0.0004 | 66.4 ± 0.5 | 0.84 ± 0.02 |
| ODSR | Epinions | EEMU partition control | 0.0342 ± 0.0008 | 0.0201 ± 0.0008 | 114.8 ± 17.2 | 1.46 ± 0.24 |
| ODSR | Epinions | Partition-based SoRecRemover | 0.0525 ± 0.0018 | 0.0321 ± 0.0010 | 65.0 ± 3.0 | 0.83 ± 0.04 |
| ODSR | Yelp2018 | Full retraining | 0.0768 ± 0.0003 | 0.0504 ± 0.0003 | 225.5 ± 122.5 | 1.00 ± 0.00 |
| ODSR | Yelp2018 | GUIDE-Fast-social | 0.0544 ± 0.0008 | 0.0357 ± 0.0004 | 99.7 ± 3.5 | 0.65 ± 0.04 |
| ODSR | Yelp2018 | EEMU partition control | 0.0486 ± 0.0005 | 0.0318 ± 0.0003 | 188.7 ± 3.2 | 1.23 ± 0.06 |
| ODSR | Yelp2018 | Partition-based SoRecRemover | 0.0689 ± 0.0003 | 0.0459 ± 0.0003 | 130.4 ± 55.0 | 0.60 ± 0.06 |
| DiffNet++ | Epinions | Full retraining | 0.0322 ± 0.0002 | 0.0170 ± 0.0004 | 41.8 ± 1.9 | 1.00 ± 0.00 |
| DiffNet++ | Epinions | GUIDE-Fast-social | 0.0006 ± 0.0000 | 0.0003 ± 0.0000 | 35.4 ± 2.3 | 0.85 ± 0.09 |
| DiffNet++ | Epinions | EEMU partition control | 0.0119 ± 0.0006 | 0.0069 ± 0.0004 | 124.2 ± 2.1 | 2.98 ± 0.19 |
| DiffNet++ | Epinions | Partition-based SoRecRemover | 0.0012 ± 0.0001 | 0.0006 ± 0.0000 | 35.6 ± 2.1 | 0.85 ± 0.07 |
| DiffNet++ | Yelp2018 | Full retraining | 0.0355 ± 0.0016 | 0.0231 ± 0.0012 | 69.1 ± 2.4 | 1.00 ± 0.00 |
| DiffNet++ | Yelp2018 | GUIDE-Fast-social | 0.0082 ± 0.0005 | 0.0054 ± 0.0002 | 60.4 ± 2.0 | 0.87 ± 0.00 |
| DiffNet++ | Yelp2018 | EEMU partition control | 0.0317 ± 0.0005 | 0.0207 ± 0.0003 | 217.9 ± 16.0 | 3.16 ± 0.33 |
| DiffNet++ | Yelp2018 | Partition-based SoRecRemover | 0.0345 ± 0.0011 | 0.0222 ± 0.0007 | 58.3 ± 2.2 | 0.84 ± 0.03 |

Values are means and sample standard deviations across seeds 42–44. Updates exclude original ensemble setup. Timing ratios pair each method with same-device, same-seed retraining on shared H100 NVLs. Every withdrawal batch affected all five EEMU partition-control shards. Per-seed EEMU data and all seven method aggregates.

GUIDE-Fast-social has higher mean Recall@20 than the EEMU partition control on both ODSR datasets. The EEMU partition control has higher mean Recall@20 than GUIDE on both DiffNet++ datasets and leads the partitioned variants on DiffNet++/Epinions. It takes longer than the paired full-retraining reference in all four settings. The five-epoch budget fixes the number of passes over each shard. User membership overlaps across edge shards, so the total number of user batches can differ from the user-partitioned methods.

## SoRecRemover with mixture-of-experts aggregation

The optional `sorecremover_moe` method keeps SoRecRemover's social transport partitions, independent shard training, and support-weighted expert item catalogs. A shard is an expert. The base method scores users with their owning shard's catalog. The mixture scores each item by a weighted sum of the expert catalogs using the owning shard's user embedding.

The router receives four values for each user–expert pair:

- Fraction of the user's retained training items supported by that expert.
- Fraction of the user's retained social neighbors assigned to that expert.
- Whether the expert owns the user.
- Log-transformed retained training activity.

A shared 16-unit MLP learns a residual to a prior that gives the owner 80% of the weight and shares the rest across other experts. A softmax yields one vector of expert weights per user. The router minimizes pairwise ranking loss on retained validation positives with sampled negatives and a 0.01 KL penalty toward the prior. The shard embeddings stay fixed during router fitting. The benchmark refits the router after deletion using the retained graph and excludes test data from router training and negative masking.

The final score is a dot product in concatenated expert coordinates. Full-catalog evaluation remains exact and uses the existing evaluator. The factor dimension grows with the number of experts, so report test time and memory separately from deletion update time. The results file records the mean owner weight and mean weights across experts.

Run the matched DiffNet++/Epinions diagnostic with the command below.

```bash
python scripts/run_partition_social.py --backbone DiffNet++ --dataset epinions \
  --seed 42 --device cuda:0 --epochs 15 --validation-every 2 \
  --methods retrain sorecremover_partition sorecremover_moe \
  --output results/partition_social/moe_validation/diffnet_epinions_seed42_15epochs.json
```

### Mixture-of-experts pilot results

Both pilots use seed 42, five shards, and 1% user withdrawal. The comparisons match the split, partition statistics, evaluated users, and epoch budget within each setting.

| Setting | Epochs | Existing Recall@20 | MoE Recall@20 | Existing NDCG@20 | MoE NDCG@20 | Mean MoE owner weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffNet++ / Epinions | 15 | 0.03607 | 0.03553 | 0.01952 | 0.01896 | 0.8474 |
| ODSR / Yelp2018 | 5 | 0.06885 | 0.06885 | 0.04592 | 0.04590 | 0.9937 |

On DiffNet++/Epinions, full retraining reached 0.03340 Recall@20 and 0.01734 NDCG@20 in the same 15-epoch case. The existing aggregator reached 0.00127 Recall@20 in the earlier five-epoch case with the same seed. The 15-epoch update times were 60.6 seconds for full retraining, 54.2 seconds for the existing aggregator, and 53.9 seconds for MoE. DiffNet++ 15-epoch results.

The ODSR/Yelp2018 quality comparison reuses the completed seed-42 result for the existing aggregator on GPU 1. The MoE pilot ran on GPU 0, so the table reports utility without a device-paired runtime comparison for this setting. The router assigned almost all weight to the owner shard on average. Existing ODSR result and ODSR MoE result.

## SoRecRemover with graph-aware expert routing

The optional `sorecremover_graph_moe` variant adapts graph-dependent routing and sparse expert selection from Graph Mixture of Experts. The experts are independently fitted recommender shards. One-hop and two-hop graph signals drive expert selection at aggregation. [Graph Mixture of Experts, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html).

1. **Graph-aware sharding.** Use the existing interaction-SVD preferences and add normalized one-hop and two-hop social averages before balanced transport. The existing social-edge penalty encourages neighbors to stay together. Sort users by training interaction count plus social degree, then enforce equal shard quotas within consecutive 64-per-shard workload bands. The workload is a proxy for fitting cost. The partition remains fixed after withdrawals.
2. **Expert alignment.** Choose the retained shard with the most training interactions as the reference. Align each expert to that shard using weighted orthogonal Procrustes on item anchors observed in both shards. Weights are the log-transformed smaller interaction count. The fit is regularized toward the identity because the shards share item initialization. Apply each rotation to both user and item factors, preserving that expert's dot-product scores. With fewer anchors than embedding dimensions, use the existing coordinates.
3. **Graph-aware routing.** Start with the four features from the first MoE variant and add two-hop social membership and user–item–user membership. Train a shared MLP on retained validation ranking pairs with the owner prior and a small penalty on batch-level expert weight imbalance. Shard model parameters stay fixed.
4. **Sparse aggregation and calibration.** Keep the owner expert and the highest-scoring alternate expert. Hold out 20% of validation users from router fitting and use their full-catalog Recall@20 to select a mixing strength of 0, 0.5, or 1. The owner-only choice wins ties. A value of zero uses the owning shard's aligned, support-weighted catalog. Group users by active expert pair to keep full-catalog dot-product scoring exact with at most two experts per user. The test split is used only for final evaluation.

After a withdrawal, the runner refits affected shards and rebuilds alignment, support statistics, and routing from retained data. No graph messages or cross-shard gradients enter expert training. The complete variant changes sharding, alignment, and routing together. Separate ablations would be needed to attribute any utility gain to a particular component.

```bash
python scripts/run_partition_social.py --backbone DiffNet++ --dataset epinions \
  --seed 42 --device cuda:0 --epochs 15 --validation-every 2 \
  --methods sorecremover_graph_moe \
  --output results/partition_social/graph_moe_validation/diffnet_epinions_seed42_15epochs.json
```

### Graph-aware expert routing pilot results

These matched seed-42 pilots use five shards and 1% user withdrawal. Both backbones use the existing aggregator's epoch budget for that setting. The original-aggregator reference runs came from the same GPU and protocol, with shared load at different times.

| Setting | Epochs | Existing Recall@20 | GraphMoE Recall@20 | Existing NDCG@20 | GraphMoE NDCG@20 | Existing / GraphMoE update (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffNet++ / Epinions | 15 | 0.03607 | 0.03788 | 0.01952 | 0.02016 | 54.2 / 53.5 |
| ODSR / Yelp2018 | 5 | 0.06885 | 0.07017 | 0.04592 | 0.04671 | 193.8 / 233.9 |

Both runs selected mixing strength 0.5 with the user-disjoint router calibration set. The final mean owner weights were 97.3% on Epinions and 93.8% on Yelp2018. DiffNet++ pilot and ODSR pilot.

The coefficient of variation in training interaction counts fell from 13.7% to 7.4% on Epinions and from 17.0% to 13.4% on Yelp2018. The share of social links kept within shards also fell from 46.2% to 44.4% on Epinions and from 82.0% to 65.7% on Yelp2018. These statistics characterize the seed-42 partitions. The pilot combines changes to partitioning, alignment, and aggregation, and the observed ranking gains describe the combined method. Partition diagnostics.

## Extended 30-epoch benchmark

The extended benchmark uses a 30-epoch maximum, validation at epoch 1 and every two epochs, and patience of ten validation checks. Each result records the fitted epochs, best validation checkpoint epoch, and optimizer steps for each original and updated model. The splits, seeds 42–44, 1% withdrawal requests, five shards, and evaluation metrics follow the short-budget benchmark.

The complete sweep contains 108 method runs across ODSR, DiffNet++, Epinions, and Yelp2018. It includes full retraining, RecEraser-social, UltraRE-social, GraphEraser-social, GUIDE-Fast-social, the EEMU partition control, SoRecRemover, dense SoRecRemover-MoE, and SoRecRemover-GraphMoE. Each seed, dataset, and backbone comparison runs on one physical GPU. The queue runs full retraining, SoRecRemover, and GraphMoE first for each case, then the other methods.

Local GPU 0 runs the six Yelp2018 cases. Local GPU 1 runs five Epinions cases. DGL GPU 0 runs ODSR/Epinions seed 44 under the shared lock, alongside the existing five-epoch queue. The worker reads a pinned runner snapshot, reuses only validated completed results, and writes per-method logs. The collector syncs the DGL outputs and generates the full summary after it verifies all 108 runs. Progress and partial results. The final file is `results/partition_social/epoch30_v1/summary.json`.

## Matched GraphMoE comparison

The three-seed comparison extends the existing five-epoch benchmark with GraphMoE on ODSR and DiffNet++ for Epinions and Yelp2018. It reuses all 84 baseline cells and the completed ODSR/Yelp2018 GraphMoE seed-42 run. The remaining 11 cells train GraphMoE only. Older records without deletion hashes are matched using the recorded seed, deletion count, split, and benchmark configuration. The report includes absolute wall times from shared H100 NVL GPUs. Use the recorded source and GPU fields to interpret timing differences.

The extended 30-epoch sweep was halted after its in-flight runs drained. Its completed results remain available as a separate training-budget sensitivity check. Matched comparison. Extended-benchmark results.

## Ablation study

Following UltraRE's stage-wise evaluation, we measure which partitioning and aggregation choices affect retained-user ranking and deletion cost. We hold the data split, deletion request, backbone, training budget, and seed fixed within each comparison. We use ODSR on Yelp2018 and DiffNet++ on Epinions for seeds 42–44. The full model and aggregation controls share the same trained shard experts.

- **Effect of partitioning.** Remove the two-hop preference view while retaining direct-neighbor information, then separately replace workload-aware assignment with equal user capacity. Report Recall@20, NDCG@20, the fraction of social edges retained within shards, per-shard interaction counts, and deletion update time.
- **Effect of aggregation.** Compare the full GraphMoE router with owner-only scoring, a router using local features without two-hop social or collaborative features, a router without load regularization, and a router with fixed secondary-expert mixing instead of validation calibration. Report post-deletion ranking, aggregation time, and activation of the second expert. Every variant uses the same post-deletion expert weights.
- **Effect of shard count.** Compare 3, 5, and 10 shards on seed 42 as a sensitivity check. Report ranking utility, deletion time, and affected shards. Present this sweep separately from the three-seed component comparisons.

Results are generated by `scripts/run_graph_moe_ablation.sh`. The component comparisons should report means and sample standard deviations across seeds, with paired deltas against the full variant. [UltraRE, Section 5.2.3](https://proceedings.neurips.cc/paper_files/paper/2023/file/29a0ea49a103a233b17c0705cdeccb66-Paper-Conference.pdf).

## Paper-aligned shard configuration

The main comparison uses 10 shards for every partition method and a single model for full retraining. The shard-count sensitivity uses 10, 20, and 50 shards. These are the main and sensitivity settings from UltraRE Sections 5.1.2 and 5.2.3. The earlier five-shard, five-epoch table is a pilot under its own protocol.

For ODSR and DiffNet++, we use a common maximum of 30 epochs, validate at epoch 1 and every two epochs thereafter, and stop after 10 validation checks without improvement in Recall@20. Final evaluation uses the best validation checkpoint. The 30-epoch cap is a protocol choice for these social recommendation backbones. Completed 30-epoch full-retraining runs are reused because their single-model training path is independent of the requested shard count. The component controls use 10 shards, and the main GraphMoE runs on ODSR/Yelp2018 and DiffNet++/Epinions also evaluate the routing controls on their shared trained experts.

[UltraRE](https://proceedings.neurips.cc/paper_files/paper/2023/file/29a0ea49a103a233b17c0705cdeccb66-Paper-Conference.pdf). Paper-aligned comparison progress.

The [verified snapshot](../artifacts/graph-moe-results.json) includes all 96 main runs, 16 additional ablation runs, and six router cases.
