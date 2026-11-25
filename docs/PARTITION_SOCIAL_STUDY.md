# Douban-Book and Yelp2018 withdrawal benchmark

The primary benchmark uses ODSR on Douban-Book and Yelp2018 and a paper-based RecFlow reproduction on its native Yelp split. Each pairing has eight procedures and seeds 42, 43, and 44. Use the ODSR release's interaction splits and recorded social links. Verify the dataset hashes before running. For ODSR the training loader excludes test pairs outside the training index and reserves validation interactions per user.

| Seed-42 graph before withdrawal | Douban-Book | Yelp2018 |
|---|---:|---:|
| Users | 12,638 | 45,919 |
| Items | 22,222 | 45,538 |
| Training interactions | 432,700 | 845,453 |
| Validation interactions | 46,030 | 84,579 |
| Test interactions | 119,307 | 253,578 |
| Directed source social links | 163,748 | 709,459 |
| Withdrawn accounts | 126 | 459 |

Fit each backbone for up to 30 epochs and validate at epoch 1 and each even epoch. Use 10 shards, uniform 1% account withdrawal, and the full item catalog for ranking. ODSR uses the released dataset-specific adaptive-view, contrastive, and dropout settings. RecFlow uses two propagation layers, 64 dimensions, a conditional vector field, learning rate 0.001 with 0.96 epoch decay, 4,096 interactions per batch, and weight decay 0.001. Its Yelp archive supplies all three interaction splits and the trust graph.

Compare full retraining, RecEraser-social, UltraRE-social, GraphEraser-social, GUIDE-Fast-social, the EEMU partition control, partitioned SoRecRemover, and GraphMoE. Pair records by dataset, backbone, seed, split digest, social-source digest, and withdrawal digest. Report ranking for retained test users and time the update separately from initial setup and final evaluation.

Component controls use ODSR on Yelp2018. Evaluate one-hop partitioning and equal user capacity for three seeds. Evaluate owner-only scoring, local routing, removal of load regularization, and fixed mixing on shared experts. Compare 10, 20, and 50 shards at seed 42. The additional monolithic sensitivity study uses RecFlow on its native Yelp split and rates 0.0005, 0.001, and 0.005. Run up to 90 epochs, retain the best validation checkpoint, and use validation to select the rate independently for each seed.

GraphMoE refits changed local experts and rebuilds shared scoring under the original data-derived user assignment. This assignment remains a dependency of the retained model.
