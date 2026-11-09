# Frozen method and evaluation

## Deletion semantics

User deletion removes all user-item interactions and both incoming and outgoing social edges for requested users. User/item universes remain fixed. Graph normalization and derived views are reconstructed from retained edges. Requested rows have no incident edges, including artificial social self-loops. Utility evaluates retained users; the membership audit can inspect native requested-user representations.

## ODSR SoRecRemover

The public command is `scripts/run_sorecremover.py --backbone ODSR`. Its frozen implementation is `scripts/run_odsr_native_topology.py`, supported by `src/social_unlearning/odsr_native_topology.py` and `isolated_reset.py`.

A paired probe samples at most 1,024 original training interactions without replacement. It reuses negatives and compares evaluation-mode native-loss gradients at identical parameters on the original graph and the altered graph's retained probe. The difference is normalized to a displacement of 0.001 times the relevant parameter norm. Requested embedding rows are excluded from this correction and reset independently with a fixed seed offset of 710000. Sensitivity weights are then computed once, with a mixture coefficient of 0.25. Five retained-training epochs use learning rate 0.0001 and native recommendation/contrastive losses; retained validation Recall@20 selects the checkpoint.

This is a finite-probe first-order approximation. It does not use a Hessian inverse and does not estimate the full stochastic dropout objective. Small requests may have few or no deleted interactions in the sampled probe; metadata records actual coverage. The method provides no exact-erasure certificate.

Ablations remove correction (`no_topology`), replace weights by ones (`uniform_weights`), or omit reset (`no_reset`). `full` is the frozen method. The historical reverse-loss and guarded objectives included as imported dependencies are not the proposed method in this release.

## EGCL adaptation

`scripts/run_sorecremover.py --backbone EGCL` performs a native-objective topology correction and retained recovery using the EGCL port. EGCL preserves full candidate denominators in contrastive losses. Unlike ODSR, an isolated requested user has zero propagated output under the current EGCL aggregation. Native zero-row observations cannot by themselves establish removal of influence from shared item embeddings or parameters.

## Evaluation contract

- Use seeds 42, 43, and 44, and report the number of completed seeds with mean and sample standard deviation. Do not silently mix incomplete rows with three-seed summaries.
- A seed determines the data split and deletion ordering. For the same dataset/seed/fraction, every method must use identical deleted users and the same scratch-retraining reference. Fractions select prefixes of the same ordering within a seed.
- Select training checkpoints using validation data. Do not tune on test utility or target membership labels.
- Report utility (including Recall/NDCG at 20), fidelity to retraining, forgetting diagnostics, and end-to-end runtime/peak allocated memory. Reset-only controls are needed to interpret deleted-preference diagnostics.
- A deleted positive item beating a sampled negative is a direct preference diagnostic, not membership-inference AUC or proof of shared-influence removal.
- User-level MIA requires genuine never-history-member controls. These controls have no interaction/social history in their target training graph; empty ID rows can still occur in native global objectives. State this distinction explicitly.
- The benchmark-style attack uses a user's native embedding concatenated with the mean representation of historical items and a small neural classifier. Its implementation interprets the published network description; it is not claimed as verbatim official attack code.
- Shadow/attack identities and controlled target identities are separated and checked. Weak original-target attack discrimination makes post-deletion near-chance results inconclusive. Report threshold calibration and FPR as well as AUC.
- Timing on shared GPUs is descriptive. For an efficiency claim, rerun on an otherwise idle device, including graph rebuilding, correction/weight preparation, and optimization. Resumed invocation timing may exclude earlier attempts; inspect each result's timing scope.
