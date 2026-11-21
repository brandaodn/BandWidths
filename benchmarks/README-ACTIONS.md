# BandWidths: reproducible GitHub Actions benchmark

Workflow: `.github/workflows/bandwidth-benchmark.yml`.

## How to start

After merging the pull request into the repository default branch, open **Actions** > **BandWidths - RCM refinement benchmark** > **Run workflow**. Select:

- **pilot**: five eligible small matrices, seed 47, budget 100 movements.
- **full**: eligible topologically distinct matrices up to 800 vertices and 12,000 edges, five seeds (47, 78, 109, 140, 171), 100/250/500 movement budgets and six matrix shards per seed/budget.

Default: two independent evaluation repetitions per matrix/seed/budget; synthetic training graph count is 55 per DQN variant.

Artifacts: `validated-matrix-inventory` (all 163 file statuses and selected manifest), `models-...` (four DQN weights per seed/budget), `training-...` (training times), `results-...` (per-evaluation data, including all baselines), and `final-bandwidth-benchmark-results` (aggregated comparisons, per-family and per-matrix tables).

## Included policies

RCM, Random dynamic, fixed adjacent swap, fixed critical-neighbor swap, Double DQN with original 10D state, enriched 29D state, two 29D feature ablations, adaptive-epsilon 29D DQN, and Discounted-UCB.

Models are trained **only on synthetic graphs** and evaluated on held-out real `.mtx` files. The real-data grouping is inferred from filenames, not validated SuiteSparse metadata. Family results are descriptive; this workflow **does not perform real-family leave-one-out retraining**.

## Validation and experimental limits

- The graph is built as a simple undirected unweighted graph from non-diagonal nonzeros. Rectangular and unsupported storage formats are excluded and documented.
- Duplicate detection uses exact labeled adjacency, not graph isomorphism.
- Budgets are numbers of proposed moves **per method**, not matched CPU time. Inference time includes RCM; training cost is reported separately.
- Significance comparisons are exploratory; do not treat individual repetitions as independent matrices or generalize beyond the eligible set.
- Upper bounds 800 vertices and 12,000 edges intentionally exclude costly instances; this is a **restricted subbenchmark**, not the full collection.
- GitHub standard runners have finite RAM, job timeouts and concurrent-job quotas; run pilot first and tune limits if any shard fails.

## Local validation (no remote service needed)

```bash
python -m pip install -r benchmarks/requirements-actions.txt
python benchmarks/rcm_actions.py inventory --repo . --output /tmp/benchmark-inventory --mode pilot --pilot-count 2
python benchmarks/rcm_actions.py train --seed-index 0 --steps 5 --train-graphs 3 --output /tmp/benchmark-models
python benchmarks/rcm_actions.py evaluate --repo . --manifest /tmp/benchmark-inventory/manifest.json --models /tmp/benchmark-models --seed-index 0 --steps 5 --reps 1 --shard 0 --shards 1 --output /tmp/benchmark-eval
python benchmarks/rcm_actions.py aggregate --input /tmp/benchmark-eval --output /tmp/benchmark-summary
```

**Caution:** Workflow setup and local smoke tests do not mean the complete benchmark has run on GitHub. Run and inspect the `pilot` job first.
