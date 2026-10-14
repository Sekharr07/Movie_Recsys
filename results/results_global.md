# Results - split: one global time cutoff (newest 20% of all ratings)

Dataset: {"users": 943, "items": 1682, "ratings": 100000, "sparsity_pct": 93.7, "share_of_ratings_on_top20pct_items": 0.646}

Chosen hyper-parameters (on validation): ItemKNN k=20, BPR reg=0.01, hybrid ItemKNN weight=0.3

## Top-10 ranking (relevant = rated >= 4 in the test period)

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Novelty | Users |
|---|---|---|---|---|---|---|---|
| Popularity | 0.1082 | 0.0830 | 0.1341 | 0.4592 | 0.0452 | 8.0143 | 98 |
| ItemKNN | 0.0765 | 0.0491 | 0.0919 | 0.3163 | 0.1587 | 8.7790 | 98 |
| BiasedMF (rating-ranked) | 0.0694 | 0.0527 | 0.0939 | 0.3367 | 0.1052 | 10.0528 | 98 |
| BPR-MF | 0.1194 | 0.0839 | 0.1498 | 0.4694 | 0.1980 | 8.8883 | 98 |
| Hybrid (ItemKNN + BPR-MF) | 0.1020 | 0.0725 | 0.1225 | 0.4082 | 0.1700 | 8.7567 | 98 |

## Is each model really different from Popularity? (paired bootstrap over users)

| Model | NDCG@10 | Diff vs Popularity | 95% CI | Distinguishable from Popularity? |
|---|---|---|---|---|
| Popularity | 0.1341 | (baseline) | | |
| ItemKNN | 0.0919 | -0.0422 | [-0.0797, -0.0051] | yes (worse) |
| BiasedMF (rating-ranked) | 0.0939 | -0.0402 | [-0.0819, +0.0001] | no (within noise) |
| BPR-MF | 0.1498 | +0.0157 | [-0.0207, +0.0509] | no (within noise) |
| Hybrid (ItemKNN + BPR-MF) | 0.1225 | -0.0116 | [-0.0497, +0.0252] | no (within noise) |

## Rating prediction (RMSE, lower is better)

| Model | RMSE |
|---|---|
| Global mean | 1.1190 |
| User+item bias | 1.0322 |
| BiasedMF (rating-ranked) | 1.0228 |
