# Results - split: per-user cutoff (each user's newest 20% of ratings)

Dataset: {"users": 943, "items": 1682, "ratings": 100000, "sparsity_pct": 93.7, "share_of_ratings_on_top20pct_items": 0.646}

Chosen hyper-parameters (on validation): ItemKNN k=50, BPR reg=0.002, hybrid ItemKNN weight=0.7

## Top-10 ranking (relevant = rated >= 4 in the test period)

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Novelty | Users |
|---|---|---|---|---|---|---|---|
| Popularity | 0.0782 | 0.0805 | 0.0990 | 0.4487 | 0.0428 | 7.7419 | 907 |
| ItemKNN | 0.1212 | 0.1384 | 0.1599 | 0.5854 | 0.2568 | 8.6172 | 907 |
| BiasedMF (rating-ranked) | 0.0482 | 0.0436 | 0.0548 | 0.3098 | 0.1908 | 9.7869 | 907 |
| BPR-MF | 0.1090 | 0.1295 | 0.1474 | 0.5513 | 0.3859 | 8.7503 | 907 |
| Hybrid (ItemKNN + BPR-MF) | 0.1234 | 0.1417 | 0.1672 | 0.5943 | 0.2628 | 8.6024 | 907 |

## Is each model really different from Popularity? (paired bootstrap over users)

| Model | NDCG@10 | Diff vs Popularity | 95% CI | Distinguishable from Popularity? |
|---|---|---|---|---|
| Popularity | 0.0990 | (baseline) | | |
| ItemKNN | 0.1599 | +0.0609 | [+0.0479, +0.0745] | yes (better) |
| BiasedMF (rating-ranked) | 0.0548 | -0.0442 | [-0.0543, -0.0341] | yes (worse) |
| BPR-MF | 0.1474 | +0.0484 | [+0.0356, +0.0605] | yes (better) |
| Hybrid (ItemKNN + BPR-MF) | 0.1672 | +0.0682 | [+0.0541, +0.0818] | yes (better) |

## Rating prediction (RMSE, lower is better)

| Model | RMSE |
|---|---|
| Global mean | 1.2079 |
| User+item bias | 1.0175 |
| BiasedMF (rating-ranked) | 0.9701 |
