# 🎬 Movie Recommender: from popularity baseline to hybrid ranking

A recommendation system built from scratch in NumPy on **MovieLens 100K** (100,000 ratings, 943 users,
1,682 movies). Five models are compared under a leakage-aware evaluation with ranking metrics,
bootstrap confidence intervals, and "beyond accuracy" measures, then served through a FastAPI backend
and a Streamlit demo.

![alt text](image.png)

## Headline result

Under a per-user chronological split (907 evaluated users, top-10 recommendations), the **hybrid
(ItemKNN + BPR-MF) reached NDCG@10 = 0.167 versus 0.099 for the popularity baseline**
(+0.068, 95% bootstrap CI [+0.054, +0.082]) and raised catalog coverage from 4.3% to 26.3%.
ItemKNN and BPR-MF also beat popularity significantly, while rating-optimized matrix factorization
ranked significantly *worse* than popularity despite the best RMSE.

Conclusions depend on the evaluation protocol: under a single global time cutoff (only ~98 evaluable
users) the picture is much less favorable. Both are reported below.

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py                  # per-user split  -> results/results_per_user.md
python run_experiments.py --split global   # global cutoff   -> results/results_global.md
pytest                                     # unit tests (metrics, splits)
uvicorn api:app --reload                   # REST API  -> http://127.0.0.1:8000/docs
python -m streamlit run app.py             # interactive demo
```

The first run downloads MovieLens 100K automatically. If the download fails (e.g. a certificate error on
Windows), download `ml-100k.zip` from GroupLens manually and extract it so that
`data/ml-100k/u.data` exists. The dataset is not redistributed in this repository.

## Models

| Stage | Model | Idea |
|---|---|---|
| 1 | **Popularity**, global mean, user+item bias | Baselines every other model must beat |
| 2 | **ItemKNN** | Item-item cosine similarity (with shrinkage, top-k neighbours): "people who interacted with X also interacted with Y" |
| 3 | **BiasedMF** | Matrix factorization trained with SGD to predict ratings |
| 3 | **BPR-MF** | Matrix factorization trained for *ranking*: items a user interacted with should outscore random ones |
| 5 | **Hybrid** | Per-user z-scored blend of ItemKNN and BPR-MF scores |
| 5 | **GenreColdStart** | New users pick genres; movies are ranked by a Bayesian-average rating |

All models are plain NumPy (`recsys/models.py`), so every line is readable.

## Evaluation protocol

* **Chronological splits, never random** (70% train / 10% validation / 20% test). A random split lets a
  model train on a user's future ratings, which cannot happen in production. Two variants are reported:
  * **Global cutoff:** the newest 20% of all ratings form the test set. On MovieLens 100K this leaves
    only ~98 evaluable users because users tend to rate in bursts, so estimates are noisy.
  * **Per-user cutoff:** each user's newest 20% of ratings form the test set (907 evaluable users). The
    trade-off is that, across users, train can contain events later in calendar time than another
    user's test events. This is the headline protocol because of its far larger evaluation population.
* **Hyperparameters are chosen on validation only** (ItemKNN k ∈ {20, 50, 100}; BPR regularization ∈
  {0.002, 0.01, 0.05}; hybrid weight ∈ {0.3, 0.5, 0.7}). Final models are refit on train+validation and
  scored once on the untouched test set.
* **Relevant item = rated ≥ 4 in the test period.** Items already rated in train are never recommended.
  Only users with at least one relevant test item and at least one train interaction are evaluated.
* **Metrics at K = 10:** Precision, Recall, NDCG, HitRate; plus *coverage* (share of the catalog ever
  recommended) and *novelty* (mean self-information of recommended items; higher = less mainstream).
  RMSE is reported separately for the rating models.
* **Significance:** every model is compared with Popularity using a paired bootstrap (2,000 resamples
  over users). A model is called better or worse only when the 95% CI excludes zero.

## Results

### Per-user chronological split (headline): 907 users

Split sizes: train 69,575 / validation 10,044 / test 20,381. Chosen on validation: ItemKNN k=50,
BPR regularization 0.002, hybrid ItemKNN weight 0.7.

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Novelty |
|---|---|---|---|---|---|---|
| Popularity | 0.0782 | 0.0805 | 0.0990 | 0.4487 | 0.0428 | 7.7419 |
| ItemKNN | 0.1212 | 0.1384 | 0.1599 | 0.5854 | 0.2568 | 8.6172 |
| BiasedMF (rating-ranked) | 0.0482 | 0.0436 | 0.0548 | 0.3098 | 0.1908 | 9.7869 |
| BPR-MF | 0.1090 | 0.1295 | 0.1474 | 0.5513 | 0.3859 | 8.7503 |
| **Hybrid (ItemKNN + BPR-MF)** | 0.1234 | 0.1417 | **0.1672** | 0.5943 | 0.2628 | 8.6024 |

Difference in NDCG@10 versus Popularity (paired bootstrap over users):

| Model | NDCG@10 | Diff vs Popularity | 95% CI | Distinguishable? |
|---|---|---|---|---|
| Popularity | 0.0990 | (baseline) | | |
| ItemKNN | 0.1599 | +0.0609 | [+0.0479, +0.0745] | yes (better) |
| BiasedMF | 0.0548 | -0.0442 | [-0.0543, -0.0341] | yes (worse) |
| BPR-MF | 0.1474 | +0.0484 | [+0.0356, +0.0605] | yes (better) |
| Hybrid | 0.1672 | +0.0682 | [+0.0541, +0.0818] | yes (better) |

Rating prediction (RMSE, lower is better): global mean 1.2079, user+item bias 1.0175, BiasedMF 0.9701.

### Global time cutoff: only 98 evaluable users

Split sizes: train 70,000 / validation 10,000 / test 20,000. Chosen on validation: ItemKNN k=20, BPR
regularization 0.01, hybrid ItemKNN weight 0.3.

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Novelty |
|---|---|---|---|---|---|---|
| Popularity | 0.1082 | 0.0830 | 0.1341 | 0.4592 | 0.0452 | 8.0143 |
| ItemKNN | 0.0765 | 0.0491 | 0.0919 | 0.3163 | 0.1587 | 8.7790 |
| BiasedMF (rating-ranked) | 0.0694 | 0.0527 | 0.0939 | 0.3367 | 0.1052 | 10.0528 |
| BPR-MF | 0.1194 | 0.0839 | 0.1498 | 0.4694 | 0.1980 | 8.8883 |
| Hybrid (ItemKNN + BPR-MF) | 0.1020 | 0.0725 | 0.1225 | 0.4082 | 0.1700 | 8.7567 |

| Model | NDCG@10 | Diff vs Popularity | 95% CI | Distinguishable? |
|---|---|---|---|---|
| Popularity | 0.1341 | (baseline) | | |
| ItemKNN | 0.0919 | -0.0422 | [-0.0797, -0.0051] | yes (worse) |
| BiasedMF | 0.0939 | -0.0402 | [-0.0819, +0.0001] | no (within noise) |
| BPR-MF | 0.1498 | +0.0157 | [-0.0207, +0.0509] | no (within noise) |
| Hybrid | 0.1225 | -0.0116 | [-0.0497, +0.0252] | no (within noise) |

Rating prediction (RMSE): global mean 1.1190, user+item bias 1.0322, BiasedMF 1.0228.

Raw outputs are in `results/` (`results_per_user.md/.json`, `results_global.md/.json`).

## Key takeaways

1. **Beating popularity is real but needs the right test.** With 907 users, the hybrid, ItemKNN and
   BPR-MF all beat popularity with confidence intervals well above zero. With ~98 users (global cutoff),
   BPR-MF and the hybrid cannot be distinguished from popularity and ItemKNN is significantly worse.
   Conclusions depend on the evaluation protocol, so both are reported, and the per-user split is the
   headline because of its far larger evaluation population. I initially ran only the global split,
   noticed the evaluable-user count was tiny, and added the per-user split and bootstrap intervals.
2. **Accurate ratings do not mean good rankings.** BiasedMF has the best RMSE (0.970) but is
   significantly worse than popularity at top-10 ranking. It is trained to predict ratings of items users
   already chose to rate, not to separate items a user would engage with from those they would not.
   BPR-MF optimizes ranking directly and scores far higher.
3. **Accuracy vs. diversity is visible in the numbers.** Popularity recommends 4.3% of the catalog;
   the hybrid 26.3%; BPR-MF 38.6% at somewhat lower NDCG than the hybrid (0.147 vs 0.167).
4. **The hybrid is comparable to ItemKNN, not clearly better.** Its NDCG@10 is slightly higher
   (0.167 vs 0.160), but this difference was not tested for significance, so no claim is made.

## Product thinking

* **Cold start (new user):** with no history, `GenreColdStart` ranks well-rated movies in the genres the
  user picks, using a Bayesian average so a movie with two 5-star ratings does not outrank a proven
  favorite. This component is included in the demo but was **not** quantitatively evaluated.
* **Cold start (new item):** collaborative models cannot place a movie nobody has rated; content features
  (genres here, plot/metadata embeddings in a fuller system) are the standard bridge.
* **Explainability:** each recommendation shows the user's own movies that contribute most to it, taken from
  item-item similarity. This explains the neighborhood signal, not BPR's latent factors.
* **Popularity bias:** accuracy-optimized models concentrate on already-popular items; coverage and
  novelty make this visible, and diversity-aware re-ranking is the natural next step.
* **How I would A/B test it:** primary metric = click-through or watch-through on recommended slots;
  guardrails = catalog coverage, session abandonment, p95 latency, share of long-tail recommendations;
  randomize by user and run at least 1-2 weeks to cover weekly seasonality and novelty effects.

## API and demo

* `GET /recommend/{user_id}?k=10&model=Hybrid`: personalized recommendations with explanations
* `GET /similar/{item_id}`: similar movies from item-item similarity
* `GET /cold-start?genres=Comedy&genres=Drama`: recommendations for a brand-new user
* Streamlit app (`python -m streamlit run app.py`) with tabs for existing users, new users, and similar movies.

Models are fitted once at startup on all available ratings, using the hyperparameters selected by
`run_experiments.py` when `results/results_per_user.json` exists.

## Limitations

* Offline metrics on an older explicit-ratings dataset are only a proxy for online behavior.
* Any rated item is treated as an implicit positive signal; richer feedback (watch time, skips) is not available.
* Dense NumPy matrices are fine at 100K ratings; production would use sparse matrices and approximate
  nearest-neighbor search for retrieval.
* Hyperparameter grids are small, and results come from a single random seed.
* The hybrid-vs-ItemKNN difference and the cold-start component were not tested for significance.
* The per-user split can place a user's test events earlier in calendar time than other users' training events.

## Future work

Significance testing between the top models, multiple seeds, diversity-aware re-ranking (MMR), a
content + collaborative two-tower model using plot/metadata embeddings, sequence-aware models, a
cold-start simulation (evaluate with only each user's first N ratings), and an online A/B test harness.

## Project layout

```
recsys/
  data.py        loading, global and per-user chronological splits
  metrics.py     RMSE, ranking metrics, coverage, novelty
  models.py      all recommenders
  service.py     builds the serving models (used by the API and demo)
tests/           hand-computed NDCG check, no-leakage checks for both splits
run_experiments.py   api.py   app.py
results/         generated result tables (.md and .json)
```

## Acknowledgements

F. M. Harper and J. A. Konstan. *The MovieLens Datasets: History and Context.* ACM Transactions on
Interactive Intelligent Systems, 2015. MovieLens data is provided by GroupLens and is not redistributed here.

## Author

Sayan Sekhar Ghosh
IIT (ISM) Dhanbad

