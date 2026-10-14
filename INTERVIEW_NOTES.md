# Interview notes: 

**1. Why a time-based split?**
Random splits leak the future into training. In production you predict tomorrow from yesterday, so
the test set must be the newest data.

**2. Why did you tune on a validation set and not on test?**
Tuning on test makes the reported number optimistic (you've fit your choices to it). I picked
hyper-parameters on validation, refit on train+val, and touched test once.

**3. What is NDCG@K and why use it over accuracy?**
A top-K list is a ranking, so position matters: a hit at rank 1 is worth more than at rank 10.
NDCG sums hits discounted by 1/log2(rank+1) and divides by the best possible ordering (0..1).
RMSE only measures rating error and says nothing about whether the top of the list is good.

**4. Why does rating-optimised MF rank badly but BPR rank well?**
Predicting ratings rewards getting 1-5 right for items users *already chose to rate*. Ranking all
items needs to separate "would engage" from "wouldn't". BPR trains exactly that: for each user, an item
they interacted with should outscore a random one. (Check the BiasedMF vs. BPR rows in my results.)

**5. How does ItemKNN work, and what is shrinkage for?**
Build item vectors from who interacted with them, compute cosine similarity, score a candidate by summing
its similarity to the user's items. Shrinkage (adding a constant to the denominator) stops two items
co-occurring for 2 users from looking identical to items co-occurring for 200.

**6. Why z-score before blending models?**
Scores from different models live on different scales (counts vs. dot products). Normalising per user
makes weights meaningful; the weight was chosen on validation.

**7. How do you handle cold start?**
New user: genre picker + Bayesian-average ranking, or popularity. New item: content features
(genres) until it collects interactions. As interactions arrive, collaborative models take over.

**8. What's popularity bias and how did you measure it?**
Models trained on clicks over-recommend what's already popular, starving the long tail. I report
coverage and novelty next to accuracy to expose that trade-off.

**9. How would you scale this to millions of users?**
Sparse matrices, approximate nearest neighbours for retrieval, a two-stage retrieve-then-rank design,
offline training + cached embeddings, online A/B tests with guardrails.

**10. What are the weaknesses?**
Offline-only evaluation; implicit-feedback assumption; small tuning grid; ML-100K is tiny and old.
