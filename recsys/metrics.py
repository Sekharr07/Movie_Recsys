"""Evaluation: rating accuracy (RMSE) and top-K ranking quality."""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate_rating(model, test):
    preds = model.predict(test.user.values, test.item.values)
    return rmse(test.rating.values, preds)


def evaluate_ranking(model, train, test, n_users, n_items, k=10,
                     rel_threshold=4, chunk=256):
    """Top-K ranking metrics.

    * An item is *relevant* for a user if they rated it >= rel_threshold in the
      test period.
    * Items the user already rated in train are never recommended.
    * Only users with >=1 relevant test item and >=1 train interaction count.

    Returns precision/recall/NDCG/hit-rate@K plus two "beyond accuracy"
    metrics: catalog coverage and novelty (mean self-information of the
    recommended items; higher = less mainstream).
    """
    seen = np.zeros((n_users, n_items), dtype=bool)
    seen[train.user.values, train.item.values] = True

    rel = test[test.rating >= rel_threshold]
    relm = np.zeros((n_users, n_items), dtype=bool)
    relm[rel.user.values, rel.item.values] = True
    n_rel = relm.sum(1)

    users = np.where((n_rel > 0) & seen.any(1))[0]

    pop = np.bincount(train.item.values, minlength=n_items) + 1.0
    self_info = -np.log2(pop / pop.sum())

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal_cum = np.cumsum(discounts)

    prec, rec, ndcg, hit, nov = [], [], [], [], []
    recommended = np.zeros(n_items, dtype=bool)

    for start in range(0, len(users), chunk):
        us = users[start:start + chunk]
        s = np.array(model.scores(us), dtype=np.float64)
        s[seen[us]] = -np.inf
        top = np.argpartition(-s, k, axis=1)[:, :k]
        top_scores = np.take_along_axis(s, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)

        is_hit = relm[us[:, None], top]
        hits = is_hit.sum(1)
        dcg = (is_hit * discounts).sum(1)
        idcg = ideal_cum[np.minimum(n_rel[us], k) - 1]

        prec.append(hits / k)
        rec.append(hits / n_rel[us])
        ndcg.append(dcg / idcg)
        hit.append(hits > 0)
        nov.append(self_info[top].mean(1))
        recommended[np.unique(top)] = True

    cat = np.concatenate
    return {
        "precision": float(cat(prec).mean()),
        "recall": float(cat(rec).mean()),
        "ndcg": float(cat(ndcg).mean()),
        "hit_rate": float(cat(hit).mean()),
        "coverage": float(recommended.mean()),
        "novelty": float(cat(nov).mean()),
        "n_users_eval": int(len(users)),
        # kept for significance testing (stripped before saving to JSON)
        "per_user_ndcg": cat(ndcg),
        "eval_users": users,
    }
