"""Recommenders, from simplest to most sophisticated.

Every ranking model exposes  fit(train, n_users, n_items)  and
scores(users) -> (len(users), n_items) array (higher = better).
Rating models additionally expose predict(users, items).
All of it is plain NumPy so you can read every line.
"""
import numpy as np

from .data import GENRES


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _interaction_matrix(df, n_users, n_items, min_rating=None):
    R = np.zeros((n_users, n_items), dtype=np.float32)
    d = df if min_rating is None else df[df.rating >= min_rating]
    R[d.user.values, d.item.values] = 1.0
    return R


def _zscore_rows(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


# --------------------------------------------------------------------------
# Stage 1: baselines
# --------------------------------------------------------------------------
class Popularity:
    """Recommend the most-rated items to everyone. The bar to beat."""
    name = "Popularity"

    def fit(self, train, n_users, n_items):
        self.pop = np.bincount(train.item.values, minlength=n_items).astype(np.float32)
        return self

    def scores(self, users):
        return np.tile(self.pop, (len(users), 1))


class GlobalMean:
    name = "Global mean"

    def fit(self, train, n_users, n_items):
        self.mu = float(train.rating.mean())
        return self

    def predict(self, users, items):
        return np.full(len(users), self.mu)


class BiasBaseline:
    """rating ~ global mean + item bias + user bias (damped averages)."""
    name = "User+item bias"

    def __init__(self, reg_item=25, reg_user=10):
        self.reg_item, self.reg_user = reg_item, reg_user

    def fit(self, train, n_users, n_items):
        u, i, r = train.user.values, train.item.values, train.rating.values
        self.mu = r.mean()
        cnt_i = np.bincount(i, minlength=n_items)
        self.bi = np.bincount(i, weights=r - self.mu, minlength=n_items) / (cnt_i + self.reg_item)
        resid = r - self.mu - self.bi[i]
        cnt_u = np.bincount(u, minlength=n_users)
        self.bu = np.bincount(u, weights=resid, minlength=n_users) / (cnt_u + self.reg_user)
        return self

    def predict(self, users, items):
        return np.clip(self.mu + self.bu[users] + self.bi[items], 1, 5)


# --------------------------------------------------------------------------
# Stage 2: item-based collaborative filtering
# --------------------------------------------------------------------------
class ItemKNN:
    """'People who interacted with X also interacted with Y.'

    Item-item cosine similarity on the binary user-item matrix, with a
    shrinkage term (so two items co-rated by 2 users do not look 'identical')
    and only the top-k neighbours per item kept.
    """
    name = "ItemKNN"

    def __init__(self, k=50, shrink=10.0, min_rating=None):
        self.k, self.shrink, self.min_rating = k, shrink, min_rating

    def fit(self, train, n_users, n_items):
        R = _interaction_matrix(train, n_users, n_items, self.min_rating)
        co = R.T @ R                                   # co-occurrence counts
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink)
        np.fill_diagonal(sim, 0.0)
        if self.k < n_items:
            drop = np.argpartition(-sim, self.k, axis=1)[:, self.k:]
            np.put_along_axis(sim, drop, 0.0, axis=1)
        self.R, self.sim = R, sim.astype(np.float32)
        return self

    def scores(self, users):
        return self.R[users] @ self.sim.T

    # --- explainability -----------------------------------------------------
    def explain(self, user, item, top=3):
        """Items the user interacted with that most support recommending `item`."""
        contrib = self.R[user] * self.sim[item]
        order = np.argsort(-contrib)[:top]
        return [int(j) for j in order if contrib[j] > 0]

    def similar(self, item, k=10):
        order = np.argsort(-self.sim[item])[:k]
        return [(int(j), float(self.sim[item, j])) for j in order if self.sim[item, j] > 0]


# --------------------------------------------------------------------------
# Stage 3: matrix factorization
# --------------------------------------------------------------------------
class BiasedMF:
    """FunkSVD-style matrix factorization trained with SGD (rating prediction).

    rating(u, i) ~ mu + b_u + b_i + p_u . q_i      (p_u, q_i are small vectors)
    """
    name = "BiasedMF (rating-ranked)"

    def __init__(self, dim=32, lr=0.01, reg=0.05, epochs=30, batch=64, seed=0):
        self.dim, self.lr, self.reg = dim, lr, reg
        self.epochs, self.batch, self.seed = epochs, batch, seed
        self.history = []

    def fit(self, train, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        u, i = train.user.values, train.item.values
        r = train.rating.values.astype(np.float32)
        self.mu = float(r.mean())
        self.bu = np.zeros(n_users, np.float32)
        self.bi = np.zeros(n_items, np.float32)
        self.P = rng.normal(0, 0.05, (n_users, self.dim)).astype(np.float32)
        self.Q = rng.normal(0, 0.05, (n_items, self.dim)).astype(np.float32)
        lr, reg = self.lr, self.reg
        for _ in range(self.epochs):
            perm = rng.permutation(len(r))
            sq = 0.0
            for s in range(0, len(perm), self.batch):
                idx = perm[s:s + self.batch]
                ub, ib, rb = u[idx], i[idx], r[idx]
                pu, qi = self.P[ub], self.Q[ib]
                err = rb - (self.mu + self.bu[ub] + self.bi[ib] + (pu * qi).sum(1))
                sq += float((err ** 2).sum())
                np.add.at(self.bu, ub, lr * (err - reg * self.bu[ub]))
                np.add.at(self.bi, ib, lr * (err - reg * self.bi[ib]))
                np.add.at(self.P, ub, lr * (err[:, None] * qi - reg * pu))
                np.add.at(self.Q, ib, lr * (err[:, None] * pu - reg * qi))
            self.history.append(np.sqrt(sq / len(r)))
        return self

    def predict(self, users, items):
        p = self.mu + self.bu[users] + self.bi[items] + (self.P[users] * self.Q[items]).sum(1)
        return np.clip(p, 1, 5)

    def scores(self, users):
        return self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T


class BPRMF:
    """Matrix factorization trained for RANKING (Bayesian Personalized Ranking).

    Instead of predicting ratings, learn vectors so that for each user the
    items they interacted with score higher than random items they did not:
        maximize  log sigmoid( s(u, i+) - s(u, j-) )
    This matches what a recommender is actually asked to do (rank a list).
    """
    name = "BPR-MF"

    def __init__(self, dim=32, lr=0.05, reg=0.01, epochs=30, batch=64,
                 min_rating=None, seed=0):
        self.dim, self.lr, self.reg = dim, lr, reg
        self.epochs, self.batch, self.min_rating, self.seed = epochs, batch, min_rating, seed
        self.history = []

    def fit(self, train, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        d = train if self.min_rating is None else train[train.rating >= self.min_rating]
        pu_all, pi_all = d.user.values, d.item.values
        seen = np.zeros((n_users, n_items), dtype=bool)
        seen[train.user.values, train.item.values] = True

        self.P = rng.normal(0, 0.1, (n_users, self.dim)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.dim)).astype(np.float32)
        self.b = np.zeros(n_items, np.float32)
        lr, reg = self.lr, self.reg

        for _ in range(self.epochs):
            perm = rng.permutation(len(pu_all))
            loss = 0.0
            for s in range(0, len(perm), self.batch):
                idx = perm[s:s + self.batch]
                u, i = pu_all[idx], pi_all[idx]
                j = rng.integers(0, n_items, size=len(u))       # negative sample
                for _ in range(5):                               # avoid sampling seen items
                    bad = seen[u, j]
                    if not bad.any():
                        break
                    j[bad] = rng.integers(0, n_items, size=int(bad.sum()))
                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                x = (pu * (qi - qj)).sum(1) + self.b[i] - self.b[j]
                g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))   # = sigmoid(-x)
                loss += float(np.log1p(np.exp(-np.clip(x, -30, 30))).sum())
                np.add.at(self.P, u, lr * (g[:, None] * (qi - qj) - reg * pu))
                np.add.at(self.Q, i, lr * (g[:, None] * pu - reg * qi))
                np.add.at(self.Q, j, lr * (-g[:, None] * pu - reg * qj))
                np.add.at(self.b, i, lr * (g - reg * self.b[i]))
                np.add.at(self.b, j, lr * (-g - reg * self.b[j]))
            self.history.append(loss / len(pu_all))
        return self

    def scores(self, users):
        return self.P[users] @ self.Q.T + self.b[None, :]


# --------------------------------------------------------------------------
# Stage 5: hybrid + cold start
# --------------------------------------------------------------------------
class Blend:
    """Weighted blend of already-fitted models (scores are z-scored per user first)."""

    def __init__(self, models, weights, name="Hybrid"):
        self.models, self.weights, self.name = models, weights, name

    def scores(self, users):
        return sum(w * _zscore_rows(np.asarray(m.scores(users), dtype=np.float64))
                   for m, w in zip(self.models, self.weights))


class GenreColdStart:
    """Content-based fallback for brand-new users (no history at all).

    The user just picks genres they like. We rank movies in those genres by a
    Bayesian-average rating (ratings shrunk toward the global mean so that a
    movie with two 5-star ratings does not beat a proven classic).
    """

    def __init__(self, prior_strength=10, min_count=20):
        self.C, self.min_count = prior_strength, min_count

    def fit(self, train, items):
        n = len(items)
        cnt = np.bincount(train.item.values, minlength=n)
        total = np.bincount(train.item.values, weights=train.rating.values, minlength=n)
        m = train.rating.mean()
        self.cnt = cnt
        self.bayes = (self.C * m + total) / (self.C + cnt)
        self.G = items[GENRES].values.astype(np.float32)
        return self

    def recommend(self, genres, k=10, exclude=()):
        gi = [GENRES.index(g) for g in genres]
        if gi:
            match = self.G[:, gi].sum(1)
        else:
            match = np.ones(len(self.cnt))
        frac = match / max(len(gi), 1)
        for min_c in (self.min_count, 1):
            eligible = (match > 0) & (self.cnt >= min_c)
            if eligible.sum() >= k:
                break
        score = np.where(eligible, self.bayes + 0.25 * frac, -np.inf)
        if len(exclude):
            score[list(exclude)] = -np.inf
        order = np.argsort(-score)[:k]
        return [int(j) for j in order if np.isfinite(score[j])]
