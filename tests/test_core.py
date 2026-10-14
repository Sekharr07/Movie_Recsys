import numpy as np
import pandas as pd
import pytest

from recsys.data import time_split
from recsys.metrics import evaluate_ranking


class FixedScores:
    def __init__(self, row):
        self.row = np.array(row, dtype=float)

    def scores(self, users):
        return np.tile(self.row, (len(users), 1))


def test_ranking_metrics_hand_computed():
    # One user, 5 items. Item 0 is in train (must be excluded from the list).
    # Relevant test items: 1 and 3.  Model ranks: 1, 2 (top-2 after masking item 0).
    train = pd.DataFrame({"user": [0], "item": [0], "rating": [5], "ts": [1]})
    test = pd.DataFrame({"user": [0, 0], "item": [1, 3], "rating": [5, 5], "ts": [2, 2]})
    m = FixedScores([10, 9, 8, 1, 0])
    r = evaluate_ranking(m, train, test, n_users=1, n_items=5, k=2)
    idcg = 1 + 1 / np.log2(3)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["ndcg"] == pytest.approx(1 / idcg)
    assert r["hit_rate"] == 1.0
    assert r["coverage"] == pytest.approx(2 / 5)


def test_time_split_has_no_leakage():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"user": rng.integers(0, 10, 500), "item": rng.integers(0, 50, 500),
                       "rating": rng.integers(1, 6, 500), "ts": rng.integers(0, 10_000, 500)})
    tr, va, te = time_split(df)
    assert tr.ts.max() <= va.ts.min() <= va.ts.max() <= te.ts.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_per_user_split_is_chronological_per_user():
    from recsys.data import per_user_time_split
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"user": np.repeat(np.arange(20), 30), "item": rng.integers(0, 100, 600),
                       "rating": rng.integers(1, 6, 600), "ts": rng.integers(0, 10_000, 600)})
    tr, va, te = per_user_time_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    for u in range(20):
        a, b, c = (x[x.user == u] for x in (tr, va, te))
        assert len(a) > 0 and len(b) > 0 and len(c) > 0
        assert a.ts.max() <= b.ts.min() <= b.ts.max() <= c.ts.min()
