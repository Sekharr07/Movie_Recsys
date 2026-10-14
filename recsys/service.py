"""Builds the production recommender (fit on ALL data) shared by the API and the demo app."""
import json
import os

import numpy as np

from .data import GENRES, load
from .models import BPRMF, Blend, GenreColdStart, ItemKNN, Popularity


class RecService:
    def __init__(self, data_dir=None, results_path="results/results_per_user.json", epochs=30):
        data_dir = data_dir or os.environ.get("RECSYS_DATA_DIR", "data")
        self.ratings, self.items = load(data_dir)
        self.nu = int(self.ratings.user.max()) + 1
        self.ni = int(self.items.item.max()) + 1
        # Use the hyper-parameters chosen by run_experiments.py when available.
        chosen = {"knn_k": 100, "bpr_reg": 0.01, "hybrid_w": 0.7}
        if os.path.exists(results_path):
            chosen = json.load(open(results_path))["chosen"]
        self.pop = Popularity().fit(self.ratings, self.nu, self.ni)
        self.knn = ItemKNN(k=chosen["knn_k"]).fit(self.ratings, self.nu, self.ni)
        self.bpr = BPRMF(reg=chosen["bpr_reg"], epochs=epochs).fit(self.ratings, self.nu, self.ni)
        w = chosen["hybrid_w"]
        self.hybrid = Blend([self.knn, self.bpr], [w, 1 - w])
        self.cold = GenreColdStart().fit(self.ratings, self.items)
        self.titles = self.items.title.values
        self.seen = self.knn.R > 0
        self.genres = GENRES
        self.models = {"Hybrid": self.hybrid, "ItemKNN": self.knn,
                       "BPR-MF": self.bpr, "Popularity": self.pop}

    def title(self, i):
        return str(self.titles[i])

    def recommend(self, user, k=10, model="Hybrid"):
        s = np.array(self.models[model].scores(np.array([user]))[0], dtype=np.float64)
        s[self.seen[user]] = -np.inf
        top = np.argsort(-s)[:k]
        out = []
        for i in top:
            why = [self.title(j) for j in self.knn.explain(user, int(i), top=2)]
            out.append({"item": int(i), "title": self.title(i), "score": float(s[i]),
                        "because_you_liked": why})
        return out

    def history(self, user, n=8):
        r = self.ratings[self.ratings.user == user].sort_values(["rating", "ts"], ascending=False)
        return [{"item": int(i), "title": self.title(i), "rating": int(x)}
                for i, x in zip(r.item.values[:n], r.rating.values[:n])]

    def similar(self, item, k=10):
        return [{"item": j, "title": self.title(j), "similarity": round(s, 3)}
                for j, s in self.knn.similar(item, k)]

    def cold_start(self, genres, k=10):
        return [{"item": i, "title": self.title(i)} for i in self.cold.recommend(genres, k)]
