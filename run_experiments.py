"""Run every experiment and write results/results_<split>.md + .json.

    python run_experiments.py                    # per-user split (recommended)
    python run_experiments.py --split global     # one global time cutoff
    python run_experiments.py --quick            # fewer epochs, smoke test

Protocol
  1. Chronological split 70% train / 10% validation / 20% test.
  2. Hyper-parameters are chosen on VALIDATION (models fit on train only).
  3. Final models are refit on train+validation and scored once on TEST.
  4. Every model is compared to Popularity with a paired bootstrap 95% CI.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from recsys.data import load, split_data
from recsys.metrics import evaluate_ranking, evaluate_rating
from recsys.models import (BPRMF, BiasBaseline, BiasedMF, Blend, GlobalMean,
                           ItemKNN, Popularity)

K = 10
ARRAY_KEYS = ("per_user_ndcg", "eval_users")


def dataset_stats(ratings, n_users, n_items):
    counts = ratings.item.value_counts().values
    top20 = counts[: int(0.2 * len(counts))].sum() / counts.sum()
    return {
        "users": int(n_users), "items": int(n_items), "ratings": int(len(ratings)),
        "sparsity_pct": round(float(100 * (1 - len(ratings) / (n_users * n_items))), 2),
        "share_of_ratings_on_top20pct_items": round(float(top20), 3),
    }


def ndcg_on(model, fit_df, eval_df, nu, ni):
    return evaluate_ranking(model, fit_df, eval_df, nu, ni, k=K)["ndcg"]


def bootstrap_vs_baseline(raw, baseline="Popularity", n_boot=2000, seed=0):
    """Paired bootstrap over users: mean NDCG and its difference vs the baseline."""
    rng = np.random.default_rng(seed)
    base = raw[baseline]["per_user_ndcg"]
    out = {}
    for name, res in raw.items():
        x = res["per_user_ndcg"]
        d = x - base
        idx = rng.integers(0, len(x), (n_boot, len(x)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        out[name] = {"ndcg": float(x.mean()), "diff": float(d.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi)}
    return out


def main(quick=False, data_dir="data", split="per_user"):
    t0 = time.time()
    epochs = 8 if quick else 30
    ratings, items = load(data_dir)
    nu, ni = int(ratings.user.max()) + 1, int(items.item.max()) + 1
    train, val, test = split_data(ratings, split)
    trainval = pd.concat([train, val])

    stats = dataset_stats(ratings, nu, ni)
    print("Dataset:", stats)
    print(f"Split = {split}   train={len(train)}  val={len(val)}  test={len(test)}")

    # ---- 1. hyper-parameter selection on validation ------------------------
    print("\nTuning on validation set (NDCG@10) ...")
    knn_scores = {}
    for k in (20, 50, 100):
        m = ItemKNN(k=k).fit(train, nu, ni)
        knn_scores[k] = ndcg_on(m, train, val, nu, ni)
        print(f"  ItemKNN k={k:<4} val NDCG@10 = {knn_scores[k]:.4f}")
    best_k = max(knn_scores, key=knn_scores.get)

    bpr_scores = {}
    for reg in (0.002, 0.01, 0.05):
        m = BPRMF(reg=reg, epochs=epochs).fit(train, nu, ni)
        bpr_scores[reg] = ndcg_on(m, train, val, nu, ni)
        print(f"  BPR-MF reg={reg:<6} val NDCG@10 = {bpr_scores[reg]:.4f}")
    best_reg = max(bpr_scores, key=bpr_scores.get)

    knn_v = ItemKNN(k=best_k).fit(train, nu, ni)
    bpr_v = BPRMF(reg=best_reg, epochs=epochs).fit(train, nu, ni)
    blend_scores = {}
    for w in (0.3, 0.5, 0.7):
        blend_scores[w] = ndcg_on(Blend([knn_v, bpr_v], [w, 1 - w]), train, val, nu, ni)
        print(f"  Hybrid w_knn={w}   val NDCG@10 = {blend_scores[w]:.4f}")
    best_w = max(blend_scores, key=blend_scores.get)
    print(f"Chosen: ItemKNN k={best_k}, BPR reg={best_reg}, hybrid w_knn={best_w}")

    # ---- 2. final fit on train+val, evaluate once on test -------------------
    print("\nRefitting on train+val and evaluating on TEST ...")
    pop = Popularity().fit(trainval, nu, ni)
    knn = ItemKNN(k=best_k).fit(trainval, nu, ni)
    bpr = BPRMF(reg=best_reg, epochs=epochs).fit(trainval, nu, ni)
    bmf = BiasedMF(epochs=epochs).fit(trainval, nu, ni)
    hybrid = Blend([knn, bpr], [best_w, 1 - best_w], name="Hybrid (ItemKNN + BPR-MF)")

    raw = {}
    for m in (pop, knn, bmf, bpr, hybrid):
        raw[m.name] = evaluate_ranking(m, trainval, test, nu, ni, k=K)
        print(f"  {m.name:<28} NDCG@10={raw[m.name]['ndcg']:.4f}")
    rank_rows = {n: {k: v for k, v in r.items() if k not in ARRAY_KEYS} for n, r in raw.items()}
    boot = bootstrap_vs_baseline(raw)

    rating_rows = {}
    for m in (GlobalMean().fit(trainval, nu, ni),
              BiasBaseline().fit(trainval, nu, ni), bmf):
        rating_rows[m.name] = evaluate_rating(m, test)

    # ---- 3. report ---------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    rank_df = pd.DataFrame(rank_rows).T.rename(columns={
        "precision": f"Precision@{K}", "recall": f"Recall@{K}", "ndcg": f"NDCG@{K}",
        "hit_rate": f"HitRate@{K}", "coverage": "Coverage", "novelty": "Novelty",
        "n_users_eval": "Users"})
    rating_df = pd.DataFrame({"RMSE": rating_rows})

    def md(df, digits=4):
        cols = ["Model"] + list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for name, row in df.iterrows():
            cells = [str(int(v)) if c == "Users" else f"{v:.{digits}f}"
                     for c, v in zip(df.columns, row)]
            lines.append("| " + " | ".join([name] + cells) + " |")
        return "\n".join(lines)

    ci_lines = ["| Model | NDCG@10 | Diff vs Popularity | 95% CI | Distinguishable from Popularity? |",
                "|---|---|---|---|---|"]
    for name, b in boot.items():
        if name == "Popularity":
            ci_lines.append(f"| {name} | {b['ndcg']:.4f} | (baseline) | | |")
            continue
        sig = "yes (better)" if b["ci_lo"] > 0 else "yes (worse)" if b["ci_hi"] < 0 else "no (within noise)"
        ci_lines.append(f"| {name} | {b['ndcg']:.4f} | {b['diff']:+.4f} | "
                        f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {sig} |")

    split_desc = {"global": "one global time cutoff (newest 20% of all ratings)",
                  "per_user": "per-user cutoff (each user's newest 20% of ratings)"}[split]
    report = (
        f"# Results - split: {split_desc}\n\n"
        f"Dataset: {json.dumps(stats)}\n\n"
        f"Chosen hyper-parameters (on validation): ItemKNN k={best_k}, "
        f"BPR reg={best_reg}, hybrid ItemKNN weight={best_w}\n\n"
        f"## Top-{K} ranking (relevant = rated >= 4 in the test period)\n\n"
        + md(rank_df) + "\n\n"
        f"## Is each model really different from Popularity? (paired bootstrap over users)\n\n"
        + "\n".join(ci_lines) + "\n\n"
        "## Rating prediction (RMSE, lower is better)\n\n" + md(rating_df) + "\n"
    )
    with open(f"results/results_{split}.md", "w") as f:
        f.write(report)
    with open(f"results/results_{split}.json", "w") as f:
        json.dump({"split": split, "stats": stats, "ranking": rank_rows, "bootstrap": boot,
                   "rating": rating_rows,
                   "chosen": {"knn_k": best_k, "bpr_reg": best_reg, "hybrid_w": best_w}},
                  f, indent=2)
    print("\n" + report)
    print(f"Done in {time.time() - t0:.0f}s -> results/results_{split}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", choices=["per_user", "global"], default="per_user")
    a = ap.parse_args()
    main(quick=a.quick, data_dir=a.data_dir, split=a.split)
