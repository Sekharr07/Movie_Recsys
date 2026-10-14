"""Data loading and time-based splitting for MovieLens 100K."""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download(data_dir="data"):
    """Download + unzip MovieLens 100K if it is not already on disk."""
    folder = os.path.join(data_dir, "ml-100k")
    if os.path.exists(os.path.join(folder, "u.data")):
        return folder
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "ml-100k.zip")
    print("Downloading MovieLens 100K ...")
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_dir)
    return folder


def load(data_dir="data"):
    """Return (ratings, items). User/item ids are shifted to be 0-based."""
    folder = download(data_dir)
    ratings = pd.read_csv(
        os.path.join(folder, "u.data"), sep="\t",
        names=["user", "item", "rating", "ts"],
    )
    cols = ["item", "title", "release", "video_release", "url"] + GENRES
    items = pd.read_csv(
        os.path.join(folder, "u.item"), sep="|", names=cols, encoding="latin-1"
    )
    ratings["user"] -= 1
    ratings["item"] -= 1
    items["item"] -= 1
    items = items.sort_values("item").reset_index(drop=True)
    return ratings, items


def time_split(ratings, val_frac=0.1, test_frac=0.2):
    """Chronological split: oldest -> train, then validation, newest -> test.

    A random split would let the model 'see the future' (e.g. train on a
    user's ratings from May and test on April). Production systems always
    predict forward in time, so we evaluate the same way.
    """
    r = ratings.sort_values("ts", kind="mergesort").reset_index(drop=True)
    n = len(r)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    a, b = n - n_test - n_val, n - n_test
    return r.iloc[:a], r.iloc[a:b], r.iloc[b:]


def per_user_time_split(ratings, val_frac=0.1, test_frac=0.2):
    """Per-user chronological split: each user's NEWEST ratings are held out.

    Why: with one global time cutoff only a few users have ratings after the
    cutoff (MovieLens users rate in bursts), so the test set shrinks to ~100
    users and metrics become very noisy. Here every user contributes to
    train/val/test, and within each user nothing from the future is used to
    predict the past. Trade-off: across users, train can contain events that
    happen later (in calendar time) than some users' test events.
    """
    r = ratings.sort_values(["user", "ts"], kind="mergesort").reset_index(drop=True)
    pos = (r.groupby("user").cumcount() + 1) / r.groupby("user")["item"].transform("size")
    is_test = pos > 1 - test_frac
    is_val = (pos > 1 - test_frac - val_frac) & ~is_test
    return r[~is_test & ~is_val], r[is_val], r[is_test]


def split_data(ratings, mode="per_user", val_frac=0.1, test_frac=0.2):
    if mode == "global":
        return time_split(ratings, val_frac, test_frac)
    if mode == "per_user":
        return per_user_time_split(ratings, val_frac, test_frac)
    raise ValueError(f"unknown split mode: {mode}")
