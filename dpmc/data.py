"""Dataset loaders, cached evaluation blocks, splits and the error metric.

MovieLens-25M: ``ratings.csv`` from https://files.grouplens.org/datasets/movielens/ml-25m.zip,
half-star ratings on ``[0.5, 5.0]`` (10 levels, ``Delta = 4.5``).
Amazon Video Games: the 5-core rating-only file of Amazon Reviews 2023,
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz,
integer stars on ``[1, 5]`` (5 levels, ``Delta = 4``).

Goodreads comics and graphic novels: ``goodreads_interactions_comics_graphic.json.gz`` of
the UCSD Book Graph (Wan and McAuley, RecSys 2018),
https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_comics_graphic.json.gz,
explicit ratings on ``[1, 5]`` (a rating of 0 means the book was shelved but not rated
and is dropped).

A block keeps the ``m`` most active users and then the ``n`` most active items among
them. The ``ml-25m-rand`` block instead draws ``m`` users uniformly at random (seeded by
``data_seed``) from all MovieLens users and then keeps the ``n`` most active items among
them, so it contains the light users that the top-activity blocks exclude. Blocks are
cached as ``cache_<dataset>_<m>x<n>_s<seed>.npz``.
"""
import json
import os

import numpy as np


def load_ml25m(data_dir, m, n, seed):
    """Top-``m`` users by activity, then top-``n`` items, from MovieLens-25M."""
    import pandas as pd
    path = os.path.join(data_dir, "ml-25m", "ratings.csv")
    if not os.path.exists(path):
        alt = os.path.join(data_dir, "ratings.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"ratings.csv not found under {data_dir}")
    print(f"[data] reading {path}", flush=True)
    df = pd.read_csv(path, usecols=["userId", "movieId", "rating"])
    r_min, r_max = 0.5, 5.0
    top_users = df["userId"].value_counts().index[:m]
    df = df[df["userId"].isin(top_users)]
    top_items = df["movieId"].value_counts().index[:n]
    df = df[df["movieId"].isin(top_items)]
    u_index = {u: i for i, u in enumerate(df["userId"].unique())}
    i_index = {v: j for j, v in enumerate(df["movieId"].unique())}
    rows = df["userId"].map(u_index).to_numpy()
    cols = df["movieId"].map(i_index).to_numpy()
    vals = df["rating"].to_numpy(dtype=float)
    shape = (len(u_index), len(i_index))
    dens = vals.shape[0] / (shape[0] * shape[1])
    print(f"[data] matrix {shape}, |Omega|={vals.shape[0]}, density={dens:.4f}", flush=True)
    return rows, cols, vals, shape, r_min, r_max


def load_amazon(data_dir, m, n, seed):
    """Top-``m`` users by activity, then top-``n`` items, from Amazon Video Games (5-core)."""
    import pandas as pd
    path = os.path.join(data_dir, "Video_Games.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video_Games.csv not found under {data_dir}")
    print(f"[data] reading {path}", flush=True)
    df = pd.read_csv(path, usecols=["user_id", "parent_asin", "rating"])
    r_min, r_max = 1.0, 5.0
    top_users = df["user_id"].value_counts().index[:m]
    df = df[df["user_id"].isin(top_users)]
    top_items = df["parent_asin"].value_counts().index[:n]
    df = df[df["parent_asin"].isin(top_items)]
    u_index = {u: i for i, u in enumerate(df["user_id"].unique())}
    i_index = {it: j for j, it in enumerate(df["parent_asin"].unique())}
    rows = df["user_id"].map(u_index).to_numpy()
    cols = df["parent_asin"].map(i_index).to_numpy()
    vals = df["rating"].to_numpy(dtype=float)
    shape = (len(u_index), len(i_index))
    print(f"[data] matrix {shape}, |Omega|={vals.shape[0]}, "
          f"density={vals.shape[0]/(shape[0]*shape[1]):.4f}", flush=True)
    return rows, cols, vals, shape, r_min, r_max


def _index(df, ucol, icol, r_min, r_max):
    """Contiguous user and item indices of a filtered frame."""
    u_index = {u: i for i, u in enumerate(df[ucol].unique())}
    i_index = {it: j for j, it in enumerate(df[icol].unique())}
    rows = df[ucol].map(u_index).to_numpy()
    cols = df[icol].map(i_index).to_numpy()
    vals = df["rating"].to_numpy(dtype=float)
    shape = (len(u_index), len(i_index))
    print(f"[data] matrix {shape}, |Omega|={vals.shape[0]}, "
          f"density={vals.shape[0]/(shape[0]*shape[1]):.4f}", flush=True)
    return rows, cols, vals, shape, r_min, r_max


def load_ml25m_random(data_dir, m, n, seed):
    """``m`` MovieLens-25M users drawn uniformly at random, then their top-``n`` items."""
    import pandas as pd
    path = os.path.join(data_dir, "ml-25m", "ratings.csv")
    if not os.path.exists(path):
        path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(path, usecols=["userId", "movieId", "rating"])
    users = np.sort(df["userId"].unique())
    pick = np.random.default_rng(seed).choice(users, size=m, replace=False)
    df = df[df["userId"].isin(pick)]
    top_items = df["movieId"].value_counts().index[:n]
    df = df[df["movieId"].isin(top_items)]
    return _index(df, "userId", "movieId", 0.5, 5.0)


def load_goodreads(data_dir, m, n, seed):
    """Top-``m`` users, then top-``n`` books, from the Goodreads comics interactions."""
    import gzip
    import pandas as pd
    path = os.path.join(data_dir, "goodreads_interactions_comics_graphic.json.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    print(f"[data] reading {path}", flush=True)
    us, bs, rs = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            r = int(d["rating"])
            if r > 0:
                us.append(d["user_id"]); bs.append(d["book_id"]); rs.append(r)
    df = pd.DataFrame({"user_id": us, "book_id": bs, "rating": rs})
    top_users = df["user_id"].value_counts().index[:m]
    df = df[df["user_id"].isin(top_users)]
    top_items = df["book_id"].value_counts().index[:n]
    df = df[df["book_id"].isin(top_items)]
    return _index(df, "user_id", "book_id", 1.0, 5.0)


LOADERS = {"ml-25m": load_ml25m, "ml-25m-rand": load_ml25m_random,
           "amazon": load_amazon, "goodreads": load_goodreads}
DATASETS = tuple(LOADERS)


def load_cached(dataset, data_dir, m, n, data_seed, cache_dir):
    """Load a block from the cache, building it from the raw file on first use."""
    key = "cache_%s_%dx%d_s%d.npz" % (dataset, m, n, data_seed)
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        z = np.load(path)
        print("[data] cache hit %s" % path, flush=True)
        shape = tuple(int(x) for x in z["shape"])
        return z["rows"], z["cols"], z["vals"], shape, float(z["r_min"]), float(z["r_max"])
    load = LOADERS[dataset]
    rows, cols, vals, shape, r_min, r_max = load(data_dir, m, n, data_seed)
    np.savez_compressed(path, rows=rows, cols=cols, vals=vals,
                        shape=np.asarray(shape), r_min=r_min, r_max=r_max)
    print("[data] cache write %s" % path, flush=True)
    return rows, cols, vals, tuple(int(x) for x in shape), float(r_min), float(r_max)


def grid_step(dataset):
    """Spacing of the public rating grid."""
    return 0.5 if dataset.startswith("ml-25m") else 1.0


def train_test_split(n_obs, test_frac, seed):
    """Seeded split of observation indices into ``(train, test)``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    n_test = int(round(test_frac * n_obs))
    return perm[n_test:], perm[:n_test]


def three_way_split(n_obs, val_frac, test_frac, seed):
    """Seeded split into ``(train, validation, test)``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    n_test = int(round(test_frac * n_obs))
    n_val = int(round(val_frac * n_obs))
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def rmse(pred, true):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def ranking_metrics(score, train_rows, train_cols, test_rows, test_cols, test_vals,
                    k=10, threshold=4.0):
    """Recall@k and NDCG@k (binary relevance) of the score matrix, averaged over users.

    For each user every item outside the user's training ratings is ranked by ``score``;
    the relevant items are the user's held-out ratings at or above ``threshold``. Users
    without a relevant held-out item are skipped. Returns ``(recall, ndcg, n_users)``.
    """
    S = np.array(score, dtype=float, copy=True)
    S[train_rows, train_cols] = -np.inf
    rel = test_vals >= threshold
    users = np.unique(test_rows[rel])
    if users.size == 0:
        return float("nan"), float("nan"), 0
    top = np.argpartition(-S[users], kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(S[users], top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    relset = {}
    for u, i in zip(test_rows[rel], test_cols[rel]):
        relset.setdefault(int(u), set()).add(int(i))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec, ndcg = [], []
    for r, u in enumerate(users):
        R = relset[int(u)]
        hit = np.array([int(i) in R for i in top[r]], float)
        rec.append(hit.sum() / min(k, len(R)))
        ndcg.append(float(hit @ disc) / float(disc[:min(k, len(R))].sum()))
    return float(np.mean(rec)), float(np.mean(ndcg)), int(users.size)


def save(out_path, payload):
    """Write JSON atomically so an interrupted run never leaves a truncated file."""
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
