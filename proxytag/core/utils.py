import numpy as np
import pandas as pd


def build_item_popularity_percentiles(
    train_df, n_items, n_bins=10
):
    """
    Returns:
      item_count: np.ndarray [n_items] int
      item_bucket: np.ndarray [n_items] int in [0..n_bins]
        - 0 means cold-start (count == 0)
        - 1..n_bins are interaction-percentile bins
          (bucket 1 = most popular items)
    """

    # counts per item from TRAIN only
    item_count = np.zeros(n_items, dtype=np.int64)
    vc = train_df["iid"].value_counts()
    item_count[vc.index.values.astype(int)] = vc.values.astype(np.int64)

    # initialize buckets
    item_bucket = np.zeros(n_items, dtype=np.int64)

    warm_mask = item_count > 0
    warm_items = np.nonzero(warm_mask)[0]
    warm_counts = item_count[warm_mask]

    if warm_counts.size == 0:
        return item_count, item_bucket  # all cold

    # sort warm items by popularity DESC
    order = np.argsort(-warm_counts)
    sorted_items = warm_items[order]
    sorted_counts = warm_counts[order]

    # cumulative interaction mass
    cum_counts = np.cumsum(sorted_counts)
    total_counts = cum_counts[-1]

    # interaction percentiles in (0, 1]
    interaction_pct = cum_counts / total_counts

    # map percentiles -> buckets [1..n_bins]
    # e.g. pct in (0, 0.01] -> bucket 1
    buckets = np.ceil(interaction_pct * n_bins).astype(np.int64)
    buckets = np.clip(buckets, 1, n_bins)

    # write back
    item_bucket[sorted_items] = buckets

    return item_count, item_bucket