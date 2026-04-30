import os
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_id_mappings(train_df, val_df, test_df, user_id_col="userId", item_id_col="itemId"):
    """Map raw IDs -> contiguous indices across ALL splits (supports cold-start in val/test)."""
    # Build list of dataframes to concat, excluding None values
    dfs_for_users = [train_df[user_id_col]]
    dfs_for_items = [train_df[item_id_col]]

    if val_df is not None:
        dfs_for_users.append(val_df[user_id_col])
        dfs_for_items.append(val_df[item_id_col])

    if test_df is not None:
        dfs_for_users.append(test_df[user_id_col])
        dfs_for_items.append(test_df[item_id_col])

    all_users = pd.concat(dfs_for_users).unique()
    all_items = pd.concat(dfs_for_items).unique()

    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {m: i for i, m in enumerate(all_items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}

    def map_df(df):
        df = df.copy()
        df["uid"] = df[user_id_col].map(user2idx)
        df["iid"] = df[item_id_col].map(item2idx)

        # Check for unmapped IDs
        if df["uid"].isna().any() or df["iid"].isna().any():
            n_unmapped_users = df["uid"].isna().sum()
            n_unmapped_items = df["iid"].isna().sum()
            raise ValueError(f"Found unmapped IDs: {n_unmapped_users} users, {n_unmapped_items} items")

        df["uid"] = df["uid"].astype(np.int64)
        df["iid"] = df["iid"].astype(np.int64)
        return df

    train_mapped = map_df(train_df)
    val_mapped = map_df(val_df) if val_df is not None else None
    test_mapped = map_df(test_df) if test_df is not None else None

    return train_mapped, val_mapped, test_mapped, user2idx, idx2user, item2idx, idx2item


def build_user_history(train_df, val_df=None, test_df=None):
    """user_history[u] = set of all items user interacted with across train/val/test."""
    hist = defaultdict(set)
    dfs = [train_df]
    if val_df is not None:
        dfs.append(val_df)
    if test_df is not None:
        dfs.append(test_df)

    for df in dfs:
        for u, i in zip(df["uid"].values, df["iid"].values):
            hist[int(u)].add(int(i))
    return hist


class NegativeSampler:
    def __init__(self, n_items, user_history, seed=42):
        self.n_items = n_items
        self.user_history = user_history
        self.rng = np.random.default_rng(seed)

    def sample(self, user_id, n_neg):
        interacted = self.user_history.get(user_id, set())
        available_items = self.n_items - len(interacted)

        if available_items <= 0:
            return []

        # If requested negatives exceed available, return all available
        n_neg = min(n_neg, available_items)

        # sample with rejection, but vectorized
        negs = []
        seen = set()
        max_attempts = n_neg * 10  # prevent infinite loops
        attempts = 0

        while len(negs) < n_neg and attempts < max_attempts:
            candidates = self.rng.integers(0, self.n_items, size=n_neg * 2)
            for c in candidates:
                if c not in interacted and c not in seen:
                    negs.append(c)
                    seen.add(c)
                    if len(negs) == n_neg:
                        break
            attempts += 1

        return negs


class TrainDataset(Dataset):
    """
    For each positive (u, pos_i), returns a bundle of:
      users: [1 + n_neg]
      items: [1 + n_neg]
      labels:[1 + n_neg]
    """
    def __init__(self, train_df: pd.DataFrame, neg_sampler: NegativeSampler, n_neg: int = 4):
        self.df = train_df.reset_index(drop=True)
        self.neg_sampler = neg_sampler
        self.n_neg = n_neg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        u = int(row["uid"])
        pos_i = int(row["iid"])

        negs = self.neg_sampler.sample(u, self.n_neg)

        users = [u] * (1 + self.n_neg)
        items = [pos_i] + negs
        labels = [1.0] + [0.0] * self.n_neg

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )


def train_collate_fn(batch):
    users = torch.cat([b[0] for b in batch], dim=0)
    items = torch.cat([b[1] for b in batch], dim=0)
    labels = torch.cat([b[2] for b in batch], dim=0)
    return users, items, labels


class DummyDataset(Dataset):
    """Used only to trigger Lightning's validation loop."""
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(0)


class RecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        neg_sampler_train: NegativeSampler,
        batch_size: int = 512,
        n_neg_train: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.neg_sampler_train = neg_sampler_train
        self.batch_size = batch_size
        self.n_neg_train = n_neg_train
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TrainDataset(self.train_df, self.neg_sampler_train, self.n_neg_train)
        self.val_ds = DummyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,          # single-process, safest
            collate_fn=train_collate_fn,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        # dummy loader; we evaluate manually in the LightningModule
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0)
