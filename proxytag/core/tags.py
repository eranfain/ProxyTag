import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_item_tag_tensors(
    items_df_path: str,
    item2idx: dict,
    model_name: str = "all-MiniLM-L6-v2",
    max_tags: int = 16,
    device: str = "cpu",
    normalize: bool = True,
    overwrite: bool = False,
    item_id_col: str = "asin",
):
    """
    Loads items parquet, and either:
      - loads cached 'embs' + 'emb_mask', OR
      - computes embeddings from 'tags' and persists them.

    Returns:
      item_tag_embs: torch.float32 [num_items, max_tags, emb_dim]
      item_tag_mask: torch.bool    [num_items, max_tags]
    """
    items_df = pd.read_parquet(items_df_path)

    has_cache = (not overwrite) and ("embs" in items_df.columns) and ("emb_mask" in items_df.columns)

    # We always need an embedding dimension
    # If cached exists, infer it from first non-empty row; else infer from model
    if has_cache:
        print("Loading cached tag embeddings from parquet")

        emb_dim = None
        for v in items_df["embs"].values:
            if v is not None and len(v) > 0:
                # infer from first vector
                first = v[0]
                if isinstance(first, (list, tuple, np.ndarray)) and len(first) == 1 and isinstance(first[0], (list, tuple, np.ndarray)):
                    first = first[0]
                emb_dim = int(np.asarray(first).shape[0])
                break

        if emb_dim is None:
            # cache exists but empty -> fallback to model dim
            st_model = SentenceTransformer(model_name, device="cpu")
            emb_dim = st_model.get_sentence_embedding_dimension()

        num_items = len(item2idx)
        item_tag_embs = torch.zeros((num_items, max_tags, emb_dim), dtype=torch.float32)
        item_tag_mask = torch.zeros((num_items, max_tags), dtype=torch.bool)

        # fast lookup by item ID -> row index (keep native type: int or string)
        itemid_to_row = {iid: i for i, iid in enumerate(items_df[item_id_col].values)}

        for item_id, item_idx in item2idx.items():
            ridx = itemid_to_row.get(item_id, None)
            if ridx is None:
                continue

            row_embs = items_df.iloc[ridx]["embs"]
            if row_embs is None or len(row_embs) == 0:
                continue

            arr = np.stack(row_embs).astype(np.float32)
            t = min(arr.shape[0], max_tags)
            if t > 0:
                item_tag_embs[item_idx, :t] = torch.from_numpy(arr[:t])
                item_tag_mask[item_idx, :t] = True

        if normalize:
            # Normalize non-empty vectors (keeps zeros as zeros)
            flat = item_tag_embs[item_tag_mask]
            if flat.numel() > 0:
                item_tag_embs[item_tag_mask] = F.normalize(flat, dim=-1)

        return item_tag_embs.to(device), item_tag_mask.to(device)

    # ---------------- Compute and persist ----------------
    print("Computing tag embeddings (cache not found in parquet)")
    if "tags" not in items_df.columns:
        raise ValueError("items parquet must contain a 'tags' column when cache is missing.")

    # Use ST model for encoding; do it on CPU unless you explicitly want GPU
    st_device = device if device.startswith("cuda") else "cpu"
    st_model = SentenceTransformer(model_name, device=st_device)
    emb_dim = st_model.get_sentence_embedding_dimension()

    num_items = len(item2idx)
    item_tag_embs = torch.zeros((num_items, max_tags, emb_dim), dtype=torch.float32)
    item_tag_mask = torch.zeros((num_items, max_tags), dtype=torch.bool)

    # Prepare columns to persist (variable-length lists)
    embs_col = [None] * len(items_df)
    mask_col = [None] * len(items_df)

    itemid_to_row = {iid: i for i, iid in enumerate(items_df[item_id_col].values)}
    itemid_to_tags = dict(zip(items_df[item_id_col].values, items_df["tags"].fillna("").values))

    for item_id, item_idx in tqdm(item2idx.items(), desc="Encoding tags"):
        ridx = itemid_to_row.get(item_id, None)
        if ridx is None:
            continue

        tag_str = itemid_to_tags.get(item_id, "")
        if not isinstance(tag_str, str) or len(tag_str.strip()) == 0:
            embs_col[ridx] = []
            mask_col[ridx] = []
            continue

        tags = [t.strip() for t in tag_str.split("|") if t.strip()]
        tags = tags[:max_tags]
        if not tags:
            embs_col[ridx] = []
            mask_col[ridx] = []
            continue

        with torch.no_grad():
            vecs = st_model.encode(
                tags,
                batch_size=min(64, len(tags)),
                convert_to_numpy=True,
                normalize_embeddings=False,  # we normalize ourselves
                show_progress_bar=False,
            )

        vecs = np.asarray(vecs, dtype=np.float32)
        if normalize and vecs.size > 0:
            # L2 normalize rows
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms

        t = min(vecs.shape[0], max_tags)
        if t > 0:
            item_tag_embs[item_idx, :t] = torch.from_numpy(vecs[:t])
            item_tag_mask[item_idx, :t] = True

        embs_col[ridx] = vecs[:t].tolist()
        mask_col[ridx] = [True] * t

    items_df["embs"] = embs_col
    items_df["emb_mask"] = mask_col
    items_df.to_parquet(items_df_path, index=False)
    print(f"Saved embeddings cache to: {items_df_path}")

    return item_tag_embs.to(device), item_tag_mask.to(device)
