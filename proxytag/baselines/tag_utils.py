import torch
from proxytag.core.tags import build_item_tag_tensors


def load_and_pool_tags(tags_path, item2idx, item_id_col, device='cpu', max_tags=16):
    """
    Load tag parquet, encode via sentence-transformers, mean-pool to [n_items, emb_dim].

    Args:
        tags_path: Path to parquet with item_id and tags columns
        item2idx: Dict mapping raw item IDs to contiguous indices
        item_id_col: Name of the item ID column in the parquet
        device: Target device for the returned tensor
        max_tags: Max tags per item to embed before pooling

    Returns:
        pooled: torch.Tensor [n_items, emb_dim] - mean-pooled tag embeddings per item
    """
    item_tag_embs, item_tag_mask = build_item_tag_tensors(
        tags_path, item2idx, item_id_col=item_id_col, device='cpu', max_tags=max_tags
    )
    # Mean pool: [n_items, max_tags, emb_dim] → [n_items, emb_dim]
    mask_expanded = item_tag_mask.unsqueeze(-1).float()
    pooled = (item_tag_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
    return pooled.to(device)
