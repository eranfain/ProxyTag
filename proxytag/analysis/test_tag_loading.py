#!/usr/bin/env python3
"""
Test if tags load correctly for different tag files.

This simulates what happens during model training when tags are loaded.
"""

import pandas as pd
import torch
from proxytag.core.tags import build_item_tag_tensors
from sentence_transformers import SentenceTransformer


def test_tag_file(tag_file, name, max_tags=36):
    """Test tag loading for one file."""
    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")

    # Load items data
    items_df = pd.read_parquet(tag_file)
    print(f"Loaded {len(items_df)} items")

    # Build tag tensors (simulating model initialization)
    print(f"\nBuilding tag tensors (max_tags={max_tags})...")

    # Load sentence transformer
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Build tensors
    item_tag_embs, item_tag_mask = build_item_tag_tensors(
        items_df=items_df,
        item_id_col='asin',
        tag_col='tags',
        st_model=st_model,
        max_tags=max_tags,
        overwrite=False,
    )

    print(f"\nTag tensor statistics:")
    print(f"  item_tag_embs shape: {item_tag_embs.shape}")
    print(f"  item_tag_mask shape: {item_tag_mask.shape}")

    # Check for items with no tags
    items_with_no_tags = (~item_tag_mask.any(dim=1)).sum().item()
    print(f"  Items with no tags: {items_with_no_tags} ({items_with_no_tags/len(items_df)*100:.1f}%)")

    # Check tag count distribution
    tag_counts = item_tag_mask.sum(dim=1)  # [num_items]
    print(f"  Tags per item - Mean: {tag_counts.float().mean():.1f}, "
          f"Median: {tag_counts.float().median():.1f}, "
          f"Max: {tag_counts.max().item()}")

    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {item_tag_embs.mean():.6f}")
    print(f"  Std: {item_tag_embs.std():.6f}")
    print(f"  Min: {item_tag_embs.min():.6f}")
    print(f"  Max: {item_tag_embs.max():.6f}")

    # Check for NaN or Inf
    has_nan = torch.isnan(item_tag_embs).any().item()
    has_inf = torch.isinf(item_tag_embs).any().item()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    # Sample a few items
    print(f"\nSample items:")
    for i in [0, 100, 1000]:
        n_tags = tag_counts[i].item()
        if n_tags > 0:
            emb_mean = item_tag_embs[i, :n_tags, :].mean().item()
            emb_std = item_tag_embs[i, :n_tags, :].std().item()
            print(f"  Item {i}: {n_tags} tags, emb mean={emb_mean:.4f}, std={emb_std:.4f}")
        else:
            print(f"  Item {i}: 0 tags")

    return item_tag_embs, item_tag_mask


def main():
    tag_files = {
        'kar (BROKEN)': 'data/amazon-books/tags/full/tags_kar.parquet',
        'llm_rec_orig (BROKEN)': 'data/amazon-books/tags/full/tags_llm_rec_orig.parquet',
        'single_use (BROKEN)': 'data/amazon-books/tags/full/tags_single_use_knowledge_recommend_content.parquet',
        'multi_use (WORKING)': 'data/amazon-books/tags/full/tags_multi_use_knowledge_recommend_content.parquet',
        'categories (WORKING)': 'data/amazon-books/tags/full/tags_categories.parquet',
    }

    for name, tag_file in tag_files.items():
        try:
            test_tag_file(tag_file, name)
        except Exception as e:
            print(f"\n❌ ERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("If all files loaded successfully with similar embedding statistics,")
    print("then tags are loading correctly and the bug must be elsewhere.")


if __name__ == "__main__":
    main()
