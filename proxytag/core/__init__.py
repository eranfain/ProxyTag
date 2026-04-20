"""Core modules: data loading, evaluation, model, tags, utilities."""

from proxytag.core.data import (
    load_df,
    build_id_mappings,
    build_user_history,
    NegativeSampler,
    RecDataModule,
)
from proxytag.core.eval import eval_split_fast, eval_split_by_bucket
from proxytag.core.model import HybridRecLightning
from proxytag.core.tags import build_item_tag_tensors
from proxytag.core.utils import build_item_popularity_percentiles
