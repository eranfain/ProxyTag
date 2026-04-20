import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from proxytag.core.eval import eval_split_fast


class HybridRecLightning(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        embedding_reg: float = 0,
        lr: float = 1e-4,
        use_item_id: bool = True,
        use_tags: bool = True,
        n_neg_eval: int = 100,
        K: int = 10,
        seed: int = 42,
        tag_encoding_type: str = "cross_encoder",
        max_eval_samples: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["item_tag_embs", "item_tag_mask"])

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)

        # Use consistent initialization for both user and item embeddings
        # std = 1/sqrt(dim) is standard for embeddings
        init_std = 1.0 / (hidden_dim ** 0.5)
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)

        self.item_tag_embs = None
        self.item_tag_mask = None

        self.hidden_dim = hidden_dim
        self.tag_input_dim = 384  # MiniLM
        self.embedding_reg = embedding_reg
        self.tag_encoding_type = tag_encoding_type
        
        # Project tag tokens into CF space
        self.tag_proj = nn.Linear(self.tag_input_dim, self.hidden_dim)
        
        # User-conditioned attention (all in 128-dim space)
        self.tag_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Post-attention/pooling projection - selects correct dimension based on encoding type
        if self.tag_encoding_type == "cross_encoder":
            # Cross-encoder: attention output is hidden_dim -> hidden_dim
            self.tag_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            # Mean pooling: pooled tag embeddings are tag_input_dim -> hidden_dim
            self.tag_fc = nn.Linear(self.tag_input_dim, self.hidden_dim)

        # Gate - initialize based on whether we're using item IDs
        if use_item_id:
            # Hybrid mode: start with items contributing ~88%, tags ~12%
            # sigmoid(-2.0) ≈ 0.12, so: 0.88*item + tags
            self.tag_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            # Tags-only mode: start with tags contributing ~50% (balanced)
            self.tag_gate = nn.Parameter(torch.tensor(0.0))
        
        self.lr = lr
        self.use_item_id = use_item_id
        self.use_tags = use_tags

    def set_item_tag_tensors(self, item_tag_embs, item_tag_mask):
        # Replace any NaN values in embeddings with zeros
        if torch.isnan(item_tag_embs).any():
            print(f"⚠️ WARNING: Found NaN in item_tag_embs, replacing with zeros")
            item_tag_embs = torch.where(torch.isnan(item_tag_embs), torch.zeros_like(item_tag_embs), item_tag_embs)

        self.item_tag_embs = item_tag_embs
        self.item_tag_mask = item_tag_mask

    def encode_tags(self, item_ids: torch.Tensor) -> torch.Tensor:
        tag_embs = self.item_tag_embs[item_ids.cpu()]   # [B,T,D]
        mask = self.item_tag_mask[item_ids.cpu()]       # [B,T]
    
        tag_embs = tag_embs.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
    
        m = mask.unsqueeze(-1).float()
        pooled = (tag_embs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    
        out = self.tag_fc(pooled)
        return out
        
    def _log_embedding_norms(self, item_vec, tag_vec):
        # Log only once per epoch to avoid noise
        if self.global_step == 0 or self.global_step % 1000 == 0:
            with torch.no_grad():
                item_norm = item_vec.norm(dim=1).mean()
                tag_norm = tag_vec.norm(dim=1).mean()
                alpha = torch.sigmoid(self.tag_gate)
    
            # Lightning logging (goes to TensorBoard)
            print("Item norm: ", item_norm)
            print("Tag norm: ", tag_norm)
            print("Alpha: ", alpha)

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        user_vec = self.user_emb(users)                  # [B, 128]
    
        item_vec = torch.zeros_like(user_vec)
        if self.use_item_id:
            item_vec = self.item_emb(items)

        tags_vec = torch.zeros_like(user_vec)
        if self.use_tags:
            if self.tag_encoding_type == "cross_encoder":
                tags_vec = self.encode_tags_user_conditioned(user_vec, items)
            elif self.tag_encoding_type == "mean_pooling":
                tags_vec = self.encode_tags(items)
            else:
                raise ValueError(f"Tag encoding type {self.tag_encoding_type} not supported")
    
        return user_vec, item_vec, tags_vec
    
    def training_step(self, batch, batch_idx):
        """
        batch:
            users  : [B]
            items  : [B]
            labels : [B]  (1 for positive, 0 for negative)
        """
        users, items, labels = batch

        users = users.to(self.device, non_blocking=True)
        items = items.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).float()

        # ---- forward ----
        user_vec, item_vec, tags_vec = self(users, items)  # calls user-conditioned attention

        # Debug first step
        if self.global_step == 0:
            print(f"\n{'='*60}")
            print(f"FIRST TRAINING STEP DEBUG")
            print(f"{'='*60}")
            print(f"Batch size: {len(users)}")
            print(f"user_vec: shape={user_vec.shape}, has_nan={torch.isnan(user_vec).any()}, mean={user_vec.mean():.4f}, std={user_vec.std():.4f}")
            print(f"item_vec: shape={item_vec.shape}, has_nan={torch.isnan(item_vec).any()}, mean={item_vec.mean():.4f}, std={item_vec.std():.4f}")
            print(f"tags_vec: shape={tags_vec.shape}, has_nan={torch.isnan(tags_vec).any()}, mean={tags_vec.mean():.4f}, std={tags_vec.std():.4f}")
            if torch.isnan(tags_vec).any():
                print(f"  ⚠️ WARNING: tags_vec contains NaN!")
                print(f"  Checking tag embeddings and masks...")
                if self.item_tag_embs is not None:
                    sample_items = items[:5]
                    sample_embs = self.item_tag_embs[sample_items.cpu()]
                    sample_masks = self.item_tag_mask[sample_items.cpu()]
                    print(f"  Sample items: {sample_items.tolist()}")
                    print(f"  Sample embs has_nan: {torch.isnan(sample_embs).any()}")
                    print(f"  Sample masks any_true: {sample_masks.any()}")
                    print(f"  Sample masks sum per item: {sample_masks.sum(dim=1).tolist()}")
                else:
                    print(f"  ⚠️ item_tag_embs is None!")
            print(f"{'='*60}\n")

        alpha = torch.sigmoid(self.tag_gate)         # scalar
        item_combined_vec = alpha * item_vec + tags_vec
        
        # dot-product score
        scores = (user_vec * item_combined_vec).sum(dim=1)
        scores = torch.clamp(scores, -20, 20)
    
        # ---- main loss ----
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels)
    
        # ---- optional embedding norm regularization ----
        reg_loss = torch.tensor(0.0, device=self.device)
    
        if self.hparams.get("embedding_reg", 0.0) > 0:
            reg_lambda = self.hparams.embedding_reg
            reg_item = item_vec.pow(2).sum(dim=1).mean()
            reg_loss = reg_lambda * reg_item
        
        loss = bce_loss + reg_loss
    
        # ---- logging ----
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
    
        if reg_loss.item() > 0:
            self.log("train/reg", reg_loss, on_step=True, on_epoch=True, prog_bar=False)
    
        # ---- debug (low frequency) ----
        if self.use_tags and self.global_step % 1000 == 0:
            with torch.no_grad():
                tag_norm = item_vec.norm(dim=1).mean()
                user_norm = user_vec.norm(dim=1).mean()
                self.log("debug/tag_item_norm", tag_norm, prog_bar=False)
                self.log("debug/user_norm", user_norm, prog_bar=False)
    
            if hasattr(self, "tag_gate"):
                alpha = torch.sigmoid(self.tag_gate)
                self.log("debug/tag_gate", alpha, prog_bar=False)
    
        return loss

    def encode_tags_user_conditioned(
        self,
        user_vec: torch.Tensor,   # [B, 128]
        item_ids: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Returns user-conditioned tag vector: [B, 128]
        """
        # Fetch tag tokens (CPU → GPU)
        tag_embs = self.item_tag_embs[item_ids.cpu()]   # [B, T, 384]
        mask = self.item_tag_mask[item_ids.cpu()]       # [B, T]

        tag_embs = tag_embs.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        # Debug on first training call only
        if self.training and self.global_step == 0 and not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"\n--- encode_tags_user_conditioned DEBUG ---")
            print(f"tag_embs: shape={tag_embs.shape}, has_nan={torch.isnan(tag_embs).any()}, mean={tag_embs.mean():.4f}")
            print(f"mask: shape={mask.shape}, any_true={mask.any()}, sum={mask.sum()}")
            print(f"Items with no tags: {(~mask.any(dim=1)).sum()}/{len(mask)}")

        # If no tags at all → return zeros
        if not mask.any():
            return torch.zeros(
                item_ids.size(0),
                self.hidden_dim,
                device=self.device,
            )

        # ---- project tags into CF space ----
        tag_embs = self.tag_proj(tag_embs)               # [B, T, 128]

        # Instead of normalizing to unit length (which removes magnitude info),
        # just clip extreme values and zero out masked positions
        tag_embs = tag_embs.masked_fill(~mask.unsqueeze(-1), 0.0)
        tag_embs = torch.clamp(tag_embs, min=-10.0, max=10.0)  # prevent extreme values

        if self.training and self.global_step == 0:
            print(f"After projection & normalize: has_nan={torch.isnan(tag_embs).any()}")

        # ---- user-conditioned attention ----
        # Handle items with no tags (all masked) separately to avoid NaN from attention
        has_any_tags = mask.any(dim=1)  # [B] - True if item has at least one tag

        # Initialize output with zeros (use user_vec dtype for consistency with model precision)
        tag_vec = torch.zeros(
            item_ids.size(0),
            self.hidden_dim,
            device=self.device,
            dtype=user_vec.dtype,
        )

        # Only process items that have at least one tag
        if has_any_tags.any():
            items_with_tags = has_any_tags.nonzero(as_tuple=False).squeeze(-1)

            q = user_vec[items_with_tags].unsqueeze(1)   # [B', 1, 128]
            k = v = tag_embs[items_with_tags]            # [B', T, 128]

            # dtype safety (important for bf16/fp16)
            attn_dtype = self.tag_attn.in_proj_weight.dtype
            q = q.to(attn_dtype)
            k = k.to(attn_dtype)
            v = v.to(attn_dtype)

            key_padding_mask = ~mask[items_with_tags]   # [B', T] - True = ignore

            if self.training and self.global_step == 0:
                print(f"Processing {len(items_with_tags)}/{len(item_ids)} items with tags")
                print(f"key_padding_mask: all_true={key_padding_mask.all(dim=1).sum()}/{len(key_padding_mask)}")

            attn_out, _ = self.tag_attn(
                q, k, v,
                key_padding_mask=key_padding_mask,
            )                                            # [B', 1, 128]

            if self.training and self.global_step == 0:
                print(f"After attention: has_nan={torch.isnan(attn_out).any()}")

            attn_out = attn_out.squeeze(1)               # [B', 128]
            tag_vec_with_tags = self.tag_fc(attn_out)   # [B', 128]

            if self.training and self.global_step == 0:
                print(f"After tag_fc: has_nan={torch.isnan(tag_vec_with_tags).any()}")
                print(f"tag_vec dtype: {tag_vec.dtype}, tag_vec_with_tags dtype: {tag_vec_with_tags.dtype}")

            # Place the results back into the full batch (ensure dtype matches)
            tag_vec[items_with_tags] = tag_vec_with_tags.to(tag_vec.dtype)

        if self.training and self.global_step == 0:
            print(f"After tag_fc: has_nan={torch.isnan(tag_vec).any()}")
            print(f"--- end encode_tags_user_conditioned DEBUG ---\n")

        # Final normalization keeps scale stable
        # tag_vec = F.normalize(tag_vec, dim=1)

        return tag_vec
    
    def validation_step(self, batch, batch_idx):
        # We do not compute batch-wise validation loss.
        # This exists ONLY to trigger Lightning's validation loop.
        return None

    def on_validation_epoch_end(self):
        # Manual validation evaluation using datamodule's val_df and neg sampler
        dm = self.trainer.datamodule
        metrics = eval_split_fast(
            model=self,
            df=dm.val_df,
            neg_sampler=dm.neg_sampler_eval,
            n_items=self.hparams.num_items,
            n_neg=self.hparams.n_neg_eval,
            K=self.hparams.K,
            max_eval_samples=self.hparams.get("max_eval_samples", None),
            seed=self.hparams.seed,
        )

        # log with names that include K for clarity
        K = self.hparams.K
        self.log("val/loss", metrics["loss"], prog_bar=True)
        self.log(f"val/precision@{K}", metrics["precision@K"], on_epoch=True, prog_bar=True)
        self.log(f"val/recall@{K}", metrics["recall@K"], on_epoch=True, prog_bar=False)
        self.log(f"val/ndcg@{K}", metrics["ndcg@K"], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,   # good default for embeddings + attention
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return optimizer
