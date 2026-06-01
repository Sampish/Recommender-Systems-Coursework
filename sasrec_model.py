"""
SASRec: Self-Attentive Sequential Recommendation
PyTorch implementation

Reference: https://github.com/pmixer/SASRec.pytorch
Paper: https://arxiv.org/abs/1808.09781
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation Model
    
    Uses Transformer architecture with causal masking to predict next item
    from user's sequential interaction history.
    """
    
    def __init__(self, n_items, maxlen, d_model=64, n_heads=2, n_blocks=2, dropout=0.2, use_time_features=True):
        """
        Args:
            n_items: Number of unique items (excluding padding index 0)
            maxlen: Maximum sequence length
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_blocks: Number of Transformer blocks
            dropout: Dropout probability
            use_time_features: Whether to use temporal features (time_bin)
        """
        super(SASRec, self).__init__()
        
        self.n_items = n_items
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_time_features = use_time_features
        
        # Item embedding (item 0 reserved for padding)
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        
        # Position embedding (learnable)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        
        # Temporal embedding (time_bin: 0-3 for 6-hour periods)
        if use_time_features:
            self.time_bin_emb = nn.Embedding(4, d_model // 4)  # 4 time bins
            # Linear projection to match dimensions
            self.time_proj = nn.Linear(d_model // 4, d_model)
        
        # Dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        
        # Final layer norm
        self.ln = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, seqs, time_bins=None):
        """
        Args:
            seqs: (batch_size, maxlen) LongTensor of item indices
            time_bins: (batch_size, maxlen) LongTensor of time bins (0-3), optional
        
        Returns:
            (batch_size, maxlen, d_model) output embeddings
        """
        batch_size, seq_len = seqs.shape
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=seqs.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        item_embs = self.item_emb(seqs)  # (batch, maxlen, d_model)
        pos_embs = self.pos_emb(positions)  # (batch, maxlen, d_model)
        
        # Combine item + position embeddings
        x = item_embs + pos_embs
        
        # Add temporal embeddings if available
        if self.use_time_features and time_bins is not None:
            time_embs = self.time_bin_emb(time_bins)  # (batch, maxlen, d_model//4)
            time_embs = self.time_proj(time_embs)  # (batch, maxlen, d_model)
            x = x + time_embs  # Add temporal context
        
        x = self.emb_dropout(x)
        
        # Create attention mask (attend only to non-padding)
        # mask: (batch, maxlen) with 1 for real items, 0 for padding
        mask = (seqs != 0).float()
        
        # Create causal mask (prevent attending to future items)
        # causal_mask: (maxlen, maxlen) upper triangular with False on diagonal
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=seqs.device),
            diagonal=1
        )
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask, causal_mask)
        
        # Final layer norm
        x = self.ln(x)
        
        return x
    
    def predict(self, seqs, candidates, time_bins=None):
        """
        Predict scores for candidate items given sequences
        
        Args:
            seqs: (batch_size, maxlen) LongTensor of item sequences
            candidates: (batch_size, n_candidates) LongTensor of candidate items
            time_bins: (batch_size, maxlen) LongTensor of time bins, optional
        
        Returns:
            (batch_size, n_candidates) scores
        """
        # Get sequence representations
        seq_output = self.forward(seqs, time_bins)  # (batch, maxlen, d_model)
        
        # Take last position output (for next item prediction)
        # Find last non-padding position for each sequence
        seq_lengths = (seqs != 0).sum(dim=1)  # (batch,)
        last_indices = (seq_lengths - 1).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        last_indices = last_indices.expand(-1, -1, self.d_model)  # (batch, 1, d_model)
        
        # Gather last position embeddings
        seq_emb = torch.gather(seq_output, 1, last_indices).squeeze(1)  # (batch, d_model)
        
        # Get candidate item embeddings
        cand_embs = self.item_emb(candidates)  # (batch, n_candidates, d_model)
        
        # Compute dot product scores
        scores = (seq_emb.unsqueeze(1) * cand_embs).sum(dim=-1)  # (batch, n_candidates)
        
        return scores


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head self-attention and FFN"""
    
    def __init__(self, d_model, n_heads, dropout=0.2):
        super(TransformerBlock, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Multi-head attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask, causal_mask):
        """
        Args:
            x: (batch, seq_len, d_model) input
            mask: (batch, seq_len) padding mask (1 for real, 0 for padding)
            causal_mask: (seq_len, seq_len) causal attention mask
        
        Returns:
            (batch, seq_len, d_model) output
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-head self-attention with residual + layer norm
        attn_out = self._multihead_attention(x, x, x, mask, causal_mask)
        x = self.ln1(x + self.dropout(attn_out))
        
        # Feed-forward with residual + layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        
        # Apply padding mask to output
        x = x * mask.unsqueeze(-1)
        
        return x
    
    def _multihead_attention(self, q, k, v, mask, causal_mask):
        """Multi-head attention"""
        batch_size, seq_len, d_model = q.shape
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_head)
        Q = self.q_linear(q).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.k_linear(k).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.v_linear(v).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention: (batch, n_heads, seq_len, d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # Apply causal mask (prevent attending to future positions)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Apply padding mask (prevent attending to padding)
        # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
        padding_mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(padding_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) x (batch, n_heads, seq_len, d_head)
        # -> (batch, n_heads, seq_len, d_head)
        attn_out = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, d_model)
        
        # Final linear projection
        attn_out = self.out_linear(attn_out)
        
        return attn_out


def train_sasrec(model, train_loader, optimizer, device, epoch):
    """
    Train SASRec for one epoch
    
    Args:
        model: SASRec model
        train_loader: DataLoader with (seqs, pos_items, neg_items) or (seqs, time_bins, pos_items, neg_items)
        optimizer: PyTorch optimizer
        device: torch device
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle both formats: with and without time_bins
        if len(batch_data) == 4:
            seqs, time_bins, pos_items, neg_items = batch_data
            seqs = seqs.to(device)
            time_bins = time_bins.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
        else:
            seqs, pos_items, neg_items = batch_data
            seqs = seqs.to(device)
            time_bins = None
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
        
        # Forward pass
        seq_output = model(seqs, time_bins)  # (batch, maxlen, d_model)
        
        # Get positive and negative item embeddings
        pos_embs = model.item_emb(pos_items)  # (batch, maxlen, d_model)
        neg_embs = model.item_emb(neg_items)  # (batch, maxlen, d_model)
        
        # Compute logits
        pos_logits = (seq_output * pos_embs).sum(dim=-1)  # (batch, maxlen)
        neg_logits = (seq_output * neg_embs).sum(dim=-1)  # (batch, maxlen)
        
        # Create mask for non-padding positions
        mask = (pos_items != 0).float()
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-10) * mask
        neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-10) * mask
        
        loss = (pos_loss + neg_loss).sum() / mask.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_sasrec(model, test_data, user_train_seqs, n_items, device, k=10):
    """
    Evaluate SASRec on test data
    
    Args:
        model: SASRec model
        test_data: List of (user_id, test_item) tuples
        user_train_seqs: Dict mapping user_id -> training sequence
        n_items: Total number of items
        device: torch device
        k: Top-K for evaluation
    
    Returns:
        Dict with NDCG@K and Hit Rate@K
    """
    model.eval()
    
    ndcg_scores = []
    hit_rates = []
    
    with torch.no_grad():
        for user_id, test_item in test_data:
            # Get user's training sequence
            seq = user_train_seqs.get(user_id, [0])  # Padding if no history
            
            # Prepare sequence tensor
            seq_tensor = torch.LongTensor([seq]).to(device)
            
            # Get all item candidates (excluding test item for fairness)
            candidates = torch.arange(1, n_items + 1, dtype=torch.long, device=device).unsqueeze(0)
            
            # Predict scores
            scores = model.predict(seq_tensor, candidates).squeeze(0)  # (n_items,)
            
            # Get top-K items
            _, top_indices = torch.topk(scores, k)
            top_items = candidates.squeeze(0)[top_indices].cpu().numpy()
            
            # Check if test item is in top-K
            hit = 1.0 if test_item in top_items else 0.0
            hit_rates.append(hit)
            
            # Compute NDCG
            if hit:
                rank = np.where(top_items == test_item)[0][0]
                ndcg = 1.0 / np.log2(rank + 2)
            else:
                ndcg = 0.0
            ndcg_scores.append(ndcg)
    
    return {
        'NDCG@{}'.format(k): np.mean(ndcg_scores),
        'HR@{}'.format(k): np.mean(hit_rates)
    }
