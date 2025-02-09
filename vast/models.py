#!/usr/bin/env python
"""
models.py

Defines model architectures for stance detection.
Available models:
  - StanceClassifier: a simple linear classifier.
  - AttentionClassifier: a self-attention classifier.
  - CrossAttentionClassifier: an advanced model using cross-attention
    to fuse document and topic embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StanceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        A simple linear classifier.
        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits


class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=4096, num_classes=3, dropout_rate=0.1, num_heads=4):
        """
        This classifier expects as input a concatenated vector of dimension 2*embed_dim.
        It splits the input into two tokens (document and topic), stacks them,
        and passes them through a multi-head self-attention layer. The output is mean-pooled
        and fed into a final linear classification layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        # Multi-head self-attention expects input shape (seq_len, batch, embed_dim).
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)
        # Final classifier: maps pooled representation (of dimension embed_dim) to num_classes.
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: shape (batch, 2*embed_dim)
        batch_size = x.size(0)
        # Split the concatenated vector into two tokens.
        doc_token = x[:, :self.embed_dim]       # shape (batch, embed_dim)
        topic_token = x[:, self.embed_dim:]       # shape (batch, embed_dim)
        # Stack them to form a sequence: shape (batch, 2, embed_dim)
        tokens = torch.stack([doc_token, topic_token], dim=1)
        # Transpose to shape (seq_len, batch, embed_dim) as required by nn.MultiheadAttention.
        tokens = tokens.transpose(0, 1)
        # Apply multi-head self-attention.
        attn_output, _ = self.attention(tokens, tokens, tokens)
        # Residual connection + dropout + layer normalization.
        tokens = self.layernorm(tokens + self.dropout(attn_output))
        # Pool across the sequence (here using mean pooling).
        pooled = tokens.mean(dim=0)  # shape (batch, embed_dim)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class CrossAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=4096, num_classes=3, num_heads=4, dropout_rate=0.1):
        """
        A cross-attention classifier.
        Expects an input concatenated vector of shape (batch, 2*embed_dim)
        where the first half corresponds to the document and the second half to the topic.
        It uses the document as the query and the topic as key and value in a multi-head cross-attention layer.
        Args:
            embed_dim (int): Dimension of each individual embedding.
            num_classes (int): Number of output classes.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        # The multi-head attention layer expects input shape (seq_len, batch, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: shape (batch, 2*embed_dim)
        doc = x[:, :self.embed_dim]     # document part
        topic = x[:, self.embed_dim:]     # topic part
        # Unsqueeze to add sequence dimension: (1, batch, embed_dim)
        doc = doc.unsqueeze(0)
        topic = topic.unsqueeze(0)
        # Compute cross-attention: document is query, topic is key and value.
        attn_output, _ = self.cross_attn(query=doc, key=topic, value=topic)
        # Add residual connection, apply dropout and layer normalization.
        fused = self.layernorm(doc + self.dropout(attn_output))
        fused = fused.squeeze(0)  # (batch, embed_dim)
        logits = self.classifier(fused)
        return logits
