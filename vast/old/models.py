import torch
import torch.nn as nn
import torch.nn.functional as F

def init_he(m):
    if isinstance(m, nn.Linear):
        # Kaiming Uniform initialization with 'relu' nonlinearity.
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
        # apply xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return logits

class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=4096, num_classes=3, dropout_rate=0.1, num_heads=4):
        """
        This classifier expects as input a concatenated vector of dimension 2*embed_dim.
        It splits the input into two tokens (document and topic), stacks them,
        and passes them through a multi-head self-attention layer.
        The output is mean-pooled and fed into a final linear classification layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(init_he)

    def forward(self, x):
        # x: (batch, 2*embed_dim)
        batch_size = x.size(0)
        doc_token = x[:, :self.embed_dim]       # (batch, embed_dim)
        topic_token = x[:, self.embed_dim:]       # (batch, embed_dim)
        tokens = torch.stack([doc_token, topic_token], dim=1)  # (batch, 2, embed_dim)
        tokens = tokens.transpose(0, 1)  # (2, batch, embed_dim)
        attn_output, _ = self.attention(tokens, tokens, tokens)
        tokens = self.layernorm(tokens + self.dropout(attn_output))
        pooled = tokens.mean(dim=0)  # (batch, embed_dim)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

class CrossAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=4096, num_classes=3, num_heads=4, dropout_rate=0.1):
        """
        A cross-attention classifier.
        Expects an input concatenated vector of shape (batch, 2*embed_dim)
        where the first half corresponds to the document and the second half to the topic.
        Uses the document as query and the topic as key and value in a multi-head cross-attention layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(init_he)

    def forward(self, x):
        doc = x[:, :self.embed_dim]     # (batch, embed_dim)
        topic = x[:, self.embed_dim:]     # (batch, embed_dim)
        doc = doc.unsqueeze(0)            # (1, batch, embed_dim)
        topic = topic.unsqueeze(0)        # (1, batch, embed_dim)
        attn_output, _ = self.cross_attn(query=doc, key=topic, value=topic)
        fused = self.layernorm(doc + self.dropout(attn_output))
        fused = fused.squeeze(0)  # (batch, embed_dim)
        logits = self.classifier(fused)
        return logits

# class TGAClassifier(nn.Module):
#     """
#     Topic-Grouped Attention Classifier.
#     This model takes a concatenated vector (batch, 2*embed_dim), splits it into document and topic embeddings,
#     then computes a generalized topic representation via learnable prototypes.
#     It transforms the document embedding, fuses it with the generalized topic representation,
#     and passes the result through a feed-forward network for classification.
#     """
#     def __init__(self, embed_dim=4096, num_classes=3, num_prototypes=50, hidden_dim=512, dropout_rate=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_prototypes = num_prototypes
#         self.dropout = nn.Dropout(dropout_rate)
#         # Learnable global topic prototypes.
#         self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
#         # Transform the document embedding.
#         self.doc_transform = nn.Linear(embed_dim, hidden_dim)
#         # Final classifier: concatenates the transformed document and the generalized topic representation.
#         self.pred_layer = nn.Sequential(
#             nn.Linear(hidden_dim + embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, num_classes)
#         )
#         self.apply(init_he)

#     def forward(self, x):
#         # x: (batch, 2*embed_dim)
#         doc = x[:, :self.embed_dim]   # (batch, embed_dim)
#         topic = x[:, self.embed_dim:] # (batch, embed_dim)
#         topic_norm = F.normalize(topic, p=2, dim=1)
#         prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
#         sim = torch.matmul(topic_norm, prototypes_norm.t())  # (batch, num_prototypes)
#         weights = F.softmax(sim, dim=1)  # (batch, num_prototypes)
#         c_dt = torch.matmul(weights, self.prototypes)  # (batch, embed_dim)
#         doc_transformed = self.doc_transform(doc)  # (batch, hidden_dim)
#         combined = torch.cat([doc_transformed, c_dt], dim=1)
#         combined = self.dropout(combined)
#         logits = self.pred_layer(combined)
#         return logits

# class TGAClassifier(nn.Module):
#     """
#     Topic-Grouped Attention Classifier.
#     This model takes a concatenated vector (batch, 2*embed_dim) produced by our dataset loader,
#     splits it into document and topic embeddings, and computes a generalized topic representation
#     using a set of learnable prototypes. It then transforms the document embedding, fuses it with
#     the generalized topic representation, and passes the result through a feed-forward network for classification.
#     """
#     def __init__(self, embed_dim=4096, num_classes=3, num_prototypes=50, hidden_dim=512, dropout_rate=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_prototypes = num_prototypes
#         self.dropout = nn.Dropout(dropout_rate)
#         # Learnable global topic prototypes; shape: (num_prototypes, embed_dim)
#         self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
#         # Linear transformation for the document embedding.
#         self.doc_transform = nn.Linear(embed_dim, hidden_dim)
#         # Final prediction layer; it takes the concatenation of the transformed document (hidden_dim)
#         # and the generalized topic representation (embed_dim) and maps it to num_classes.
#         self.pred_layer = nn.Sequential(
#             nn.Linear(hidden_dim + embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (batch, 2*embed_dim)
#         Returns:
#             logits: Tensor of shape (batch, num_classes)
#         """
#         # Split the concatenated embeddings.
#         doc = x[:, :self.embed_dim]    # (batch, embed_dim)
#         topic = x[:, self.embed_dim:]  # (batch, embed_dim)
        
#         # Normalize topic embeddings and prototypes.
#         topic_norm = F.normalize(topic, p=2, dim=1)  # (batch, embed_dim)
#         prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # (num_prototypes, embed_dim)
        
#         # Compute cosine similarity: (batch, num_prototypes)
#         sim = torch.matmul(topic_norm, prototypes_norm.t())
#         # Softmax to obtain weights.
#         weights = F.softmax(sim, dim=1)
#         # Weighted sum over prototypes yields a generalized topic representation.
#         c_dt = torch.matmul(weights, self.prototypes)  # (batch, embed_dim)
        
#         # Transform the document embedding.
#         doc_transformed = self.doc_transform(doc)  # (batch, hidden_dim)
#         # Concatenate the transformed document representation with the generalized topic representation.
#         combined = torch.cat([doc_transformed, c_dt], dim=1)  # (batch, hidden_dim + embed_dim)
#         combined = self.dropout(combined)
#         logits = self.pred_layer(combined)
#         return logits
