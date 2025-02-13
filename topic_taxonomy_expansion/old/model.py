import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import dgl
from dgl.nn.pytorch.conv import GATConv
import torch.utils.checkpoint as checkpoint
import gc
import re
from transformers import AutoTokenizer
import random

# Helper: Weight Initialization Function
# -----------------------------------------------------------------------------
def init_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. DocumentEncoder
# -----------------------------------------------------------------------------
class DocumentEncoder(nn.Module):
    def __init__(self, encoder_model=None, model_name="nvidia/NV-Embed-v2", device="cuda"):
        """
        Initialize the DocumentEncoder.
        
        Args:
          encoder_model: (Optional) a pretrained encoder model.
          model_name (str): Name of the model to load if encoder_model is None.
          device (str): The device to load the model onto.
        """
        super(DocumentEncoder, self).__init__()
        self.device = device
        if encoder_model:
            self.model = encoder_model
        else:
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        # Freeze the document encoder parameters.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, doc_input_ids, doc_attention_mask):
        """
        Encode a batch of document strings.
        
        Args:
          doc_input_ids (list of Tensor): List of input IDs for each document.
          doc_attention_mask (list of Tensor): List of attention masks for each document.
        
        Returns:
          tuple: (global_embeddings, token_embeddings, input_ids)
            global_embeddings: Tensor of shape [B, hidden_dim]
            token_embeddings: Tensor of shape [B, seq_len, hidden_dim]
            input_ids: Tensor of shape [B, seq_len]
        """
        encoded_input = {"input_ids": doc_input_ids,
                             "attention_mask": doc_attention_mask}
        # Combine list of dictionaries into one dictionary if necessary.
        if isinstance(encoded_input, list):
            combined = {}
            for key in encoded_input[0]:
                combined[key] = torch.stack([d[key] for d in encoded_input])
            encoded_input = combined
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
        with torch.no_grad():
            outputs = self.model.forward(encoded_input, output_hidden_states=True)
            token_embeddings = outputs['token_embeddings']
            global_embeddings = outputs['sentence_embedding']
        return global_embeddings.float(), token_embeddings.float(), encoded_input['input_ids']

# -----------------------------------------------------------------------------
# 2. GraphTopicEncoder
# -----------------------------------------------------------------------------
class GraphTopicEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        """
        Initialize the GraphTopicEncoder.
        
        Args:
          in_dim (int): Input dimension.
          hidden_dim (int): Hidden dimension for each GAT layer.
          num_layers (int): Number of GAT layers.
          num_heads (int): Number of attention heads per layer.
        """
        super(GraphTopicEncoder, self).__init__()
        self.num_layers = num_layers
        self.downward_layers = nn.ModuleList()
        self.upward_layers = nn.ModuleList()
        self.sideward_layers = nn.ModuleList()
        for i in range(num_layers):
            inp = in_dim if i == 0 else hidden_dim * num_heads
            self.downward_layers.append(GATConv(inp, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True))
            self.upward_layers.append(GATConv(inp, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True))
            self.sideward_layers.append(GATConv(inp, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True))
        self.fc_out = nn.Linear(hidden_dim * num_heads, hidden_dim)
        # Initialize fc_out weights
        init_linear(self.fc_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, downward_graph, upward_graph, sideward_graph, node_feats):
        """
        Encode the fixed topic graph.
        
        Args:
          downward_graph, upward_graph, sideward_graph: DGLGraphs representing different relations.
          node_feats: Initial node features (typically from the fixed topic embeddings).
          
        Returns:
          Tensor of shape [num_nodes, hidden_dim] representing topic embeddings.
        """
        h_down = node_feats
        for layer in self.downward_layers:
            h_down = layer(downward_graph, h_down)
            h_down = h_down.flatten(1)
            h_down = F.elu(h_down, inplace=True)
            h_down=self.dropout(h_down)
        h_up = node_feats
        for layer in self.upward_layers:
            h_up = layer(upward_graph, h_up)
            h_up = h_up.flatten(1)
            h_up = F.elu(h_up, inplace=True)
            h_up=self.dropout(h_up)
        h_side = node_feats
        for layer in self.sideward_layers:
            h_side = layer(sideward_graph, h_side)
            h_side = h_side.flatten(1)
            h_side = F.elu(h_side, inplace=True)
            h_side=self.dropout(h_side)
        # Combine the directional information.
        h = h_down + h_up - h_side
        h = self.fc_out(h)
        return h

# -----------------------------------------------------------------------------
# 3. PhraseDecoder
# -----------------------------------------------------------------------------
class PhraseDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=6, num_heads=8,
                 max_length=16, pad_token_id=0, bos_token_id=101, eos_token_id=102,
                 use_checkpointing=False, enable_copy_mechanism=False, dropout=0.1):
        """
        Initialize the PhraseDecoder.
        
        Args:
          vocab_size (int): Vocabulary size.
          hidden_dim (int): Hidden dimension.
          num_layers (int): Number of transformer decoder layers.
          num_heads (int): Number of attention heads.
          max_length (int): Maximum sequence length to generate.
          pad_token_id (int): Padding token id.
          bos_token_id (int): Beginning-of-sequence token id.
          eos_token_id (int): End-of-sequence token id.
          use_checkpointing (bool): Whether to use gradient checkpointing.
          enable_copy_mechanism (bool): Whether to enable the copy mechanism.
        """
        super(PhraseDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        init_linear(self.fc_out)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_checkpointing = use_checkpointing
        self.enable_copy_mechanism = enable_copy_mechanism

        # Define allowed tokens: strings containing only lowercase letters and digits.
        self.pattern = re.compile(r"^[a-z0-9 ,\.\-]+$")

        if self.enable_copy_mechanism:
            self.copy_linear = nn.Linear(hidden_dim, 1)
            self.doc_proj = nn.Linear(hidden_dim, hidden_dim)
            init_linear(self.copy_linear)
            init_linear(self.doc_proj)

    def forward(self, tgt, memory, doc_token_embeddings=None, doc_input_ids=None, tgt_mask=None):
        """
        Forward pass for the decoder.
        
        Args:
          tgt: Target sequence tokens, shape [B, seq_len].
          memory: Encoder output used as context, shape [B, 1, hidden_dim].
          doc_token_embeddings, doc_input_ids: Optional arguments for the copy mechanism.
          tgt_mask: Optional target mask.
          
        Returns:
          logits: Output logits of shape [B, seq_len, vocab_size].
        """
        tgt = tgt.to(self.positional_encoding.device)
        seq_len = tgt.size(1)
        if seq_len <= 0:
            raise ValueError("Target sequence is empty; cannot generate mask.")
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        # Create positional encodings for the target.
        pos_enc = self.positional_encoding[:, :seq_len, :]
        tgt_emb = self.embedding(tgt) + pos_enc
        tgt_emb = self.dropout(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # shape: [seq_len, batch, hidden]
        memory = memory.transpose(0, 1)
        # Use checkpointing if enabled to save memory.
        if self.use_checkpointing:
            output = tgt_emb
            for layer in self.decoder.layers:
                output = checkpoint.checkpoint(layer, output, memory, tgt_mask)
        else:
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        gen_logits = self.fc_out(output)
        # If using the copy mechanism, combine the generated probabilities with copy probabilities.
        if not self.enable_copy_mechanism or doc_token_embeddings is None or doc_input_ids is None:
            return gen_logits
        else:
            gen_prob = F.softmax(gen_logits, dim=-1)
            proj_doc = self.doc_proj(doc_token_embeddings)
            copy_scores = torch.bmm(output, proj_doc.transpose(1, 2))
            copy_attn = F.softmax(copy_scores, dim=-1)
            batch_size, tgt_seq_len, src_seq_len = copy_attn.size()
            vocab_size = gen_logits.size(-1)
            copy_dist = torch.zeros(batch_size, tgt_seq_len, vocab_size, device=output.device)
            for b in range(batch_size):
                copy_dist[b].scatter_add_(1,
                                           doc_input_ids[b].unsqueeze(0).expand(tgt_seq_len, -1),
                                           copy_attn[b])
            p_gen = torch.sigmoid(self.copy_linear(output))
            final_dist = p_gen * gen_prob + (1 - p_gen) * copy_dist
            final_log_prob = torch.log(final_dist + 1e-8)
            return final_log_prob

    def generate(self, memory, doc_token_embeddings=None, doc_input_ids=None,
                temperature=0.6, top_p=0.9, freq_penalty=0, pres_penalty=0,
                unwanted_penalty=1.0, tokenizer=None):
        """
        Generate a sequence using nucleus (top-p) sampling with additional penalties.
        
        The function applies:
        - Repetition and presence penalties (to discourage repeated tokens).
        - Unwanted token penalties: tokens that do not conform to our expected set
            (i.e. lowercase English letters, digits, and ( ,.) only) are penalized.
        
        Args:
        memory: Conditioning context, shape [B, 1, hidden_dim].
        doc_token_embeddings, doc_input_ids: Optional, for the copy mechanism.
        temperature: Scaling factor for logits.
        top_p: Nucleus sampling threshold.
        freq_penalty: Penalty per occurrence of a token.
        pres_penalty: Additional penalty if a token has already appeared.
        unwanted_penalty: Extra penalty for tokens that are unwanted.
        tokenizer: The tokenizer instance (required to decode token strings).
        
        Returns:
        generated: Tensor of generated token IDs of shape [B, generated_length].
        """
        def apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty):
            # Apply penalties based on token frequency in the generated sequence.
            for b in range(logits.size(0)):
                gen_tokens = generated[b].tolist()
                for token in set(gen_tokens):
                    count = gen_tokens.count(token)
                    logits[b, token] -= freq_penalty * count + pres_penalty
            return logits

        def apply_unwanted_penalty(logits, extra_penalty, tokenizer, generated):
            """
            Adjust the logits by applying an extra penalty to tokens that are “unwanted.”
            
            First, for every token in the vocabulary (except BOS, EOS, and PAD),
            if the decoded token does not match the allowed pattern (self.pattern),
            subtract extra_penalty from its logit.
            
            Then, for each example in the batch, we examine the generated sequence
            (starting at index 1). We define an allowed ending region as a contiguous block
            of tokens (from some index j to the end) that are allowed as sequence endings (by default, EOS and PAD).
            For all positions before that region, if a token is one of the special tokens
            (BOS, EOS, or PAD), we subtract extra_penalty from its logit.
            
            This permits the BOS token at position 0, and it permits EOS/PAD tokens if they
            appear as the contiguous ending, but it penalizes any out‐of‐place special tokens
            (including cases where a BOS token is produced consecutively starting at position 1).
            
            Args:
            logits (Tensor): Tensor of shape [B, vocab_size] containing the logits for the current time step.
            extra_penalty (float): The penalty value to subtract.
            tokenizer: The tokenizer instance (which must have bos_token_id, eos_token_id, pad_token_id).
            generated (Tensor): Tensor of shape [B, current_seq_len] containing the tokens generated so far.
            
            Returns:
            Tensor: The adjusted logits.
            """
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id

            # --- 1. Apply penalty to unwanted tokens (except BOS, EOS, and PAD) ---
            for token_id in range(logits.size(1)):
                if token_id in set([bos_token_id, eos_token_id, pad_token_id]):
                    continue  # Skip special tokens here.
                token_str = tokenizer.decode([token_id]).strip()
                if token_str == "":
                    continue
                if not self.pattern.fullmatch(token_str):
                    logits[:, token_id] -= extra_penalty

            batch_size, seq_len = generated.size()

            # --- 2. Penalize out-of-place special tokens individually ---
            # We define the allowed ending region as the contiguous block at the end
            # (starting from some index j, where j>=1) in which every token is in the set
            # allowed_end (by default, EOS and PAD). Tokens before this region are considered
            # "out-of-place" if they are special (BOS, EOS, or PAD).
            allowed_end = set([eos_token_id, pad_token_id])
            for b in range(batch_size):
                # Determine the allowed ending start index for sample b.
                # Initialize j to the sequence length; then iterate backwards from position 1.
                j = seq_len  
                for i in reversed(range(1, seq_len)):
                    if generated[b, i].item() not in allowed_end:
                        j = i + 1
                        break
                # Now, for positions 1 through j-1, if a token is special (BOS, EOS, or PAD),
                # subtract extra_penalty from its logit.
                for i in range(1, j):
                    token = generated[b, i].item()
                    if token in set([bos_token_id, eos_token_id, pad_token_id]):
                        logits[b, token] -= extra_penalty

            return logits


        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=memory.device)

        for step in range(self.max_length - 1):
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(memory.device)
            logits = self.forward(generated, memory, doc_token_embeddings, doc_input_ids, tgt_mask=tgt_mask)
            # Use only the last token’s logits.
            logits = logits[:, -1, :] / temperature

            # Apply repetition/presence penalties.
            logits = apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty)
            
            # Apply unwanted token penalty if a tokenizer is provided.
            if tokenizer is not None and unwanted_penalty > 0:
                logits = apply_unwanted_penalty(logits, unwanted_penalty, tokenizer, generated)

            # Nucleus (top-p) sampling.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to ensure at least one token remains.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))
            # Reorder the logits to the original indexing.
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.eos_token_id).all():
                break
        
        return generated

# -----------------------------------------------------------------------------
# 4. TopicExpanModel
# -----------------------------------------------------------------------------
class TopicExpanModel(nn.Module):
    def __init__(self, encoder_model, vocab_size, hidden_dim, topic_feature_dim,
                 num_topic_layers=2, num_topic_heads=4,
                 num_decoder_layers=6, num_decoder_heads=8,
                 max_length=16, pad_token_id=0, bos_token_id=101, eos_token_id=102,
                 doc_encoder_model="nvidia/NV-Embed-v2", use_checkpointing=False, 
                 enable_copy_mechanism=False, device="cuda", fixed_topic_node_feats=None,
                 use_learnable_fusion=True, bypass_graph=False,
                 temperature=1, top_p=0.9, freq_penalty=0, pres_penalty=0, unwanted_penalty=1.0,dropout=0.1):
        """
        The main model combining:
          - A DocumentEncoder that generates embeddings for documents.
          - A GraphTopicEncoder that produces embeddings for fixed topics.
          - A PhraseDecoder that generates target phrases.
          - A similarity module to align document embeddings with parent topic embeddings.
        
        Args:
          encoder_model: Pretrained document encoder model.
          vocab_size (int): Size of the vocabulary.
          hidden_dim (int): Hidden dimension used across modules.
          topic_feature_dim (int): Input dimension for the topic graph.
          num_topic_layers (int): Number of GAT layers for the topic graph.
          num_topic_heads (int): Number of attention heads per GAT layer.
          num_decoder_layers (int): Number of transformer decoder layers.
          num_decoder_heads (int): Number of attention heads in the decoder.
          max_length (int): Maximum sequence length for generation.
          pad_token_id (int): Padding token id.
          bos_token_id (int): Beginning-of-sequence token id.
          eos_token_id (int): End-of-sequence token id.
          doc_encoder_model (str): Model name for the document encoder.
          use_checkpointing (bool): If True, use gradient checkpointing.
          enable_copy_mechanism (bool): If True, enable the copy mechanism in the decoder.
          device (str): Device to use for computations.
          fixed_topic_node_feats: Precomputed fixed topic node features.
          use_learnable_fusion (bool): If True, use a learnable fusion module to combine document and parent embeddings.
          bypass_graph (bool): If True, bypass the graph topic encoder.
        """
        super(TopicExpanModel, self).__init__()
        self.device = device
        # Initialize the DocumentEncoder.
        self.document_encoder = DocumentEncoder(encoder_model, doc_encoder_model, device=device)
        # Initialize the GraphTopicEncoder.
        self.topic_encoder = GraphTopicEncoder(in_dim=topic_feature_dim,
                                               hidden_dim=hidden_dim,
                                               num_layers=num_topic_layers,
                                               num_heads=num_topic_heads,
                                               dropout=dropout)
        # Initialize the PhraseDecoder.
        self.phrase_decoder = PhraseDecoder(vocab_size, hidden_dim,
                                            num_layers=num_decoder_layers,
                                            num_heads=num_decoder_heads,
                                            max_length=max_length,
                                            pad_token_id=pad_token_id,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            use_checkpointing=use_checkpointing,
                                            enable_copy_mechanism=enable_copy_mechanism,
                                            dropout=dropout)
        # Similarity module to compare document embeddings and parent topic embeddings.
        self.similarity = nn.Bilinear(hidden_dim, hidden_dim, 1)
        init_linear(self.similarity)
        # Instantiate a tokenizer for internal use.
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        self.use_learnable_fusion = use_learnable_fusion
        self.bypass_graph = bypass_graph
        self.top_p = top_p
        self.temperature = temperature
        self.freq_penalty = freq_penalty
        self.pres_penalty = pres_penalty
        self.unwanted_penalty = unwanted_penalty

        if use_learnable_fusion:
            # Learnable fusion module: concatenates document and parent embeddings,
            # then applies a linear transformation, ReLU, dropout, and layer normalization.
            self.fusion = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            )
            self.fusion.apply(init_linear)
        if fixed_topic_node_feats is not None:
            self.register_buffer("fixed_topic_node_feats", fixed_topic_node_feats)
        else:
            self.fixed_topic_node_feats = None

    def forward(self, doc_input_ids, doc_attention_mask, downward_graph, upward_graph, sideward_graph, tgt_phrase, parent_idx):
        """
        Forward pass for training.
        
        Args:
          downward_graph, upward_graph, sideward_graph (dgl.DGLGraph): Graphs for topic relationships.
          tgt_phrase (Tensor): Ground-truth target phrase tokens, shape [B, seq_len].
          parent_idx (int or Tensor): Index (or indices) of the parent topic.
        
        Returns:
          sim_score: Similarity score between document and parent embeddings.
          logits: Decoder logits for phrase generation.
        """
        # Get document embeddings.
        global_doc_embed, doc_token_embeddings, doc_input_ids = self.document_encoder(doc_input_ids, doc_attention_mask)
        global_doc_embed = F.normalize(global_doc_embed, p=2, dim=1)
        # Get topic embeddings from the fixed graph.
        if self.bypass_graph:
            # Bypass the graph: use the fixed topic embeddings directly.
            topic_embeddings = self.fixed_topic_node_feats
        else:
            # Use the graph encoder.
            topic_embeddings = self.topic_encoder(downward_graph, upward_graph, sideward_graph, self.fixed_topic_node_feats)
            # only print with 1/200 chance
            if random.random() < 0.005:
                print("[DEBUG] Topic embeddings after graph encoder: mean = {:.4f}, std = {:.4f}".format(
                    topic_embeddings.mean().item(), topic_embeddings.std().item()))

        # If parent_idx is a single int, expand its embedding to match batch size.
        if isinstance(parent_idx, int):
            parent_embed = topic_embeddings[parent_idx].unsqueeze(0).expand(global_doc_embed.size(0), -1)
        else:
            parent_embed = topic_embeddings.index_select(0, parent_idx)
        # Fuse the document and parent embeddings.
        if self.use_learnable_fusion:
            fused_context = self.fusion(torch.cat([global_doc_embed, parent_embed], dim=1))
        else:
            fused_context = (global_doc_embed + parent_embed) / 2
        memory = fused_context.unsqueeze(1)  # shape: [B, 1, hidden_dim]
        # Decode to produce phrase logits.
        if self.phrase_decoder.enable_copy_mechanism:
            logits = self.phrase_decoder(tgt_phrase, memory, doc_token_embeddings, doc_input_ids)
        else:
            logits = self.phrase_decoder(tgt_phrase, memory)
        # Compute similarity between document and parent embeddings.
        sim_score = self.similarity(global_doc_embed, parent_embed)
        return sim_score, logits

    def generate_phrase(self, doc_input_ids, doc_attention_mask, downward_graph, upward_graph, sideward_graph, parent_idx):
        """
        Generate a phrase given input documents and topic information.
        
        Args:
          downward_graph, upward_graph, sideward_graph: Graphs representing topic relations.
          parent_idx (int or Tensor): Index (or indices) for the parent topic.
          temperature (float): Temperature for scaling logits.
        
        Returns:
          generated (Tensor): Generated token IDs, shape [B, generated_length].
        """
        with torch.no_grad():
            global_doc_embed, doc_token_embeddings, doc_input_ids = self.document_encoder(doc_input_ids, doc_attention_mask)
            if self.bypass_graph:
                topic_embeddings = self.fixed_topic_node_feats
            else:
                topic_embeddings = self.topic_encoder(downward_graph, upward_graph, sideward_graph, self.fixed_topic_node_feats)
            if isinstance(parent_idx, int):
                parent_embed = topic_embeddings[parent_idx].unsqueeze(0).expand(global_doc_embed.size(0), -1)
            else:
                parent_embed = topic_embeddings.index_select(0, parent_idx)
            if self.use_learnable_fusion:
                fused_context = self.fusion(torch.cat([global_doc_embed, parent_embed], dim=1))
            else:
                fused_context = (global_doc_embed + parent_embed) / 2
            memory = fused_context.unsqueeze(1)
            generated = self.phrase_decoder.generate(memory, doc_token_embeddings, doc_input_ids, self.temperature, self.top_p, self.freq_penalty, self.pres_penalty, self.unwanted_penalty, tokenizer=self.tokenizer)
            return generated
