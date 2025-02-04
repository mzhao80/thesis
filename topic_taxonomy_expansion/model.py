import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import dgl
from dgl.nn.pytorch.conv import GATConv
import torch.utils.checkpoint as checkpoint
import gc
from transformers import AutoTokenizer

###############################################################################
# 1. DocumentEncoder: Returns global embedding, token-level embeddings, and input IDs.
###############################################################################
class DocumentEncoder(nn.Module):
    def __init__(self, encoder_model=None, model_name="nvidia/NV-Embed-v2", device="cuda"):
        super(DocumentEncoder, self).__init__()
        self.device = device
        if encoder_model:
            self.model = encoder_model
        else:
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.model.eval()  # set to eval and freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, doc_texts):
        """
        Expects a list of document strings.
        Returns:
          - global_embeddings: [B, hidden_dim]
          - token_embeddings: [B, seq_len, hidden_dim]
          - input_ids: [B, seq_len]
        """
        encoded_input = self.model.tokenize(doc_texts)
        # If tokenization returns a list of dictionaries, combine them:
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

###############################################################################
# 2. GraphTopicEncoder: Encodes a fixed graph of parent topics using GAT layers.
###############################################################################
class GraphTopicEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=4):
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
    
    def forward(self, downward_graph, upward_graph, sideward_graph, node_feats):
        h_down = node_feats
        for layer in self.downward_layers:
            h_down = layer(downward_graph, h_down)
            h_down = h_down.flatten(1)
            h_down = F.elu(h_down, inplace=True)
        h_up = node_feats
        for layer in self.upward_layers:
            h_up = layer(upward_graph, h_up)
            h_up = h_up.flatten(1)
            h_up = F.elu(h_up, inplace=True)
        h_side = node_feats
        for layer in self.sideward_layers:
            h_side = layer(sideward_graph, h_side)
            h_side = h_side.flatten(1)
            h_side = F.elu(h_side, inplace=True)
        h = h_down + h_up - h_side
        h = self.fc_out(h)
        return h

###############################################################################
# 3. PhraseDecoder: Transformer-based decoder with optional copy mechanism.
###############################################################################
class PhraseDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=6, num_heads=8,
                 max_length=32, pad_token_id=0, bos_token_id=101, eos_token_id=102,
                 use_checkpointing=False, enable_copy_mechanism=False):
        super(PhraseDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_checkpointing = use_checkpointing
        self.enable_copy_mechanism = enable_copy_mechanism

        if self.enable_copy_mechanism:
            self.copy_linear = nn.Linear(hidden_dim, 1)
            self.doc_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tgt, memory, doc_token_embeddings=None, doc_input_ids=None, tgt_mask=None):
        tgt = tgt.to(self.positional_encoding.device)
        seq_len = tgt.size(1)
        if seq_len <= 0:
            raise ValueError("Target sequence is empty; cannot generate mask.")
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        if seq_len > self.positional_encoding.size(1):
            extra_len = seq_len - self.positional_encoding.size(1)
            extra = self.positional_encoding[:, -1:, :].expand(1, extra_len, -1)
            pos_enc = torch.cat([self.positional_encoding, extra], dim=1)
        else:
            pos_enc = self.positional_encoding[:, :seq_len, :]
        tgt_emb = self.embedding(tgt) + pos_enc
        tgt_emb = tgt_emb.transpose(0, 1)  # [seq_len, batch, hidden]
        memory = memory.transpose(0, 1)
        if self.use_checkpointing:
            output = tgt_emb
            for layer in self.decoder.layers:
                output = checkpoint.checkpoint(layer, output, memory, tgt_mask)
        else:
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        gen_logits = self.fc_out(output)
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
                temperature=1.0, top_p=0.9, freq_penalty=0.5, pres_penalty=0.5,
                unwanted_penalty=1.0, tokenizer=None):
        """
        Generate a sequence using nucleus (top-p) sampling with additional penalties.
        
        Parameters:
        temperature: scaling factor for logits.
        top_p: nucleus sampling threshold.
        freq_penalty: penalty for each occurrence of a token.
        pres_penalty: additional penalty if a token has appeared.
        unwanted_penalty: extra penalty to apply for unwanted tokens.
        tokenizer: the tokenizer instance, used to determine which tokens are unwanted.
        
        Returns:
        A tensor of shape [batch_size, generated_length] of generated token IDs.
        """
        
        def apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty):
            # logits: [batch_size, vocab_size]
            # generated: [batch_size, current_length]
            for b in range(logits.size(0)):
                gen_tokens = generated[b].tolist()
                for token in set(gen_tokens):
                    count = gen_tokens.count(token)
                    logits[b, token] -= freq_penalty * count + pres_penalty
            return logits

        def apply_unwanted_penalty(logits, generated, extra_penalty, tokenizer):
            # We'll penalize tokens that are isolated quotes and also if the last generated token is whitespace,
            # then penalize candidate tokens that decode to whitespace.
            batch_size = logits.size(0)
            
            for b in range(batch_size):
                # Always penalize the quote token.
                logits[b, self.quote_token_id] -= extra_penalty
                
                # Check the last generated token for whitespace.
                last_token_id = generated[b, -1].item()
                last_token_str = tokenizer.decode([last_token_id]).strip()
                if last_token_str == "":
                    # If the last token is whitespace, penalize candidate tokens that decode to only whitespace.
                    # For each candidate token in the vocabulary:
                    for token_id in range(logits.size(1)):
                        token_str = tokenizer.decode([token_id]).strip()
                        # If token_str is empty, then it's essentially a whitespace token.
                        if token_str == "":
                            logits[b, token_id] -= extra_penalty
            return logits

        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=memory.device)
        
        for step in range(self.max_length - 1):
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(memory.device)
            logits = self.forward(generated, memory, doc_token_embeddings, doc_input_ids, tgt_mask=tgt_mask)
            logits = logits[:, -1, :] / temperature

            # Apply repetition and presence penalties.
            logits = apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty)
            
            # Apply unwanted penalties if a tokenizer is provided.
            # if tokenizer is not None and unwanted_penalty > 0:
            #     logits = apply_unwanted_penalty(logits, generated, unwanted_penalty, tokenizer)

            # Nucleus (top-p) sampling:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices so that at least one token remains.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.eos_token_id).all():
                break
        return generated

###############################################################################
# 4. TopicExpanModel: Fuses document and parent embeddings (with an optional learnable fusion).
###############################################################################
class TopicExpanModel(nn.Module):
    def __init__(self, encoder_model, vocab_size, hidden_dim, topic_feature_dim,
                 num_topic_layers=2, num_topic_heads=4,
                 num_decoder_layers=6, num_decoder_heads=8,
                 max_length=32, pad_token_id=0, bos_token_id=101, eos_token_id=102,
                 doc_encoder_model="nvidia/NV-Embed-v2", use_checkpointing=False, 
                 enable_copy_mechanism=False, device="cuda", fixed_topic_node_feats=None,
                 use_learnable_fusion=False):
        super(TopicExpanModel, self).__init__()
        self.device = device
        self.document_encoder = DocumentEncoder(encoder_model, doc_encoder_model, device=device)
        self.topic_encoder = GraphTopicEncoder(in_dim=topic_feature_dim,
                                               hidden_dim=hidden_dim,
                                               num_layers=num_topic_layers,
                                               num_heads=num_topic_heads)
        self.phrase_decoder = PhraseDecoder(vocab_size, hidden_dim,
                                            num_layers=num_decoder_layers,
                                            num_heads=num_decoder_heads,
                                            max_length=max_length,
                                            pad_token_id=pad_token_id,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            use_checkpointing=use_checkpointing,
                                            enable_copy_mechanism=enable_copy_mechanism)
        self.similarity = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        self.use_learnable_fusion = use_learnable_fusion
        if use_learnable_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            )
        if fixed_topic_node_feats is not None:
            self.register_buffer("fixed_topic_node_feats", fixed_topic_node_feats)
        else:
            self.fixed_topic_node_feats = None

    def forward(self, doc_texts, downward_graph, upward_graph, sideward_graph, tgt_phrase, parent_idx):
        global_doc_embed, doc_token_embeddings, doc_input_ids = self.document_encoder(doc_texts)
        topic_embeddings = self.topic_encoder(downward_graph, upward_graph, sideward_graph, self.fixed_topic_node_feats)
        # Expand parent's embedding if parent_idx is a single int.
        if isinstance(parent_idx, int):
            parent_embed = topic_embeddings[parent_idx].unsqueeze(0).expand(global_doc_embed.size(0), -1)
        else:
            parent_embed = topic_embeddings.index_select(0, parent_idx)
        if self.use_learnable_fusion:
            fused_context = self.fusion(torch.cat([global_doc_embed, parent_embed], dim=1))
        else:
            fused_context = (global_doc_embed + parent_embed) / 2
        memory = fused_context.unsqueeze(1)
        if self.phrase_decoder.enable_copy_mechanism:
            logits = self.phrase_decoder(tgt_phrase, memory, doc_token_embeddings, doc_input_ids)
        else:
            logits = self.phrase_decoder(tgt_phrase, memory)
        sim_score = self.similarity(global_doc_embed, parent_embed)
        return sim_score, logits

    def generate_phrase(self, doc_texts, downward_graph, upward_graph, sideward_graph, parent_idx, temperature=1.0):
        global_doc_embed, doc_token_embeddings, doc_input_ids = self.document_encoder(doc_texts)
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
        generated = self.phrase_decoder.generate(memory, doc_token_embeddings, doc_input_ids, temperature, tokenizer=self.tokenizer)
        return generated
