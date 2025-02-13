import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch.utils.checkpoint as checkpoint
import re
import random

# -----------------------------------------------------------------------------
# Helper: Weight Initialization Function
# -----------------------------------------------------------------------------
def init_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

# -----------------------------------------------------------------------------
# 1. JointEncoder
# -----------------------------------------------------------------------------
class JointEncoder(nn.Module):
    """
    Encodes a concatenated document and parent topic string.
    
    The input is expected to be a list of strings where each string is:
      "<joint_prompt>[document text] || [parent topic text]"
      
    The encoder uses NV-Embed-v2 (via SentenceTransformer) to produce both a global
    embedding and token-level representations.
    """
    def __init__(self, model_name="nvidia/NV-Embed-v2", device="cuda"):
        super(JointEncoder, self).__init__()
        self.device = device
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        # Freeze encoder parameters.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, joint_texts):
        # Tokenize the list of joint strings.
        encoded_input = self.tokenizer(joint_texts, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
        with torch.no_grad():
            outputs = self.model.forward(encoded_input, output_hidden_states=True)
            # Expecting 'sentence_embedding' and 'token_embeddings' in outputs.
            global_embeddings = outputs['sentence_embedding']
            token_embeddings = outputs['token_embeddings']
        return global_embeddings.float(), token_embeddings.float(), encoded_input['input_ids']

# -----------------------------------------------------------------------------
# 2. PhraseDecoder
# -----------------------------------------------------------------------------
class PhraseDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=6, num_heads=8,
                 max_length=16, pad_token_id=0, bos_token_id=101, eos_token_id=102,
                 use_checkpointing=False, enable_copy_mechanism=False, dropout=0.1):
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

        # Pattern for allowed tokens (lowercase letters, digits, space and punctuation).
        self.pattern = re.compile(r"^[a-z0-9 ,\.\-]+$")

        if self.enable_copy_mechanism:
            self.copy_linear = nn.Linear(hidden_dim, 1)
            self.doc_proj = nn.Linear(hidden_dim, hidden_dim)
            init_linear(self.copy_linear)
            init_linear(self.doc_proj)

    def forward(self, tgt, memory, doc_token_embeddings=None, doc_input_ids=None, tgt_mask=None):
        tgt = tgt.to(self.positional_encoding.device)
        seq_len = tgt.size(1)
        if seq_len <= 0:
            raise ValueError("Target sequence is empty; cannot generate mask.")
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        pos_enc = self.positional_encoding[:, :seq_len, :]
        tgt_emb = self.embedding(tgt) + pos_enc
        tgt_emb = self.dropout(tgt_emb)
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
                 temperature=0.6, top_p=0.9, freq_penalty=0, pres_penalty=0,
                 unwanted_penalty=1.0, tokenizer=None):
        def apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty):
            for b in range(logits.size(0)):
                gen_tokens = generated[b].tolist()
                for token in set(gen_tokens):
                    count = gen_tokens.count(token)
                    logits[b, token] -= freq_penalty * count + pres_penalty
            return logits

        def apply_unwanted_penalty(logits, extra_penalty, tokenizer, generated):
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id
            for token_id in range(logits.size(1)):
                if token_id in {bos_token_id, eos_token_id, pad_token_id}:
                    continue
                token_str = tokenizer.decode([token_id]).strip()
                if token_str == "":
                    continue
                if not self.pattern.fullmatch(token_str):
                    logits[:, token_id] -= extra_penalty
            batch_size, seq_len = generated.size()
            allowed_end = {eos_token_id, pad_token_id}
            for b in range(batch_size):
                j = seq_len  
                for i in reversed(range(1, seq_len)):
                    if generated[b, i].item() not in allowed_end:
                        j = i + 1
                        break
                for i in range(1, j):
                    token = generated[b, i].item()
                    if token in {bos_token_id, eos_token_id, pad_token_id}:
                        logits[b, token] -= extra_penalty
            return logits

        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=memory.device)
        for step in range(self.max_length - 1):
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(memory.device)
            logits = self.forward(generated, memory, doc_token_embeddings, doc_input_ids, tgt_mask=tgt_mask)
            logits = logits[:, -1, :] / temperature
            logits = apply_repetition_penalty(logits, generated, freq_penalty, pres_penalty)
            if tokenizer is not None and unwanted_penalty > 0:
                logits = apply_unwanted_penalty(logits, unwanted_penalty, tokenizer, generated)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
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

# -----------------------------------------------------------------------------
# 3. JointTopicExpanModel
# -----------------------------------------------------------------------------
class JointTopicExpanModel(nn.Module):
    def __init__(self, encoder_model_name="nvidia/NV-Embed-v2", vocab_size=30522, hidden_dim=768,
                 num_decoder_layers=6, num_decoder_heads=8, max_length=16, pad_token_id=0,
                 bos_token_id=101, eos_token_id=102, use_checkpointing=False, enable_copy_mechanism=False,
                 device="cuda", joint_prompt=None, dropout=0.1):
        super(JointTopicExpanModel, self).__init__()
        self.device = device
        # Default prompt for joint encoding if none is provided.
        if joint_prompt is None:
            self.joint_prompt = ("Instruct: Encode the following document and parent topic, separated by ||, "
                                 "jointly for subtopic generation.\nDocument and Parent Topic: ")
        else:
            self.joint_prompt = joint_prompt
        self.joint_encoder = JointEncoder(model_name=encoder_model_name, device=device)
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
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name, trust_remote_code=True)

    def forward(self, documents, parent_topics, tgt_phrase):
        """
        Args:
          documents: list of strings (document texts)
          parent_topics: list of strings (parent topic texts)
          tgt_phrase: Tensor of target subtopic phrase token IDs, shape [B, seq_len]
        """
        # Create a joint string for each sample.
        joint_texts = [f"{self.joint_prompt}{doc} || {parent}" for doc, parent in zip(documents, parent_topics)]
        global_embed, token_embeddings, input_ids = self.joint_encoder(joint_texts)
        memory = global_embed.unsqueeze(1)  # [B, 1, hidden_dim]
        if self.phrase_decoder.enable_copy_mechanism:
            logits = self.phrase_decoder(tgt_phrase, memory, token_embeddings, input_ids)
        else:
            logits = self.phrase_decoder(tgt_phrase, memory)
        return logits

    def generate_phrase(self, documents, parent_topics, temperature=0.6, top_p=0.9, freq_penalty=0,
                        pres_penalty=0, unwanted_penalty=1.0):
        """
        Generates a subtopic phrase given a batch of documents and parent topics.
        """
        joint_texts = [f"{self.joint_prompt}{doc} || {parent}" for doc, parent in zip(documents, parent_topics)]
        global_embed, token_embeddings, input_ids = self.joint_encoder(joint_texts)
        memory = global_embed.unsqueeze(1)
        generated = self.phrase_decoder.generate(memory, token_embeddings, input_ids,
                                                 temperature=temperature, top_p=top_p,
                                                 freq_penalty=freq_penalty, pres_penalty=pres_penalty,
                                                 unwanted_penalty=unwanted_penalty, tokenizer=self.tokenizer)
        return generated
