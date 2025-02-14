# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class NVEmbedEncoder(nn.Module):
    def __init__(self, model_name="nvidia/nv-embed-v2", trust_remote_code=True):
        """
        Loads the nv-embed-v2 model with remote code trusted.
        Assumes that the loaded model implements an `encode` method that accepts:
           - texts: a list of input strings.
           - instruction: a string instruction guiding the encoding.
        """
        super(NVEmbedEncoder, self).__init__()
        self.model = AutoModel.from_pretrained("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models", trust_remote_code=trust_remote_code)
    
    def encode(self, texts, instruction):
        """
        Encodes a list of texts using the provided instruction.
        In practice, the nv-embed-v2 model (via its remote code) should use
        the instruction to process the input texts accordingly.
        
        Args:
            texts (List[str]): List of input strings.
            instruction (str): Instruction prompt for encoding.
        
        Returns:
            Tensor: A tensor of shape (batch_size, hidden_dim) representing the embeddings.
        """
        # Call the model's encode method (assuming this is implemented in the remote code)
        embeddings = self.model.encode(texts, instruction=instruction)
        return embeddings

class NVEmbedStanceClassifier(nn.Module):
    def __init__(self, 
                 model_name="nvidia/nv-embed-v2", 
                 num_labels=3, 
                 doc_instruction="", 
                 wiki_instruction="", 
                 dropout_rate=0.1):
        """
        Initializes the stance classifier.
        
        Args:
            model_name (str): Pretrained model identifier.
            num_labels (int): Number of stance labels (e.g., favor, against, neutral).
            doc_instruction (str): Instruction to encode the document–topic pair.
            wiki_instruction (str): Instruction to encode the Wikipedia text.
            dropout_rate (float): Dropout rate for the fused embeddings.
        """
        super(NVEmbedStanceClassifier, self).__init__()
        # Initialize our encoder using nv-embed-v2.
        self.encoder = NVEmbedEncoder(model_name, trust_remote_code=True)
        self.doc_instruction = doc_instruction
        self.wiki_instruction = wiki_instruction
        
        self.dropout = nn.Dropout(dropout_rate)
        # We assume nv-embed-v2 returns embeddings of dimension 4096.
        # The two embeddings (document and wiki) are concatenated.
        self.classifier = nn.Linear(4096 * 2, num_labels)

    def forward(self, doc_texts, wiki_texts):
        """
        Forward pass:
          1. Encode the document–topic pair using the document instruction.
          2. Encode the Wikipedia text using the wiki instruction.
          3. Normalize both embeddings.
          4. Concatenate and apply dropout.
          5. Pass through a linear layer for classification.
        
        Args:
            doc_texts (List[str]): List of preformatted document–topic strings.
            wiki_texts (List[str]): List of preformatted Wikipedia texts.
        
        Returns:
            Tensor: Logits over stance labels.
        """
        # Encode document and topic with its instruction.
        doc_embeddings = self.encoder.encode(doc_texts, instruction=self.doc_instruction)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        
        # Encode the Wikipedia texts with its instruction.
        wiki_embeddings = self.encoder.encode(wiki_texts, instruction=self.wiki_instruction)
        wiki_embeddings = F.normalize(wiki_embeddings, p=2, dim=1)
        
        # Concatenate the two embeddings.
        fused_embeddings = torch.cat([doc_embeddings, wiki_embeddings], dim=1)
        fused_embeddings = self.dropout(fused_embeddings)
        
        # Classify the fused representation.
        logits = self.classifier(fused_embeddings)
        return logits
