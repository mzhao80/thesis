import torch
import torch.nn as nn
from transformers import AutoModel

class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model, n_layers_freeze=0, wiki_model=None, n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model)
        # Optionally freeze early layers
        if n_layers_freeze > 0:
            for layer in self.bert.encoder.layer[:n_layers_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        # For multi-target stance detection, we output 3 predictions (one per topic)
        self.classifier = nn.Linear(hidden_size, 3 * num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, input_ids_wiki=None, attention_mask_wiki=None):
        # The input encodes the joint text (document + three targets) and the wiki text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # shape: (batch, hidden_size)
        logits = self.classifier(pooled_output)  # shape: (batch, 3*num_labels)
        # Reshape logits so that each of the 3 targets gets its own prediction vector
        logits = logits.view(-1, 3, self.num_labels)
        return logits
