"""
StrideStance Model Architecture Implementation

This module implements the core model architecture for the StrideStance approach,
adapted from He et al.'s (2022) knowledge-enhanced stance detection models. It focuses on:

1. Implementing a BERT-based sequence classification model with knowledge enhancement
2. Supporting integration of external knowledge for improved topic understanding
3. Providing the neural network foundation for the stance detection task
4. Enabling transfer learning from pre-trained language models

Key features:
- BERT-based sequence classification with stance prediction (against, favor, neutral)
- Adaptation of pre-trained sentence transformer models
- Knowledge integration mechanisms from the original He et al. implementation
- Support for document-level stance detection

This architecture serves as the backbone of the StrideStance system, providing the
neural network foundation for both the training and inference pipelines. While based
on He et al.'s work, it has been adapted to better handle long documents through
integration with the document splitting and aggregation components.
"""

import torch.nn as nn
import torch
import os


class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model='bert-base', n_layers_freeze=0, wiki_model='', n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model)

        n_layers = 12
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.bert_wiki = self.bert

        n_layers = 12
        if n_layers_freeze_wiki > 0:
            n_layers_ft = n_layers - n_layers_freeze_wiki
            for param in self.bert_wiki.parameters():
                param.requires_grad = False
            for param in self.bert_wiki.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert_wiki.encoder.layer[i].parameters():
                    param.requires_grad = True

        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden = config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                input_ids_wiki=None, attention_mask_wiki=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        if input_ids_wiki is not None:
            outputs_wiki = self.bert_wiki(input_ids_wiki, attention_mask=attention_mask_wiki)
            pooled_output_wiki = outputs_wiki.pooler_output
            pooled_output_wiki = self.dropout(pooled_output_wiki)
            pooled_output = torch.cat((pooled_output, pooled_output_wiki), dim=1)
        logits = self.classifier(pooled_output)
        return logits