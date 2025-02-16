import torch.nn as nn
import torch
import os


from transformers import AutoModel

class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model='bert-base-uncased', n_layers_freeze=0, wiki_model='', n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Load the main model and determine total layers
        if model == 'bert-base-uncased':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            n_layers = 12
        elif model == 'sentence-transformers/all-MiniLM-L6-v2':
            self.bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            n_layers = 6
        elif model == 'sentence-transformers/all-mpnet-base-v2':
            self.bert = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            n_layers = 12
        else:
            raise ValueError("Unsupported model. Use 'bert-base-uncased' or 'sentence-transformers/all-MiniLM-L6-v2' or 'sentence-transformers/all-mpnet-base-v2'.")
        
        # Freeze layers as specified
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            # Freeze all parameters first
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze pooler parameters
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            # Unfreeze the last n_layers_ft encoder layers
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # For wiki_model, if provided and different from the main model, load accordingly
        if wiki_model and wiki_model != model:
            if wiki_model == 'bert-base-uncased':
                self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')
                wiki_layers = 12
            elif wiki_model == 'sentence-transformers/all-MiniLM-L6-v2':
                self.bert_wiki = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                wiki_layers = 6
            elif wiki_model == 'sentence-transformers/all-mpnet-base-v2':
                self.bert_wiki = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
                wiki_layers = 12
            else:
                raise ValueError("Unsupported wiki model. Use 'bert-base-uncased' or 'sentence-transformers/all-MiniLM-L6-v2' or 'sentence-transformers/all-mpnet-base-v2'.")
            
            if n_layers_freeze_wiki > 0:
                n_layers_ft_wiki = wiki_layers - n_layers_freeze_wiki
                for param in self.bert_wiki.parameters():
                    param.requires_grad = False
                for param in self.bert_wiki.pooler.parameters():
                    param.requires_grad = True
                for i in range(wiki_layers - 1, wiki_layers - 1 - n_layers_ft_wiki, -1):
                    for param in self.bert_wiki.encoder.layer[i].parameters():
                        param.requires_grad = True
        else:
            # If no separate wiki model is specified, reuse the main model.
            self.bert_wiki = self.bert

        # Set up the classifier using the hidden size from the configuration.
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden = config.hidden_size
        self.classifier = nn.Linear(hidden * (2 if wiki_model and wiki_model != model else 1), num_labels)
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