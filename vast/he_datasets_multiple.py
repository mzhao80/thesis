import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = '0'
from transformers import AutoTokenizer

# Multi-target version: group by post and collect up to three topics (with corresponding wiki summaries and labels)
class VASTZeroFewShotMultiTarget(Dataset):
    def __init__(self, phase, model='bert-base-uncased', wiki_model='bert-base-uncased'):
        path = 'zero-shot-stance/data/VAST'
        if phase in ['train', 'test']:
            file_path = f'{path}/vast_{phase}.csv'
        else:
            file_path = f'{path}/vast_dev.csv'
        df = pd.read_csv(file_path)
        print(f'# of {phase} examples (rows): {df.shape[0]}')

        # Group by the post column
        grouped = df.groupby('post')

        # Load wiki summaries dictionary
        wiki_dict = pickle.load(open(f'{path}/wiki_dict.pkl', 'rb'))

        self.data = []
        for post, group in grouped:
            # Use the first processed text as the document
            doc_text = group['text_s'].iloc[0]
            # Extract topics, labels, and wiki summaries (via new_topic)
            topics = group['topic_str'].tolist()
            labels = group['label'].tolist()
            wiki_summaries = group['new_topic'].map(wiki_dict).tolist()
            if phase == 'test':
                few_shot = group['seen?'].tolist()
            else:
                few_shot = [0] * len(topics)

            # Record the original number of topics
            original_count = len(topics)
            valid_mask = [1] * original_count

            # Ensure exactly three targets: if less than 3, repeat available values and mark them as padded (0)
            if original_count < 3:
                while len(topics) < 3:
                    topics.append(topics[0])
                    labels.append(labels[0])
                    wiki_summaries.append(wiki_summaries[0])
                    few_shot.append(few_shot[0])
                    valid_mask.append(0)
            # If there are more than 3 topics, take only the first three (all considered valid)
            elif original_count > 3:
                topics = topics[:3]
                labels = labels[:3]
                wiki_summaries = wiki_summaries[:3]
                few_shot = few_shot[:3]
                valid_mask = [1, 1, 1]

            # Create a joint input string combining the document with the three topics.
            joint_input = f"text: {doc_text} targets: {topics[0]}, {topics[1]}, {topics[2]}"
            # Concatenate all wiki summaries with new lines between each article.
            combined_wiki = "\n".join(wiki_summaries)
            self.data.append({
                'joint_input': joint_input,
                'combined_wiki': combined_wiki,
                'stances': labels,         # list of 3 stance labels
                'few_shot': few_shot,       # list of 3 few-shot flags
                'valid_mask': valid_mask,   # list of 3 flags (1 if genuine, 0 if padded)
            })

        print(f'# of grouped examples: {len(self.data)}')

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Tokenize the joint input together with the combined wiki text
        joint_inputs = [d['joint_input'] for d in self.data]
        wiki_texts = [d['combined_wiki'] for d in self.data]
        encodings = self.tokenizer(joint_inputs, wiki_texts, padding=True, truncation=True, return_token_type_ids=True)

        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        self.token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)
        # Convert lists of stances, few_shot flags, and valid masks into tensors (each with shape: [num_examples, 3])
        self.stances = torch.tensor([d['stances'] for d in self.data], dtype=torch.long)
        self.few_shot = torch.tensor([d['few_shot'] for d in self.data], dtype=torch.long)
        self.valid_mask = torch.tensor([d['valid_mask'] for d in self.data], dtype=torch.long)

        print(f'max len: {self.input_ids.shape[1]}')

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'stances': self.stances[index],         # (3,)
            'few_shot': self.few_shot[index],         # (3,)
            'valid_mask': self.valid_mask[index],     # (3,)
            # The original code provided separate wiki inputs; here the wiki text is encoded jointly.
            'input_ids_wiki': torch.tensor([0]),      # Placeholder (not used)
            'attention_mask_wiki': torch.tensor([0]),   # Placeholder (not used)
        }

    def __len__(self):
        return len(self.data)

def data_loader(data, phase, topic, batch_size, model='bert-base-uncased', wiki_model='bert-base-uncased', n_workers=4):
    shuffle = True if phase == 'train' else False
    dataset = VASTZeroFewShotMultiTarget(phase, model=model, wiki_model=wiki_model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    return loader
