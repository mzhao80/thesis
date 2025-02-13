import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class DocTopicPhraseDataset(Dataset):
    """
    Dataset for document–policy_area–subtopic–phrase triples.
    
    Expects a CSV with columns:
      - "document": The full text of a document.
      - "policy_area": The parent policy area (e.g., "Political Issue", "Economics and Public Finance", etc.) from CRS.
      - "subtopic": (Not currently used in training - should be used later for cluster or subtopic discovery validation)
      - "phrase": The target phrase.
    
    For this training procedure:
      - The dataset groups rows by document and uses the first row for each document.
      - This ensures that each document appears only once.
    """
    def __init__(self, csv_file):
        # Read CSV file into a DataFrame, with every column as a string.
        self.df = pd.read_csv(csv_file)
        # drop token length
        self.df = self.df.drop(columns=["token_length"])
        self.df["subtopic"] = self.df["subtopic"].str.lower()
        # cast phrase as string
        self.df["phrase"] = self.df["phrase"].astype(str)
        # Initialize the NV-Embed-v2 tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        # Build a mapping from policy area to a unique index.
        unique_parents = self.df["policy_area"].unique().tolist()
        self.topic_to_index = {topic: idx for idx, topic in enumerate(unique_parents)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Retrieve the row corresponding to idx.
        row = self.df.iloc[idx]
        #document = row["document"]
        instruction = "Instruct: Encode the following speech from Congress for stance detection of policy topics.\nDocument: "
        document = instruction + row["document"]
        policy = row["policy_area"]
        subtopic = row["subtopic"]
        # Preprocess the phrase by stripping any surrounding quotation marks.
        phrase = row["phrase"].strip('"')
        
        # Tokenize the document, the subtopic, and the phrase without truncation or padding.
        doc_enc = self.tokenizer(document, padding=False, truncation=False, return_tensors="pt")
        sub_enc = self.tokenizer(subtopic, padding=False, truncation=False, return_tensors="pt")
        phrase_enc = self.tokenizer(phrase, padding=False, truncation=False, return_tensors="pt")
        
        # Map the policy area to its corresponding index.
        topic_idx = self.topic_to_index.get(policy)
        if topic_idx is None:
            print(idx, document, policy, phrase)
            topic_idx = len(self.topic_to_index)
            self.topic_to_index[policy] = topic_idx
        
        return {
            "doc_input_ids": doc_enc["input_ids"].squeeze(0),
            "doc_attention_mask": doc_enc["attention_mask"].squeeze(0),
            "subtopic": subtopic,
            "subtopic_input_ids": sub_enc["input_ids"].squeeze(0),
            "phrase": phrase,  # Raw target phrase (for debugging).
            "phrase_input_ids": phrase_enc["input_ids"].squeeze(0),
            "topic_idx": topic_idx,
            "policy": policy
        }

def collate_fn_with_tokenizer(tokenizer):
    """
    Collate function to vectorize a batch.
    
    Since each document now appears only once with one candidate phrase,
    this function pads:
      - Document input_ids,
      - Document attention masks, 
      - Subtopic input_ids, and
      - Phrase input_ids.
      
    Returns:
      Tuple: (list of padded doc_input_ids, padded doc_attention_mask, 
              padded subtopic_input_ids, padded phrase_input_ids, topic_idxs, policies)
    """
    
    def _collate_fn(batch):
        doc_input_ids_list = [item["doc_input_ids"] for item in batch]
        doc_attention_mask_list = [item["doc_attention_mask"] for item in batch]
        subtopic_input_ids_list = [item["subtopic_input_ids"] for item in batch]
        phrase_input_ids_list = [item["phrase_input_ids"] for item in batch]
        topic_idxs = torch.tensor([item["topic_idx"] for item in batch], dtype=torch.long)
        policies = [item["policy"] for item in batch]
        
        docs_input_ids = pad_sequence(doc_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        docs_attention_mask = pad_sequence(doc_attention_mask_list, batch_first=True, padding_value=0)
        subtopic_input_ids = pad_sequence(subtopic_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        phrase_input_ids = pad_sequence(phrase_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        return docs_input_ids, docs_attention_mask, subtopic_input_ids, phrase_input_ids, topic_idxs, policies
    return _collate_fn
