import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch

# Read in the CSV file containing the congressional speeches
csv_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv'
df = pd.read_csv(csv_path)

# Initialize the tokenizer and summarization pipeline with the facebook/bart-large-cnn model.
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", model_max_length=1024)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, tokenizer = tokenizer)

def get_token_count(text):
    """
    Returns the number of tokens in the given text using the model's tokenizer.
    """
    try:
        # Add special tokens if necessary
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Error tokenizing document: {e}")
        return 0

def summarize_document(text):
    """
    Generate a summary for the provided text with a maximum of 384 tokens.
    """
    try:
        summary = summarizer(text, max_length=384, truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error processing document: {e}")
        return ""

# Apply the token count function to create the 'original_length' column.
df['original_length'] = df['document'].apply(lambda x: get_token_count(x) if isinstance(x, str) else 0)

# Apply the summarization to the 'document' column.
df['document_long'] = df['document']
df['document'] = df['document'].apply(lambda x: summarize_document(x) if isinstance(x, str) else "")

# Save the resulting DataFrame with summaries and token lengths to a new CSV file.
output_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data_with_summaries.csv'
df.to_csv(output_path, index=False)

print(f"Summaries and original token lengths successfully saved to {output_path}")
