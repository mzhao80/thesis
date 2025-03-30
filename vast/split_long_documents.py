"""
Document Splitting Utility for StrideStance

This module implements document splitting functionality for the StrideStance system,
which is essential for handling long documents that exceed the token limit of
transformer models. The implementation focuses on:

1. Splitting documents at natural sentence boundaries to preserve coherence
2. Creating overlapping chunks to avoid losing context at chunk boundaries
3. Tracking document metadata to associate chunks with their source documents
4. Estimating combined token lengths to ensure compatibility with model constraints

Key features:
- Sentence boundary detection using NLTK
- Overlapping chunk creation with configurable overlap
- Handling of extremely long sentences by word-level splitting when necessary
- Preservation of document metadata (speakers, subtopics, etc.)
- Flexible tokenizer support through Hugging Face Transformers

This utility is designed to work with the rest of the StrideStance pipeline,
preparing documents for stance detection while maintaining document-chunk
relationships for post-processing and aggregation.
"""

import pandas as pd
import numpy as np
import re
import nltk
from transformers import AutoTokenizer
import argparse
import os

# Download nltk punkt for sentence tokenization if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description='Split long documents for stance detection')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2', 
                        help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=384, 
                        help='Maximum token length per document chunk (default: 384 to leave room for target and summary)')
    parser.add_argument('--overlap', type=int, default=50, 
                        help='Number of tokens to overlap between chunks')
    parser.add_argument('--input_csv', type=str, 
                        default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, 
                        default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_split.csv',
                        help='Path to output CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed warnings about document lengths')
    return parser.parse_args()

def split_document_at_sentence_boundaries(document, tokenizer, max_length, overlap):
    """
    Split a document into chunks of max_length tokens with overlap,
    breaking at sentence boundaries when possible.
    
    Note: max_length should be set to 384 to leave room for the target and summary
    which will be added later in the dataloader (up to 512 total tokens).
    """
    # Ensure document is a valid string
    if document is None:
        return [""]
    
    document = str(document)
    if document == "nan" or document == "":
        return [""]
    
    # Tokenize the document first to get total token count
    tokens = tokenizer.encode(document, add_special_tokens=True)
    
    # If document is already under the limit, return it as is
    if len(tokens) <= max_length:
        return [document]
    
    # Tokenize document into sentences
    sentences = nltk.sent_tokenize(document)
    
    chunks = []
    current_chunk = []
    current_length = 0
    last_sentence = None
    
    for sentence in sentences:
        # Get token count for this sentence
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        
        # If a single sentence is longer than max_length, we'll need to split it arbitrarily
        if sentence_length > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split the long sentence into chunks
            words = sentence.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                word_length = len(word_tokens)
                
                if temp_length + word_length + 1 <= max_length:  # +1 for space
                    temp_chunk.append(word)
                    temp_length += word_length + 1
                else:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_length
            
            if temp_chunk:
                current_chunk = temp_chunk
                current_length = temp_length
            
        # Normal case: adding a new sentence to the current chunk
        elif current_length + sentence_length + 1 <= max_length:  # +1 for space
            current_chunk.append(sentence)
            current_length += sentence_length + 1
            last_sentence = sentence
        else:
            # Save the current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # If we need to create overlap, add the last sentence to the new chunk
            if overlap > 0 and last_sentence:
                last_tokens = tokenizer.encode(last_sentence, add_special_tokens=False)
                if len(last_tokens) <= overlap:
                    current_chunk = [last_sentence, sentence]
                    current_length = len(last_tokens) + sentence_length + 1
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk = [sentence]
                current_length = sentence_length
            
            last_sentence = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def estimate_combined_length(document, target, summary, tokenizer):
    """
    Estimate the combined token length of document+target+summary.
    This is an approximation as the exact token count depends on the tokenizer implementation.
    
    Args:
        document: The document text
        target: The target/topic text
        summary: The summary text
        tokenizer: The tokenizer to use
    """
    # Ensure all inputs are strings
    document = str(document) if document is not None else ""
    target = str(target) if target is not None else ""
    summary = str(summary) if summary is not None else ""
    
    # Check for NaN values (which can't be encoded)
    if document == "nan":
        document = ""
    if target == "nan":
        target = ""
    if summary == "nan":
        summary = ""
    
    # Get token counts for each component
    doc_tokens = tokenizer.encode(document, add_special_tokens=False) if document else []
    target_tokens = tokenizer.encode(target, add_special_tokens=False) if target else []
    summary_tokens = tokenizer.encode(summary, add_special_tokens=False) if summary else []
    
    # Format: [CLS] "text: {document} target: {target}" [SEP] summary [SEP]
    # Need to account for the "text: " and " target: " strings and special tokens
    text_prefix_tokens = tokenizer.encode("text: ", add_special_tokens=False)
    target_prefix_tokens = tokenizer.encode(" target: ", add_special_tokens=False)
    
    return len(doc_tokens) + len(text_prefix_tokens) + len(target_prefix_tokens) + len(target_tokens) + len(summary_tokens) + 3

def main():
    args = parse_args()
    
    print(f"Loading tokenizer: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    print(f"Reading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        original_count = len(df)
        print(f"Original document count: {original_count}")
        
        if original_count == 0:
            print("Warning: Input CSV is empty. No documents to process.")
            return

        # drop rows where target is nan
        df = df.dropna(subset=['target'])
        new_count = len(df)
        print(f"New document count: {new_count}")
            
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return
    
    # Create a new dataframe to hold the split documents
    new_rows = []
    long_docs_count = 0
    new_docs_count = 0
    error_count = 0
    
    # Determine column names based on format
    target_col = 'target'  
    summary_col = 'summary'
    
    print("Processing documents...")
    for idx, row in df.iterrows():
        try:
            document = row['document'] if 'document' in row and not pd.isna(row['document']) else ""
            target = row[target_col] if target_col in row and not pd.isna(row[target_col]) else ""
            summary = row[summary_col] if summary_col in row and not pd.isna(row[summary_col]) else ""
            
            # Skip if document is missing
            if pd.isna(document) or document == "":
                new_rows.append(row)
                continue
            
            # Calculate combined length for the full document
            doc_tokens = tokenizer.encode(document, add_special_tokens=True)
            combined_length = estimate_combined_length(document, target, summary, tokenizer)
            
            if combined_length > 512 or len(doc_tokens) > args.max_length:
                long_docs_count += 1
                # Split the document to keep each chunk <= max_length (384 by default)
                chunks = split_document_at_sentence_boundaries(
                    document, tokenizer, args.max_length, args.overlap
                )
                
                # Create new rows with the split documents
                for i, chunk in enumerate(chunks):
                    new_row = row.copy()
                    new_row['document'] = chunk
                    new_row['document_chunk'] = i + 1  # Add chunk number metadata
                    new_row['document_chunks_total'] = len(chunks)
                    
                    # Verify the chunk is within our desired token limit (debugging only)
                    if args.verbose:
                        chunk_tokens = tokenizer.encode(chunk, add_special_tokens=True)
                        chunk_combined_length = estimate_combined_length(chunk, target, summary, tokenizer)
                        
                        if len(chunk_tokens) > args.max_length:
                            print(f"Warning: Chunk {i+1} for document {idx} still exceeds max_length ({len(chunk_tokens)} > {args.max_length})")
                        
                        if chunk_combined_length > 512:
                            print(f"Warning: Combined length for chunk {i+1} of document {idx} may exceed 512 tokens ({chunk_combined_length})")
                    
                    new_rows.append(new_row)
                    new_docs_count += 1
            else:
                # Add chunk metadata to original documents too for consistency
                new_row = row.copy()
                new_row['document_chunk'] = 1
                new_row['document_chunks_total'] = 1
                new_rows.append(new_row)
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            error_count += 1
            # Still include the original row to avoid data loss
            new_rows.append(row)
    
    # Create new dataframe with all rows
    new_df = pd.DataFrame(new_rows)
    
    print(f"Found {long_docs_count} documents exceeding limits")
    print(f"Created {new_docs_count} new document chunks")
    print(f"Encountered {error_count} errors during processing")
    print(f"Final document count: {len(new_df)}")
    
    # Save the results
    try:
        print(f"Saving results to: {args.output_csv}")
        new_df.to_csv(args.output_csv, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
