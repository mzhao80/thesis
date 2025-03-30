#!/usr/bin/env python
"""
Topic Generation - First Level

This script generates first-level subtopics for each document in the dataset using a fine-tuned BART model.
It takes the original documents and their policy areas as input and produces subtopics that are more specific
than the general policy areas but still broad enough to be further subdivided.

The script:
1. Loads a fine-tuned BART model for topic generation
2. Processes each document using this model to extract a subtopic
3. Handles missing or empty predictions through regeneration
4. Outputs the results to a CSV file for subsequent clustering

Output:
- step_1.csv: Original data plus a 'pred_subtopic' column containing model-generated subtopics
"""

import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

def main():
    """
    Main function to generate first-level subtopics for the taxonomy.
    """
    # Define paths.
    model_dir = "./lora_bart_topics_2023"  # Directory where the trained model was saved.
    data_file = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv"
    
    # Load the fine-tuned model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Load the entire CSV as a dataset (this retains all original columns).
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files={"data": data_file})["data"]
    
    # Define the preprocessing function that formats inputs for the model
    def preprocess_function(examples):
        """
        Prepare the input text for the model by combining policy area and document text.
        
        Args:
            examples: Batch of examples containing policy_area and document columns
            
        Returns:
            Dictionary with tokenized inputs
        """
        inputs = [
            f"Extract a short subtopic of the parent topic, {pa}, from the following speech: {doc}"
            for pa, doc in zip(examples["policy_area"], examples["document"])
        ]
        model_inputs = tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        return model_inputs
    
    # Preprocess the dataset for model inference.
    print("Preprocessing dataset...")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Set format to PyTorch tensors for the DataLoader.
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create a DataLoader for batch processing
    dataloader = DataLoader(encoded_dataset, batch_size=8)
    
    # Prepare model for evaluation.
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Generate predictions in batches
    print("Generating predictions...")
    all_preds = []
    for batch in tqdm(dataloader, desc="Generating predictions"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=3,
            max_length=16,
            length_penalty=4.0,
            early_stopping=True
        )
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(preds)
    
    # Convert the original dataset to a pandas DataFrame.
    df = pd.DataFrame(dataset)
    
    # Add a new column for the predicted subtopics.
    df["pred_subtopic"] = all_preds

    # Add each document's index
    df["matching_idx"] = df.index

    # Handle missing or empty predictions
    print("Checking for and regenerating empty predictions...")
    max_retries = 5
    for idx, row in tqdm(df.iterrows(), desc="Regeneration check"):
        retries = 0
        while retries < max_retries and (pd.isna(row["pred_subtopic"]) or row["pred_subtopic"].strip() == ""):
            # Create individual prompt for this document
            prompt = f"Extract a short subtopic of the parent topic, {row['policy_area']}, from the following speech: {row['document']}"
            encoded_input = tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            # Move the inputs to the same device as the model.
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Generate prediction
            outputs = model.generate(
                **encoded_input,
                min_length=3,
                max_length=16,
                length_penalty=4.0,
                early_stopping=True
            )
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            df.at[idx, "pred_subtopic"] = pred
            print(f"Regenerated prediction for row {idx}: {pred}")
            retries += 1
            
        # If still failed after max retries, use a default value
        if retries == max_retries and (pd.isna(df.at[idx, "pred_subtopic"]) or df.at[idx, "pred_subtopic"].strip() == ""):
            df.at[idx, "pred_subtopic"] = "Misc."
            print(f"Failed to regenerate prediction for row {idx} after {max_retries} attempts.")
    
    # Save the final CSV with all original columns plus the predictions.
    output_csv = "step_1.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
