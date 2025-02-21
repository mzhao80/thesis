import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

def main():
    # Define paths.
    model_dir = "./lora_bart_topics"  # Directory where the trained model was saved.
    data_file = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Load the entire CSV as a dataset (this retains all original columns).
    dataset = load_dataset("csv", data_files={"data": data_file})["data"]
    
    # Define the preprocessing function used during training.
    def preprocess_function(examples):
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
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Set format to PyTorch tensors for the DataLoader.
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create a DataLoader (adjust the batch size if needed).
    dataloader = DataLoader(encoded_dataset, batch_size=8)
    
    # Prepare model for evaluation.
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    # Generate predictions in batches.
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

    # Loop through rows where pred_subtopic is missing or empty and try to regenerate.
    max_retries = 5
    for idx, row in tqdm(df.iterrows(), desc="regeneration check"):
        retries = 0
        while retries < max_retries and (pd.isna(row["pred_subtopic"]) or row["pred_subtopic"].strip() == ""):
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
        if retries == max_retries:
            df.at[idx, "pred_subtopic"] = "Misc."
            print(f"Failed to regenerate prediction for row {idx} after {max_retries} attempts.")
    
    # Save the final CSV with all original columns plus the predictions.
    output_csv = "step_1.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
