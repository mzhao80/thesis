import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch
import numpy as np
import random

def main():
    # 1. Load CSV data and create a train/validation split (80/20 split)
    data_files = {"data": "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/new_training_data.csv"}
    dataset = load_dataset("csv", data_files=data_files)
    
    # Use the entire CSV as a single dataset and then split it
    dataset = dataset["data"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    # 2. Load the tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Preprocessing function: format input and tokenize both input and target.
    #    Also, for evaluation, we store the original subtopic text as "target".
    def preprocess_function(examples):
        # Combine the policy_area and document into one prompt string.
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
        
        # Tokenize the target (subtopic)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["subtopic"],
                max_length=16,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        
        # Save original target text for evaluation purposes.
        if "subtopic" in examples:
            model_inputs["target"] = examples["subtopic"]
        return model_inputs

    # For training, remove all extra columns.
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["document", "policy_area", "subtopic"]
    )
    # For validation, keep the target text by only removing document and policy_area.
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, remove_columns=["document", "policy_area"]
    )
    
    # Optionally, use a data collator for dynamic padding (here we already padded during tokenization).
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    
    # 4. Load pre-trained BART model and apply LoRA.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # Define LoRA configuration. Here we update the query and value projection matrices.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,      # Scaling factor
        lora_dropout=0.1,   # Dropout probability for LoRA layers
        target_modules=target_modules
    )
    
    # Wrap the model with LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # 5. Setup training arguments.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lora_bart_subtopics_new",
        overwrite_output_dir=True,
        learning_rate=1e-4,
        num_train_epochs=5,
        warmup_steps=500,            # Added warmup steps for smoother LR ramp-up
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=500,
        predict_with_generate=True,
        fp16=True,                   # Enable mixed precision if your GPU supports it
        save_total_limit=2,
    )
    
    # 6. Initialize the trainer.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Start fine-tuning.
    trainer.train()

    trainer.save_model("./lora_bart_subtopics")
    
    # 8. After training, generate predictions on the validation set with generation parameters.
    print("Generating predictions on validation dataset...")
    predictions_output = trainer.predict(
        val_dataset,
        length_penalty=4.0,    # Apply length penalty to encourage shorter sequences
        min_length=3,
        max_length=16,
        early_stopping=True    # Stop early if possible
    )

    # Decode generated predictions.
    preds = predictions_output.predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Retrieve target texts from the validation dataset.
    # (We stored the original subtopic text in the "target" key.)
    target_texts = val_dataset["target"]
    
    # Create a DataFrame and output to CSV.
    results_df = pd.DataFrame({
        "target": target_texts,
        "generated": decoded_preds,
    })
    output_csv = f"eval_results/results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved evaluation results to {output_csv}")

if __name__ == "__main__":
    main()
