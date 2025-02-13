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
import wandb

def main():
    # Initialize a new wandb run.
    wandb.init(project="lora_bart_topic_extraction_sweep")
    config = wandb.config

    # Get hyperparameters from wandb.config.
    r = config.r  # LoRA rank
    alpha_multiplier = config.alpha_multiplier
    lora_alpha = r * alpha_multiplier  # effective scaling factor
    # Convert target_modules string to a list.
    target_modules = [s.strip() for s in config.target_modules.split(",")]
    learning_rate = config.learning_rate
    num_train_epochs = config.num_train_epochs
    length_penalty = config.length_penalty
    warmup_steps = config.warmup_steps
    dropout = config.dropout

    # 1. Load CSV data and create a train/validation split.
    data_files = {"data": config.data_file}
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset["data"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # 2. Load the tokenizer.
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Preprocessing: combine fields and tokenize both input and target.
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
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["subtopic"],
                max_length=16,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        if "subtopic" in examples:
            model_inputs["target"] = examples["subtopic"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["document", "policy_area", "subtopic"]
    )
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, remove_columns=["document", "policy_area"]
    )

    # 4. Create a data collator.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    # 5. Load the pre-trained BART model and apply LoRA.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)

    # 6. Setup training arguments with wandb reporting.
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="steps",
        logging_steps=500,
        predict_with_generate=True,
        fp16=True,
        learning_rate=learning_rate,
        report_to=["wandb"],
        save_strategy="no"
    )

    # 7. Initialize the Trainer.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Train the model.
    trainer.train()

    # 9. Generate predictions on the validation set.
    print("Generating predictions on validation dataset...")
    predictions_output = trainer.predict(
        val_dataset,
        length_penalty=length_penalty,
        max_length=16,
        early_stopping=True
    )
    preds = predictions_output.predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    target_texts = val_dataset["target"]
    results_df = pd.DataFrame({
        "target": target_texts,
        "generated": decoded_preds,
    })

    output_csv = os.path.join(config.output_dir, f"eval_results_{r}_{lora_alpha}_{len(target_modules)}_{learning_rate}_{num_train_epochs}_{length_penalty}_{dropout}.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Saved evaluation results to {output_csv}")

    wandb.log({"final_eval": predictions_output.metrics})
    wandb.finish()

if __name__ == "__main__":
    main()
