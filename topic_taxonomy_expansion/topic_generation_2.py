import os
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.auto import tqdm

def main():
    # Define paths.
    taxonomy_file = "step_2.csv"           # Taxonomy with gpt_label and source_indices.
    original_file = "step_1.csv"       # Original CSV containing the "document" column.
    model_dir = "./lora_bart_topics"              # Directory where the trained BART model was saved.
    
    # Load the taxonomy and original CSV.
    taxonomy_df = pd.read_csv(taxonomy_file)
    original_df = pd.read_csv(original_file)
    
    # Load the fine-tuned BART model and its tokenizer.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Prepare the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    count = 0
    max_retries = 5
    
    # Iterate over each taxonomy row.
    for idx, row in tqdm(taxonomy_df.iterrows(), desc="Processing taxonomy", total=len(taxonomy_df)):
        gpt_label = row["gpt_label"]  # This is our new parent topic.
        source_indices_str = row["source_indices"]  # Semicolon-separated indices.
        
        # Parse source indices into a list of integers.
        indices = [int(i) for i in source_indices_str.split(";")]
        
        # For each corresponding document, process individually.
        for doc_idx in indices:
            count += 1
            if gpt_label == "Misc.":
                results.append({
                    "document_index": doc_idx,
                    "policy_area": row["policy_area"],
                    "subtopic_1": gpt_label,
                    "subtopic_2": "Misc."
                })
                continue

            doc_text = original_df.loc[doc_idx, "document"]
            prompt = f"Extract a detailed subtopic of the parent topic, {gpt_label}, that represents the main policy issue in the following text: {doc_text}"
            
            # Tokenize and move to device.
            encoded = tokenizer(prompt, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            retries = 0
            pred = ""
            while pred.strip() == "" and retries < max_retries:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        min_length=3,
                        max_length=16,
                        length_penalty=4.0,
                        early_stopping=True
                    )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                retries += 1

            if retries == max_retries:
                pred = "Misc."
            
            results.append({
                "document_index": doc_idx,
                "policy_area": row["policy_area"],
                "subtopic_1": gpt_label,
                "subtopic_2": pred
            })
    
    # Save the results to CSV.
    new_df = pd.DataFrame(results)
    output_csv = "step_3.csv"
    new_df.to_csv(output_csv, index=False)
    print(f"Processed {count} documents.")
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
