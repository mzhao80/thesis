import os
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gc
from openai import OpenAI
import api_keys

def main():
    # Define paths.
    taxonomy_file = "taxonomy_output.csv"          # Taxonomy with gpt_label and source_indices.
    original_file = "all_data_predictions.csv"       # Original CSV containing the "document" column.
    model_dir = "./lora_bart_subtopics"              # Directory where the trained BART model was saved.
    
    # Load the taxonomy and original CSV.
    taxonomy_df = pd.read_csv(taxonomy_file)
    original_df = pd.read_csv(original_file)
    
    # Load the fine-tuned BART model and its tokenizer.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).half()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Prepare the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    
    # Iterate over each taxonomy row.
    for idx, row in taxonomy_df.iterrows():
        gpt_label = row["gpt_label"]  # This is our new parent topic.
        source_indices_str = row["source_indices"]  # Semicolon-separated indices.
        
        # Parse source indices into a list of integers.
        indices = [int(i.strip()) for i in source_indices_str.split(";") if i.strip()]
        
        # For each corresponding document, process individually.
        for doc_idx in indices:
            doc_text = original_df.loc[doc_idx, "document"]
            prompt = f"Extract a short subtopic of the parent topic, {gpt_label}, from the following text: {doc_text}"
            
            # Tokenize and move to device.
            encoded = tokenizer(prompt, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
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
            
            results.append({
                "policy_area": row["policy_area"],
                "gpt_label": gpt_label,
                "document_index": doc_idx,
                "new_subtopic": pred
            })
    
    # Save the results to CSV.
    new_df = pd.DataFrame(results)
    new_df.to_csv("new_subtopics_per_document.csv", index=False)
    print("Saved new subtopics per document to new_subtopics_per_document.csv")

if __name__ == "__main__":
    main()
