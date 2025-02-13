import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model_joint import JointTopicExpanModel
from dataset_joint import DocTopicPhraseDataset, collate_fn
from torch.optim import AdamW
import gc
import logging
import sys
from datetime import datetime
import os

def main(config):
    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # ----------------------------
    # Data Preparation
    # ----------------------------
    full_dataset = DocTopicPhraseDataset(config["data"]["csv_file"])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")
    
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"],
                                  shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"],
                                shuffle=False, collate_fn=collate_fn)
    
    # ----------------------------
    # Model Initialization
    # ----------------------------
    model = JointTopicExpanModel(
        encoder_model_name=config["arch"]["doc_encoder_model"],
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config["arch"]["hidden_dim"],
        num_decoder_layers=config["arch"]["num_decoder_layers"],
        num_decoder_heads=config["arch"]["num_decoder_heads"],
        max_length=config["arch"]["max_phrase_length"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_checkpointing=False,
        enable_copy_mechanism=config["arch"].get("enable_copy_mechanism", False),
        device=device,
        dropout=config["arch"].get("dropout", 0.1)
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config["optimizer"]["lr"],
                      weight_decay=config["optimizer"]["weight_decay"])
    t_total = len(train_dataloader) * config["epochs"]
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    
    best_val_loss = float("inf")
    early_stop_patience = config["early_stopping"]["patience"]
    min_delta = config["early_stopping"]["min_delta"]
    current_patience = 0
    
    for epoch in range(1, config.get("epochs", 5) + 1):
        logging.info(f"\n[INFO] Starting Epoch {epoch}")
        epoch_loss_accum = 0.0
        epoch_batches = 0
        
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            documents, parent_topics, phrase_input_ids = batch
            phrase_input_ids = phrase_input_ids.to(device)
            optimizer.zero_grad()
            logits = model(documents, parent_topics, phrase_input_ids)
            gen_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                phrase_input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            loss = gen_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss_accum += loss.item()
            epoch_batches += 1
        
        avg_train_loss = epoch_loss_accum / epoch_batches
        logging.info(f"Epoch {epoch}: Average Training Loss = {avg_train_loss:.4f}")
        
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                documents, parent_topics, phrase_input_ids = batch
                phrase_input_ids = phrase_input_ids.to(device)
                logits = model(documents, parent_topics, phrase_input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    phrase_input_ids.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                val_loss_accum += loss.item()
                val_batches += 1
        avg_val_loss = val_loss_accum / val_batches
        logging.info(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss:.4f}")
        
        # ----------------------------
        # Debug Print for Validation Batch
        # ----------------------------
        try:
            # Retrieve one batch from the validation dataloader
            documents, parent_topics, phrase_input_ids = next(iter(val_dataloader))
            phrase_input_ids = phrase_input_ids.to(device)
            # Recompute the joint encoding for this batch
            joint_texts = [f"{model.joint_prompt}{doc} || {parent}" for doc, parent in zip(documents, parent_topics)]
            with torch.no_grad():
                global_embed, token_embeddings, input_ids = model.joint_encoder(joint_texts)
                memory = global_embed.unsqueeze(1)
                
                # Compute logits for the target (validation) phrase.
                if model.phrase_decoder.enable_copy_mechanism:
                    target_logits = model.phrase_decoder(phrase_input_ids, memory, token_embeddings, input_ids)
                else:
                    target_logits = model.phrase_decoder(phrase_input_ids, memory)
                
                # Generate a phrase for the same batch.
                generated_ids = model.generate_phrase(documents, parent_topics)
                if model.phrase_decoder.enable_copy_mechanism:
                    generated_logits = model.phrase_decoder(generated_ids, memory, token_embeddings, input_ids)
                else:
                    generated_logits = model.phrase_decoder(generated_ids, memory)
                
                # Decode the first sample for debugging.
                target_phrase_decoded = tokenizer.decode(phrase_input_ids[0], skip_special_tokens=True)
                generated_phrase_decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # For brevity, print a summary of the logits (first token's first 16 values).
                target_logits_sample = target_logits[0,0,:16].detach().cpu().numpy()
                generated_logits_sample = generated_logits[0,0,:16].detach().cpu().numpy()
                
                logging.info("=== Validation Debug Info ===")
                logging.info(f"Target Phrase Tokens: {phrase_input_ids[0]}")
                logging.info(f"Target Phrase: {target_phrase_decoded}")
                logging.info(f"Target Logits (first token, first 16 values): {target_logits_sample}")
                logging.info(f"Generated Phrase Tokens: {generated_ids[0]}")
                logging.info(f"Generated Phrase: {generated_phrase_decoded}")
                logging.info(f"Generated Logits (first token, first 16 values): {generated_logits_sample}")
                logging.info("=== End Debug Info ===")
        except Exception as e:
            logging.info("Error during validation debug generation: " + str(e))
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            current_patience = 0
        else:
            current_patience += 1
            logging.info(f"No improvement for {current_patience} epoch(s). Best val loss: {best_val_loss:.4f}")
        if current_patience >= early_stop_patience:
            logging.info("Early stopping triggered. No improvement observed.")
            break
        
        gc.collect()
    
    final_save_path = os.path.join("models", f"trained_joint_topic_expan_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), final_save_path)
    logging.info(f"Training complete and model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Joint Document-Parent Topic to Subtopic Generation Model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)
