import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import TopicExpanModel
from dataset import DocTopicPhraseDataset, collate_fn_with_tokenizer
from torch.optim import AdamW
import dgl
import torch.nn as nn
import gc
from sentence_transformers import SentenceTransformer
import time
import logging
from datetime import datetime
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
import sys
import psutil

modelPath = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models"
torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# Setup logging with timestamps.
# -----------------------------------------------------------------------------
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# -----------------------------------------------------------------------------
# Utility functions for debugging and GPU memory monitoring
# -----------------------------------------------------------------------------
def print_all_memory(label):
    """
    Print allocated and reserved memory
    
    Args:
      label (str): A label to print with the memory info.
      device (torch.device): The device in use.
    """
    msgs = []
    mem = psutil.virtual_memory()
    msgs.append(f"Memory Used: {mem.used/1e9:.2f} GB")
    i = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
    msgs.append(f"[GPU {i}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    full_msg = f"[MEMORY] {label}: " + "; ".join(msgs)
    logging.info(full_msg)

def print_graph_structure(g, name="Graph"):
    """
    Print the structure (in-degrees and out-degrees) for each node in a DGL graph.
    
    Args:
      g (dgl.DGLGraph): The graph to inspect.
      name (str): A name for the graph, printed in the header.
    """
    logging.info(f"\n{name} structure:")
    for node in range(g.num_nodes()):
        in_deg = g.in_degrees(node)
        out_deg = g.out_degrees(node)
        # Convert tensor degrees to Python numbers if necessary.
        if isinstance(in_deg, torch.Tensor): in_deg = in_deg.item()
        if isinstance(out_deg, torch.Tensor): out_deg = out_deg.item()
        logging.info(f"  Node {node}: in-degree = {in_deg}, out-degree = {out_deg}")

def compute_total_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# -----------------------------------------------------------------------------
# Build the fixed core graph representing the parent topic and core topics.
# -----------------------------------------------------------------------------
def build_core_graph(topic_list, device):
    """
    Build core graphs for downward, upward, and sideward relations based on a list of topics.
    
    Graph Structure:
      - Node 0 represents "Political Issue" (the parent topic).
      - Nodes 1..N-1 represent the 32 core topics.
    
    Edges:
      - Downward: from node 0 to each core topic.
      - Upward: from each core topic to node 0.
      - Sideward: between every pair of core topics.
    
    Args:
      topic_list (list of str): List of topics (the first should be the parent).
      device (torch.device): The device to move the graph to.
    
    Returns:
      tuple: (downward_graph, upward_graph, sideward_graph)
    """
    num_topics = len(topic_list)
    downward_src = [0] * (num_topics - 1)
    downward_dst = list(range(1, num_topics))
    upward_src = list(range(1, num_topics))
    upward_dst = [0] * (num_topics - 1)
    sideward_src = []
    sideward_dst = []
    # Create sideward edges between every pair of core topics.
    for i in range(1, num_topics):
        for j in range(1, num_topics):
            if i != j:
                sideward_src.append(i)
                sideward_dst.append(j)
    # Construct each graph and move to the specified device.
    downward_graph = dgl.graph((torch.tensor(downward_src, dtype=torch.int64),
                                  torch.tensor(downward_dst, dtype=torch.int64)),
                                  num_nodes=num_topics).to(device)
    upward_graph = dgl.graph((torch.tensor(upward_src, dtype=torch.int64),
                                torch.tensor(upward_dst, dtype=torch.int64)),
                                num_nodes=num_topics).to(device)
    sideward_graph = dgl.graph((torch.tensor(sideward_src, dtype=torch.int64),
                                  torch.tensor(sideward_dst, dtype=torch.int64)),
                                  num_nodes=num_topics).to(device)
    return downward_graph, upward_graph, sideward_graph

# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main(config):
    # Initialize Accelerator (this handles device placement, distributed training, and optional mixed precision)
    kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
    device = accelerator.device  # use accelerator’s device
    logging.info(f"Device: {device}")
    
    # ----------------------------
    # Data Preparation
    # ----------------------------
    # Create the training dataset from CSV.
    full_dataset = DocTopicPhraseDataset(config["data"]["csv_file"])
    # Split dataset: 80% train, 20% validation.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # Load the NV-Embed-v2 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    # Create a collate function that pads sequences appropriately.
    collate_fn = collate_fn_with_tokenizer(tokenizer)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["data"]["batch_size"],
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["data"]["batch_size"],
                                shuffle=False,
                                collate_fn=collate_fn)

    # Define the fixed list of parent and core topics.
    fixed_parent_topics = ["Political Issue"] + [
        "Agriculture and Food",
        "Animals",
        "Armed Forces and National Security",
        "Arts, Culture, Religion",
        "Civil Rights and Liberties, Minority Issues",
        "Commerce",
        "Congress",
        "Crime and Law Enforcement",
        "Economics and Public Finance",
        "Education",
        "Emergency Management",
        "Energy",
        "Environmental Protection",
        "Families",
        "Finance and Financial Sector",
        "Foreign Trade and International Finance",
        "Government Operations and Politics",
        "Health",
        "Housing and Community Development",
        "Immigration",
        "International Affairs",
        "Labor and Employment",
        "Law",
        "Native Americans",
        "Public Lands and Natural Resources",
        "Science, Technology, Communications",
        "Social Sciences and History",
        "Social Welfare",
        "Sports and Recreation",
        "Taxation",
        "Transportation and Public Works",
        "Water Resources Development"
    ]
    # Create a mapping from policy area to index.
    fixed_mapping = {topic: idx for idx, topic in enumerate(fixed_parent_topics)}
    
    # ----------------------------
    # Load Pretrained Document Encoder
    # ----------------------------
    policy_encoder = SentenceTransformer(modelPath, device=device, trust_remote_code=True)
    policy_encoder.eval()
    for param in policy_encoder.parameters():
        param.requires_grad = False
    # Encode fixed topics to obtain their embeddings.
    fixed_topic_node_feats = policy_encoder.encode(fixed_parent_topics,
                                                   convert_to_tensor=True,
                                                   device=device).float()
    # normalize feature embeddings
    fixed_topic_node_feats = F.normalize(fixed_topic_node_feats, p=2, dim=1)
    
    # ----------------------------
    # Build Graphs
    # ----------------------------
    downward_graph, upward_graph, sideward_graph = build_core_graph(fixed_parent_topics, device)
    # print_graph_structure(downward_graph, "Downward Graph")
    # print_graph_structure(upward_graph, "Upward Graph")
    # print_graph_structure(sideward_graph, "Sideward Graph")
    if accelerator.is_main_process:
        print_all_memory("After graph construction")
    
    # ----------------------------
    # Model Initialization
    # ----------------------------
    sim_loss_weight = config["optimizer"].get("sim_loss_weight", 0.05)
    
    # Initialize the Topic Expansion Model.
    model = TopicExpanModel(encoder_model=policy_encoder,
                            vocab_size=tokenizer.vocab_size,
                            hidden_dim=config["arch"]["hidden_dim"],
                            topic_feature_dim=config["arch"]["topic_feature_dim"],
                            num_topic_layers=config["arch"]["num_topic_layers"],
                            num_topic_heads=config["arch"]["num_topic_heads"],
                            num_decoder_layers=config["arch"]["num_decoder_layers"],
                            num_decoder_heads=config["arch"]["num_decoder_heads"],
                            max_length=config["arch"]["max_phrase_length"],
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            doc_encoder_model=config["arch"]["doc_encoder_model"],
                            use_checkpointing=False,
                            enable_copy_mechanism=config["arch"].get("enable_copy_mechanism", False),
                            device=device,
                            fixed_topic_node_feats=fixed_topic_node_feats,
                            use_learnable_fusion=config["arch"].get("use_learnable_fusion", True),
                            bypass_graph=config.get("bypass_graph", False),
                            top_p=config["arch"].get("top_p", 0.9),
                            temperature=config["arch"].get("temperature", 1.0),
                            freq_penalty=config["arch"].get("freq_penalty", 0),
                            pres_penalty=config["arch"].get("pres_penalty", 0),
                            unwanted_penalty=config["arch"].get("unwanted_penalty", 1.0),
                            dropout=config["arch"].get("dropout", 0.1))
    
    model.to(device)
    if accelerator.is_main_process:
        print_all_memory("After model loading")
    
    # ----------------------------
    # Optimizer and Scheduler Setup
    # ----------------------------
    optimizer = AdamW(model.parameters(),
                      lr=config["optimizer"]["lr"],
                      weight_decay=config["optimizer"]["weight_decay"])
    # Total number of training steps (per batch)
    t_total = len(train_dataloader) * config["epochs"]
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # Prepare model, optimizer, and dataloaders for distributed training.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    # ----------------------------
    # Training Loop with Mixed Precision, Early Stopping, and Checkpointing
    # ----------------------------
    model.train()
    best_val_loss = float("inf")
    early_stop_patience = config["early_stopping"]["patience"]   # Number of epochs with no improvement before stopping.
    min_delta = config["early_stopping"]["min_delta"]
    current_patience = 0

    for epoch in range(1, config.get("epochs", 5) + 1):
        logging.info(f"\n[INFO] Starting Epoch {epoch}")
        epoch_loss_accum = 0.0
        epoch_batches = 0
        
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        # ----------------------------
        # Training Phase
        # ----------------------------
        for batch_idx, batch in enumerate(train_dataloader):
            (docs_input_ids, docs_attention_mask, 
             subtopic_input_ids, phrase_input_ids, topic_idxs, policy) = batch
            topic_idxs = topic_idxs.to(device)
            if config.get("use_subtopic", False):
                phrase_input_ids = subtopic_input_ids.to(device)
            else:
                phrase_input_ids = phrase_input_ids.to(device)
             
            optimizer.zero_grad()
            
            # Mixed precision forward pass using accelerator.autocast().
            with accelerator.autocast():
                sim_score, logits = model.forward(docs_input_ids,
                                                  docs_attention_mask,
                                                  downward_graph,
                                                  upward_graph,
                                                  sideward_graph,
                                                  phrase_input_ids,
                                                  topic_idxs)
                sim_loss = F.mse_loss(sim_score, torch.ones_like(sim_score))
                if config["arch"].get("enable_copy_mechanism", False):
                    gen_loss = F.nll_loss(logits.view(-1, logits.size(-1)),
                                          phrase_input_ids.view(-1),
                                          ignore_index=tokenizer.pad_token_id)
                else:
                    gen_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                               phrase_input_ids.view(-1),
                                               ignore_index=tokenizer.pad_token_id)
                loss = gen_loss + sim_loss_weight * sim_loss
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss_accum += loss.item()
            epoch_batches += 1

            if batch_idx % 100 == 0:
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
        
        # Debug output at end of each epoch
        if accelerator.is_main_process:
            try:
                parent_index = topic_idxs[0].item()
                generated_tokens = model.module.generate_phrase(docs_input_ids, 
                                                        docs_attention_mask, 
                                                        downward_graph,
                                                        upward_graph,
                                                        sideward_graph,
                                                        parent_index)
                generated_phrase = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            except Exception as e:
                logging.info("Error during generation: " + str(e))
                generated_phrase = "<error>"

            total_grad_norm = compute_total_grad_norm(model.module.parameters())
            debug_msgs = []
            debug_msgs.append("------------------------")
            debug_msgs.append(f"----- TRAIN EPOCH {epoch} -----")
            debug_msgs.append(f"Train Target Phrase Tokens: {phrase_input_ids[0]}")
            debug_msgs.append(f"Train Target Phrase: {tokenizer.decode(phrase_input_ids[0], skip_special_tokens=True)}")
            debug_msgs.append(f"Generated Phrase: {generated_phrase}")
            debug_msgs.append(f"Train Loss: {loss.item()}")
            debug_msgs.append(f"Train Sim Loss: {sim_loss.item()}")
            debug_msgs.append(f"Train Gen Loss: {gen_loss.item()}")
            debug_msgs.append(f"Generated Tokens: {generated_tokens}")
            debug_msgs.append(f"Total gradient norm: {total_grad_norm:.4f}")
            debug_msgs.append("------------------------")
            full_debug_msg = "\n".join(debug_msgs)
            logging.info(full_debug_msg)

            print_all_memory(f"After epoch {epoch}")

        avg_train_loss = epoch_loss_accum / epoch_batches
        logging.info(f"Epoch {epoch}: Average Training Loss = {avg_train_loss:.4f}")

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        # ----------------------------
        # Validation Phase
        # ----------------------------
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                (docs_input_ids, docs_attention_mask, 
                 subtopic_input_ids, phrase_input_ids, topic_idxs, policy) = batch
                topic_idxs = topic_idxs.to(device)
                if config.get("use_subtopic", False):
                    phrase_input_ids = subtopic_input_ids.to(device)
                else:
                    phrase_input_ids = phrase_input_ids.to(device)
                
                # Forward pass in evaluation mode (no gradient tracking).
                with accelerator.autocast():
                    sim_score, logits = model.forward(docs_input_ids,
                                                      docs_attention_mask,
                                                      downward_graph,
                                                      upward_graph,
                                                      sideward_graph,
                                                      phrase_input_ids,
                                                      topic_idxs)
                    sim_loss = F.mse_loss(sim_score, torch.ones_like(sim_score))
                    if config["arch"].get("enable_copy_mechanism", False):
                        gen_loss = F.nll_loss(logits.view(-1, logits.size(-1)),
                                              phrase_input_ids.view(-1),
                                              ignore_index=tokenizer.pad_token_id)
                    else:
                        gen_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                   phrase_input_ids.view(-1),
                                                   ignore_index=tokenizer.pad_token_id)
                    loss = gen_loss + sim_loss_weight * sim_loss

                val_loss_accum += loss.item()
                val_batches += 1

            # Debug output at end of each epoch
            if accelerator.is_main_process:
                try:
                    parent_index = topic_idxs[0].item()
                    generated_tokens = model.module.generate_phrase(docs_input_ids, 
                                                            docs_attention_mask, 
                                                            downward_graph,
                                                            upward_graph,
                                                            sideward_graph,
                                                            parent_index)
                    generated_phrase = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    logging.info("Error during generation: " + str(e))
                    generated_phrase = "<error>"

                total_grad_norm = compute_total_grad_norm(model.module.parameters())
                debug_msgs = []
                debug_msgs.append("------------------------")
                debug_msgs.append(f"----- VALIDATION EPOCH {epoch} -----")
                debug_msgs.append(f"Validation Target Phrase Tokens: {phrase_input_ids[0]}")
                debug_msgs.append(f"Validation Target Phrase: {tokenizer.decode(phrase_input_ids[0], skip_special_tokens=True)}")
                debug_msgs.append(f"Generated Phrase: {generated_phrase}")
                debug_msgs.append(f"Validation Loss: {loss.item()}")
                debug_msgs.append(f"Validation Sim Loss: {sim_loss.item()}")
                debug_msgs.append(f"Validation Gen Loss: {gen_loss.item()}")
                debug_msgs.append(f"Validation Generated Tokens: {generated_tokens}")
                debug_msgs.append("------------------------")
                full_debug_msg = "\n".join(debug_msgs)
                logging.info(full_debug_msg)

                print_all_memory(f"After epoch {epoch}")

        avg_val_loss = val_loss_accum / val_batches
        logging.info(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss:.4f}")

        model.train()  # Switch back to training mode

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        # ----------------------------
        # Early Stopping Check
        # ----------------------------
        if avg_val_loss - min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            current_patience = 0
        else:
            current_patience += 1
            logging.info(f"No improvement for {current_patience} epoch(s). Best val loss: {best_val_loss:.4f}")
        if current_patience >= early_stop_patience:
            logging.info("Early stopping triggered. No improvement observed.")
            accelerator.set_trigger()

        # Later in the training script when we need to check for the breakpoint
        if accelerator.check_trigger():
            break
        
        # # ----------------------------
        # # Checkpointing Every 10 Epochs
        # # ----------------------------
        # if epoch % 10 == 0 and accelerator.is_main_process:
        #     checkpoint_path = f"trained_topic_expan_model_checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     torch.save(unwrapped_model.state_dict(), checkpoint_path)
        #     logging.info(f"Checkpoint saved: {checkpoint_path}")

    # ----------------------------
    # Save Final Model (Filtered State Dict without DocumentEncoder)
    # ----------------------------
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        filtered_state_dict = {
            k: v for k, v in unwrapped_model.state_dict().items()
            if "document_encoder" not in k
        }
        final_save_path = os.path.join("topic_taxonomy_expansion", "models",
                                    f"trained_topic_expan_model_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        torch.save(filtered_state_dict, final_save_path)
        logging.info(f"Training complete and model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Topic Expansion Model with learnable fusion, configurable loss weighting, scheduled sampling, mixed precision, early stopping, and checkpointing"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)