import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import TopicExpanModel
from dataset import DocTopicPhraseDataset, collate_fn_with_tokenizer
from optimization import AdamW, get_linear_schedule_with_warmup
import dgl
import torch.nn as nn
import gc
from sentence_transformers import SentenceTransformer
import time
import logging
from datetime import datetime
import os

modelPath = "models"
# -----------------------------------------------------------------------------
# Setup logging to both file and terminal with timestamps.
# -----------------------------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#log_file = os.path.join(log_dir, f"train_{timestamp_str}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# -----------------------------------------------------------------------------
# Utility functions for debugging and GPU memory monitoring
# -----------------------------------------------------------------------------
def print_all_gpu_memory(label, device):
    """
    Print allocated and reserved GPU memory for each available GPU.
    
    Args:
      label (str): A label to print with the memory info.
      device (torch.device): The device in use.
    """
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        logging.info(f"[MEMORY] {label}: [GPU {i}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

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
    # Set device to cuda:0 if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # ----------------------------
    # Data Preparation
    # ----------------------------
    # Create the training dataset from CSV.
    dataset = DocTopicPhraseDataset(config["data"]["csv_file"])
    logging.info("dataset loaded")
    # Load the NV-Embed-v2 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    logging.info("tokenizer loaded")
    # Create a collate function that pads sequences appropriately.
    collate_fn = collate_fn_with_tokenizer(tokenizer)
    
    dataloader = DataLoader(dataset,
                            batch_size=config["data"]["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)
    logging.info("dataloader loaded")
    
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
    # Load Pretrained Document Encoder (fixed, not fine-tuned)
    # ----------------------------
    logging.info("loading model")
    # print timestamp
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    policy_encoder = SentenceTransformer(modelPath,
                                         trust_remote_code=True, device=device)
    logging.info("model loaded")
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    policy_encoder.eval()
    for param in policy_encoder.parameters():
        param.requires_grad = False
    # Encode fixed topics to obtain their embeddings.
    fixed_topic_node_feats = policy_encoder.encode(fixed_parent_topics,
                                                   convert_to_tensor=True,
                                                   device=device).float()
    
    print_all_gpu_memory("After fixed topic features", device)
    
    # ----------------------------
    # Build Graphs
    # ----------------------------
    downward_graph, upward_graph, sideward_graph = build_core_graph(fixed_parent_topics, device)
    # print_graph_structure(downward_graph, "Downward Graph")
    # print_graph_structure(upward_graph, "Upward Graph")
    # print_graph_structure(sideward_graph, "Sideward Graph")
    print_all_gpu_memory("After graph construction", device)
    
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
                            unwanted_penalty=config["arch"].get("unwanted_penalty", 1.0))
    
    model.to(device)
    print_all_gpu_memory("After model loading", device)
    
    # ----------------------------
    # Optimizer and Scheduler Setup
    # ----------------------------
    optimizer = AdamW(model.parameters(),
                      lr=config["optimizer"]["lr"],
                      weight_decay=config["optimizer"]["weight_decay"])
    # Calculate total training steps: (number of batches * epochs)
    t_total = (len(dataloader) * config["epochs"])
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    
    # ----------------------------
    # Training Loop with Mixed Precision and Early Stopping (if desired)
    # ----------------------------
    #scaler = torch.amp.GradScaler("cuda")
    model.train()
    batch_loss_accum = 0.0
    batch_count = 0
    # Early stopping parameters here
    for epoch in range(1, config.get("epochs", 5) + 1):
        logging.info(f"\n[INFO] Starting Epoch {epoch}")
        logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        print_all_gpu_memory(f"Epoch {epoch} start", device)
        # Loop over batches.
        for batch_idx, batch in enumerate(dataloader):
            (doc_texts, docs_input_ids, docs_attention_mask, 
             subtopic_input_ids, phrase_input_ids, topic_idxs, policy) = batch
            topic_idxs = topic_idxs.to(device)

            if config.get("use_subtopic", False):
                phrase_input_ids = subtopic_input_ids.to(device)
            else:
                phrase_input_ids = phrase_input_ids.to(device)
             
            optimizer.zero_grad()

            #with torch.amp.autocast("cuda"):
            # Forward pass through the model.
            sim_score, logits = model.forward(doc_texts,
                                                downward_graph,
                                                upward_graph,
                                                sideward_graph,
                                                phrase_input_ids,
                                                topic_idxs)
            # Compute similarity loss (MSE) and generation loss (cross-entropy).
            sim_loss = F.mse_loss(sim_score, torch.ones_like(sim_score))
            if model.phrase_decoder.enable_copy_mechanism:
                gen_loss = F.nll_loss(logits.view(-1, logits.size(-1)),
                                        phrase_input_ids.view(-1),
                                        ignore_index=tokenizer.pad_token_id)
            else:
                gen_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            phrase_input_ids.view(-1),
                                            ignore_index=tokenizer.pad_token_id)
            loss = gen_loss + sim_loss_weight * sim_loss
            
            # # Mixed precision backward pass.
            # scaler.scale(loss).backward()

            # # Gradient clipping for stability.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()

            loss.backward()
            total_grad_norm = compute_total_grad_norm(model.parameters())
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            batch_loss_accum += loss.item()
            batch_count += 1
            
            # Debug output every 200 batches.
            if batch_idx % 200 == 0:
                # --- Additional debug output for gradients ---
                
                logging.info("[DEBUG] Total gradient norm: {:.4f}".format(total_grad_norm))
                # Optionally, print gradient norms per parameter:
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         param_norm = param.grad.data.norm(2)
                #         logging.info(f"[DEBUG] Gradient norm for {name}: {param_norm.item():.4f}")

                logging.info("----- DEBUG SAMPLE -----")
                # log target tokens
                logging.info(f"Target Phrase Tokens: {phrase_input_ids[0]}")
                logging.info(f"Target Phrase: {tokenizer.decode(phrase_input_ids[0], skip_special_tokens=True)}")
                try:
                    parent_index = topic_idxs[0].item()
                    generated_tokens = model.generate_phrase(doc_texts,
                                                             downward_graph,
                                                             upward_graph,
                                                             sideward_graph,
                                                             parent_index)
                    generated_phrase = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    logging.info("Error during generation:", e)
                    generated_phrase = "<error>"
                logging.info(f"Generated Phrase: {generated_phrase}")
                logging.info(f"Loss: {loss.item()}")
                logging.info(f"Sim Loss: {sim_loss.item()}")
                logging.info(f"Gen Loss: {gen_loss.item()}")
                logging.info(f"Generated Tokens: {generated_tokens}")
                logging.info("------------------------")
                print_all_gpu_memory(f"After batch {batch_idx}", device)
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
        avg_loss = batch_loss_accum / batch_count
        logging.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        print_all_gpu_memory(f"After epoch {epoch}", device)
    
    # ----------------------------
    # Save Model (Filtered State Dict without DocumentEncoder)
    # ----------------------------
    # Print keys for debugging.
    logging.info(model.state_dict().keys())
    # Filter out keys starting with "document_encoder" to avoid saving the pretrained encoder.
    filtered_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("document_encoder")}
    torch.save(filtered_state_dict, "trained_topic_expan_model_filtered.pth")
    logging.info("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Topic Expansion Model with learnable fusion, configurable loss weighting, scheduled sampling, mixed precision, and logging")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)
