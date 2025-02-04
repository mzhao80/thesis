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
        print(f"[MEMORY] {label}: [GPU {i}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def print_graph_structure(g, name="Graph"):
    """
    Print the structure (in-degrees and out-degrees) for each node in a DGL graph.
    
    Args:
      g (dgl.DGLGraph): The graph to inspect.
      name (str): A name for the graph, printed in the header.
    """
    print(f"\n{name} structure:")
    for node in range(g.num_nodes()):
        in_deg = g.in_degrees(node)
        out_deg = g.out_degrees(node)
        # Convert tensor degrees to Python numbers if necessary.
        if isinstance(in_deg, torch.Tensor): in_deg = in_deg.item()
        if isinstance(out_deg, torch.Tensor): out_deg = out_deg.item()
        print(f"  Node {node}: in-degree = {in_deg}, out-degree = {out_deg}")

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
    print(f"Using device: {device}")
    
    # ----------------------------
    # Data Preparation
    # ----------------------------
    # Create the training dataset from CSV.
    dataset = DocTopicPhraseDataset(config["data"]["csv_file"])
    # Load the NV-Embed-v2 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    # Create a collate function that pads sequences appropriately.
    collate_fn = collate_fn_with_tokenizer(tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=config["data"]["batch_size"],
                            shuffle=True,
                            num_workers=0,
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
    # Load Pretrained Document Encoder (fixed, not fine-tuned)
    # ----------------------------
    from sentence_transformers import SentenceTransformer
    policy_encoder = SentenceTransformer(config["arch"]["doc_encoder_model"],
                                         trust_remote_code=True, device=device)
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
    print_graph_structure(downward_graph, "Downward Graph")
    print_graph_structure(upward_graph, "Upward Graph")
    print_graph_structure(sideward_graph, "Sideward Graph")
    print_all_gpu_memory("After graph construction", device)
    
    # ----------------------------
    # Model Initialization
    # ----------------------------
    sim_loss_weight = config["optimizer"].get("sim_loss_weight", 0.05)
    use_learnable_fusion = config["arch"].get("use_learnable_fusion", True)
    
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
                            use_learnable_fusion=use_learnable_fusion)
    
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
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    batch_loss_accum = 0.0
    batch_count = 0
    # Early stopping parameters here
    for epoch in range(1, config.get("epochs", 5) + 1):
        print(f"\n[INFO] Starting Epoch {epoch}")
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
            try:
                # Forward pass through the model.
                sim_score, logits = model.forward(doc_texts,
                                                  downward_graph,
                                                  upward_graph,
                                                  sideward_graph,
                                                  phrase_input_ids,
                                                  topic_idxs)
            except Exception as e:
                print("Error during forward pass:", e)
                continue
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
            
            # Mixed precision backward pass.
            scaler.scale(loss).backward()
            # Gradient clipping for stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            batch_loss_accum += loss.item()
            batch_count += 1
            
            # Debug output every 500 batches.
            if batch_idx % 500 == 0:
                grad_norm = sum(param.grad.norm().item() for param in model.parameters() if param.grad is not None)
                print("----- DEBUG SAMPLE -----")
                print("Target Phrase:", tokenizer.decode(phrase_input_ids[0], skip_special_tokens=True))
                try:
                    parent_index = topic_idxs[0].item()
                    temperature = config["optimizer"].get("temperature", 1.0)
                    generated_tokens = model.generate_phrase(doc_texts,
                                                             downward_graph,
                                                             upward_graph,
                                                             sideward_graph,
                                                             parent_index,
                                                             temperature=temperature)
                    generated_phrase = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    print("Error during generation:", e)
                    generated_phrase = "<error>"
                print("Generated Phrase:", generated_phrase)
                print("Loss:", loss.item())
                print("Sim Loss:", sim_loss.item())
                print("Gen Loss:", gen_loss.item())
                print("------------------------")
                print_all_gpu_memory(f"After batch {batch_idx}", device)
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
        avg_loss = batch_loss_accum / batch_count
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        print_all_gpu_memory(f"After epoch {epoch}", device)
    
    # ----------------------------
    # Save Model (Filtered State Dict without DocumentEncoder)
    # ----------------------------
    # Print keys for debugging.
    print(model.state_dict().keys())
    # Filter out keys starting with "document_encoder" to avoid saving the pretrained encoder.
    filtered_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("document_encoder")}
    torch.save(filtered_state_dict, "trained_topic_expan_model_filtered.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Topic Expansion Model with learnable fusion, configurable loss weighting, scheduled sampling, mixed precision, and logging")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)
