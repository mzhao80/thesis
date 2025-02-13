
# import argparse
# import json
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.cluster import DBSCAN
# from transformers import AutoTokenizer
# from model import TopicExpanModel
# from dataset import DocTopicPhraseDataset, collate_fn_with_tokenizer
# import dgl

# def build_topic_graphs(topic_hierarchy, num_topics):
#     """
#     Build three graphs:
#       - downward: edges from parent to child
#       - upward: edges from child to parent
#       - sideward: edges among siblings (between every pair of children for a given parent)
#     Self-loops are added to ensure every node has at least one incoming edge.
#     """
#     # Build downward and upward edges.
#     src_down, dst_down = [], []
#     src_up, dst_up = [], []
#     src_side, dst_side = [], []
#     for parent, children in topic_hierarchy.items():
#         for child in children:
#             # Downward: from parent to child.
#             src_down.append(int(parent))
#             dst_down.append(int(child))
#             # Upward: from child to parent.
#             src_up.append(int(child))
#             dst_up.append(int(parent))
#         # Sideward edges: all pairs of distinct children.
#         for i in range(len(children)):
#             for j in range(len(children)):
#                 if i != j:
#                     src_side.append(int(children[i]))
#                     dst_side.append(int(children[j]))
    
#     downward_graph = dgl.graph(
#         (torch.tensor(src_down, dtype=torch.int64) if src_down else torch.tensor([], dtype=torch.int64),
#          torch.tensor(dst_down, dtype=torch.int64) if dst_down else torch.tensor([], dtype=torch.int64)),
#         num_nodes=num_topics)
#     upward_graph = dgl.graph(
#         (torch.tensor(src_up, dtype=torch.int64) if src_up else torch.tensor([], dtype=torch.int64),
#          torch.tensor(dst_up, dtype=torch.int64) if dst_up else torch.tensor([], dtype=torch.int64)),
#         num_nodes=num_topics)
#     sideward_graph = dgl.graph(
#         (torch.tensor(src_side, dtype=torch.int64) if src_side else torch.tensor([], dtype=torch.int64),
#          torch.tensor(dst_side, dtype=torch.int64) if dst_side else torch.tensor([], dtype=torch.int64)),
#         num_nodes=num_topics)
    
#     return downward_graph, upward_graph, sideward_graph


# def compute_parent_topic_features(topic_mapping, model_name="nvidia/NV-Embed-v2"):
#     from transformers import AutoTokenizer
#     from sentence_transformers import SentenceTransformer
#     import numpy as np

#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Computing parent topic features on device: {device}")
#     model_kwargs = {"torch_dtype": torch.bfloat16}
#     nv_model = SentenceTransformer(
#         model_name, 
#         trust_remote_code=True,
#         model_kwargs=model_kwargs
#     ).half().to(device)
    
#     parent_topics = sorted(topic_mapping.items(), key=lambda x: x[1])
#     features = []
#     for topic, idx in parent_topics:
#         topic_str = str(topic)
#         print(f"Encoding topic: {topic_str}")
#         emb = nv_model.encode(topic_str, convert_to_tensor=True)
#         if isinstance(emb, dict) and "sentence_embeddings" in emb:
#             emb = emb["sentence_embeddings"]
#         features.append(emb.cpu().numpy())
#     features = np.stack(features)
#     return torch.tensor(features, dtype=torch.float)

# def embed_phrase(model, tokenizer, phrase, device):
#     enc = tokenizer(phrase, return_tensors="pt", padding=False, truncation=False)
#     enc = {k: v.to(device) for k, v in enc.items()}
#     with torch.no_grad():
#         out = model.document_encoder(enc["input_ids"], enc["attention_mask"])
#     return out.squeeze(0).cpu().numpy()

# def run_expansion(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load dataset.
#     dataset = DocTopicPhraseDataset(config["data"]["csv_file"])
#     dataloader = DataLoader(dataset,
#                             batch_size=config["data"]["batch_size"],
#                             shuffle=False,
#                             collate_fn=collate_fn_with_tokenizer(dataset.tokenizer))
    
#     # Use fixed mapping as in training.
#     policy_areas = [
#         "Agriculture and Food",
#         "Animals",
#         "Armed Forces and National Security",
#         "Arts, Culture, Religion",
#         "Civil Rights and Liberties, Minority Issues",
#         "Commerce",
#         "Congress",
#         "Crime and Law Enforcement",
#         "Economics and Public Finance",
#         "Education",
#         "Emergency Management",
#         "Energy",
#         "Environmental Protection",
#         "Families",
#         "Finance and Financial Sector",
#         "Foreign Trade and International Finance",
#         "Government Operations and Politics",
#         "Health",
#         "Housing and Community Development",
#         "Immigration",
#         "International Affairs",
#         "Labor and Employment",
#         "Law",
#         "Native Americans",
#         "Public Lands and Natural Resources",
#         "Science, Technology, Communications",
#         "Social Sciences and History",
#         "Social Welfare",
#         "Sports and Recreation",
#         "Taxation",
#         "Transportation and Public Works",
#         "Water Resources Development"
#     ]
#     fixed_mapping = {topic: idx for idx, topic in enumerate(policy_areas)}
#     dataset.topic_to_index = fixed_mapping
    
#     # Build hierarchy: root (0) has children 1..n
#     topic_hierarchy = {0: list(range(1, len(policy_areas) + 1))}
#     for i in range(1, len(policy_areas) + 1):
#         topic_hierarchy[i] = []
#     num_topics = len(fixed_mapping)
#     graph_up, graph_side = build_topic_graphs(topic_hierarchy, num_topics)
    
#     # Compute parent topic features.
#     parent_topic_features = compute_parent_topic_features(fixed_mapping,
#                                                           model_name=config["arch"]["doc_encoder_model"])
#     parent_topic_features = parent_topic_features.to(device)
    
#     # Instantiate model.
#     tokenizer = AutoTokenizer.from_pretrained(config["arch"]["doc_encoder_model"], trust_remote_code=True)
#     model = TopicExpanModel(vocab_size=tokenizer.vocab_size,
#                             hidden_dim=config["arch"]["hidden_dim"],
#                             topic_feature_dim=config["arch"]["topic_feature_dim"],
#                             num_topic_layers=config["arch"]["num_topic_layers"],
#                             num_topic_heads=config["arch"]["num_topic_heads"],
#                             num_decoder_layers=config["arch"]["num_decoder_layers"],
#                             num_decoder_heads=config["arch"]["num_decoder_heads"],
#                             max_length=config["arch"]["max_phrase_length"],
#                             pad_token_id=config["arch"]["pad_token_id"],
#                             bos_token_id=config["arch"]["bos_token_id"],
#                             eos_token_id=config["arch"]["eos_token_id"],
#                             doc_encoder_model=config["arch"]["doc_encoder_model"])
#     model.device = device
#     model.to(device)
    
#     # Load trained model checkpoint.
#     checkpoint = torch.load(config["resume"], map_location=device)
#     model.load_state_dict(checkpoint)
#     model.eval()
    
#     generated_phrases = []
#     for batch in dataloader:
#         doc_input_ids, doc_attention_mask, _, topic_idxs = batch
#         doc_input_ids = doc_input_ids.to(device)
#         doc_attention_mask = doc_attention_mask.to(device)
#         topic_idxs = topic_idxs.to(device)
#         with torch.no_grad():
#             gen_tokens = model.generate_phrase(doc_input_ids, doc_attention_mask,
#                                                parent_topic_features, graph_up, graph_side,
#                                                topic_idxs)
#         for seq in gen_tokens:
#             text = tokenizer.decode(seq, skip_special_tokens=True)
#             generated_phrases.append(text)
    
#     print("Generated phrases (sample):")
#     for phrase in generated_phrases[:10]:
#         print(phrase)
    
#     # Embed generated phrases.
#     phrase_embeddings = []
#     for phrase in generated_phrases:
#         emb = embed_phrase(model, tokenizer, phrase, device)
#         phrase_embeddings.append(emb)
#     phrase_embeddings = np.stack(phrase_embeddings)
    
#     # Cluster using DBSCAN.
#     from sklearn.cluster import DBSCAN
#     dbscan = DBSCAN(eps=config["expansion"]["dbscan_eps"],
#                     min_samples=config["expansion"]["min_samples"],
#                     metric="cosine")
#     labels = dbscan.fit_predict(phrase_embeddings)
    
#     print("DBSCAN found the following clusters:")
#     clusters = {}
#     for phrase, label in zip(generated_phrases, labels):
#         if label == -1:
#             continue  # ignore noise
#         clusters.setdefault(label, []).append(phrase)
#     for label, phrases in clusters.items():
#         print(f"Cluster {label}:")
#         for p in phrases:
#             print(f"  {p}")
#     # (You would then integrate these clusters as new subtopics under the corresponding parent nodes.)
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Expand Topic Taxonomy")
#     parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
#     parser.add_argument("--resume", type=str, required=True, help="Path to trained model checkpoint")
#     args = parser.parse_args()
#     with open(args.config, "r") as f:
#         config = json.load(f)
#     config["resume"] = args.resume
#     run_expansion(config)
