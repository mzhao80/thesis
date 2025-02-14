import os
import argparse
import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset_wiki import WikiDataset, EmbeddingDataset
from transformers import AutoModel, AutoTokenizer
from models_wiki import JointMLPClassifier, WikiSingleClassifier, WikiDualClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TRAIN_FILE = "zero-shot-stance/data/VAST/vast_train.csv"
DEV_FILE = "zero-shot-stance/data/VAST/vast_dev.csv"
MODEL_PATH = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models"
TEST_FILE = "zero-shot-stance/data/VAST/vast_test.csv"

# Instruction prefixes and separator strings.
JOINT_PREFIX = "Instruct: Encode the following document and topic jointly for stance detection on the topic.\nDocument and topic: "
JOINT_WIKI_PREFIX = ("Instruct: Encode the following document, topic, and Wikipedia background jointly "
                     "for stance detection from the document on the topic.\nDocument, topic, and wiki: ")
DOC_PREFIX = "Instruct: Encode the following document and topic jointly for stance detection on the topic.\nDocument and topic: "
WIKI_PREFIX = ("Instruct: Encode the following Wikipedia article to provide context for stance detection on this topic "
               "in other documents.\nWikipedia article: ")
SEPARATOR = " || "

DEFAULT_EMBED_DIM = 4096
NUM_EPOCHS = 50
PATIENCE = 10

# ---------------------------
# A. Encoding Functions
# ---------------------------
def encode_single_model(model, dataset, batch_size=32, joint_instruction=""):
    """
    Encode a WikiDataset for WikiSingleClassifier.
    For each item, the document, topic, query, and wiki_text are combined into a single string.
    """
    all_embeddings = []
    all_labels = []
    all_seen = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in tqdm(dataloader, desc="Encoding - Single Model"):
        # Note: when using a dict-based dataset, the DataLoader returns a dict of lists.
        documents = batch["document"]
        topics = batch["topic"]
        queries = batch["query"]
        wiki_texts = batch["wiki_text"]
        labels = batch["label"]
        seen = batch["seen"]
        
        joint_texts = []
        for doc, topic, query, wiki in zip(documents, topics, queries, wiki_texts):
            text = f"Topic: {topic}{SEPARATOR}Document: {doc}{SEPARATOR}There is a Wikipedia article about {query} with the following text: {wiki}"
            joint_texts.append(text)
        
        # Use the provided model to encode the texts.
        joint_embeddings = model.encode(joint_texts, instruction=joint_instruction)
        joint_embeddings = F.normalize(joint_embeddings, p=2, dim=1)
        
        all_embeddings.append(joint_embeddings)
        all_labels.extend(labels)
        all_seen.extend(seen)
    
    return torch.cat(all_embeddings, dim=0), torch.tensor(all_labels), torch.tensor(all_seen)

def encode_dual_model(model, dataset, batch_size=32, doc_instruction="", wiki_instruction=""):
    """
    Encode a WikiDataset for WikiDualClassifier.
    Separately encode:
      1. A combination of the topic and document.
      2. The wiki article (using the query and wiki_text).
    """
    all_doc_embeddings = []
    all_wiki_embeddings = []
    all_labels = []
    all_seen = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in tqdm(dataloader, desc="Encoding - Dual Model"):
        documents = batch["document"]
        topics = batch["topic"]
        queries = batch["query"]
        wiki_texts = batch["wiki_text"]
        labels = batch["label"]
        seen = batch["seen"]
        
        # 1. Encode the document and topic together.
        doc_texts = [f"Topic: {topic}{SEPARATOR}Document: {doc}" for doc, topic in zip(documents, topics)]
        doc_embeddings = model.encode(doc_texts, instruction=doc_instruction)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        
        # 2. Encode the wiki text separately.
        wiki_texts_formatted = [f"{query}: {wiki}" for query, wiki in zip(queries, wiki_texts)]
        wiki_embeddings = model.encode(wiki_texts_formatted, instruction=wiki_instruction)
        wiki_embeddings = F.normalize(wiki_embeddings, p=2, dim=1)
        
        all_doc_embeddings.append(doc_embeddings)
        all_wiki_embeddings.append(wiki_embeddings)
        all_labels.extend(labels)
        all_seen.extend(seen)
    
    return (torch.cat(all_doc_embeddings, dim=0), 
            torch.cat(all_wiki_embeddings, dim=0),
            torch.tensor(all_labels), 
            torch.tensor(all_seen))

# ---------------------------
# B. Training Loop
# ---------------------------
def train_classifier(classifier, train_loader, dev_loader, learning_rate, patience=PATIENCE):
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    total_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps)
    classifier.train()
    best_dev_loss = float("inf")
    patience_counter = 0
    best_state = None
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        for batch in train_loader:
            # Assuming EmbeddingDataset returns (embeddings, labels, extra_info)
            embeddings, labels, _ = batch  
            embeddings = embeddings.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * embeddings.size(0)
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += embeddings.size(0)
        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        classifier.eval()
        dev_loss_total = 0
        dev_correct = 0
        dev_total = 0
        all_dev_preds = []
        all_dev_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                embeddings, labels, _ = batch
                embeddings = embeddings.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = classifier(embeddings)
                loss_dev = criterion(logits, labels)
                dev_loss_total += loss_dev.item() * embeddings.size(0)
                _, preds = torch.max(logits, dim=1)
                dev_correct += (preds == labels).sum().item()
                dev_total += embeddings.size(0)
                all_dev_preds.extend(preds.cpu().numpy().tolist())
                all_dev_labels.extend(labels.cpu().numpy().tolist())
        dev_loss = dev_loss_total / dev_total
        dev_acc = dev_correct / dev_total
        dev_f1 = f1_score(all_dev_labels, all_dev_preds, average="macro")
        print(f"Epoch {epoch+1} - Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = classifier.state_dict()
            patience_counter = 0
            print("Dev loss improved; saving best model.")
            best_dev_acc = dev_acc
            best_dev_f1 = dev_f1
        else:
            patience_counter += 1
            print(f"No improvement on dev loss. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
        classifier.train()
    if best_state is not None:
        classifier.load_state_dict(best_state)
    else:
        print("Warning: No improvement on dev set was observed during training.")
    print(f"\nFinal Training Loss: {total_loss / total_samples:.4f}")
    print(f"Final Training Accuracy: {total_correct / total_samples:.4f}")
    print(f"Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Best Dev Accuracy: {best_dev_acc:.4f}, Best Dev Macro F1: {best_dev_f1:.4f}")
    return classifier, best_dev_loss, best_dev_acc, best_dev_f1

# ---------------------------
# C. Dataset Preparation & Pipeline
# ---------------------------
def prepare_datasets(wiki_mode):
    """
    Load WikiDataset instances for train and dev.
    """
    print("\n=== Loading datasets ===")
    print("Loading datasets...")
    train_ds = WikiDataset(csv_file=TRAIN_FILE, wiki_mode=wiki_mode)
    dev_ds = WikiDataset(csv_file=DEV_FILE, wiki_mode=wiki_mode)
    return {'train': train_ds, 'dev': dev_ds}

def run_pipeline(nv_embed_model, lr_value, model_type, batch_size, datasets, dropout_rate, hidden_dim, num_layers):
    """Run a single experiment with the given hyperparameters."""
    print(f"\nRunning experiment with lr={lr_value}, model={model_type}, batch_size={batch_size}")
    
    with torch.no_grad():
        if model_type == "single":
            train_embeddings, train_labels, train_seen = encode_single_model(
                nv_embed_model,
                datasets['train'],
                batch_size=batch_size,
                joint_instruction=JOINT_WIKI_PREFIX
            )
            dev_embeddings, dev_labels, dev_seen = encode_single_model(
                nv_embed_model,
                datasets['dev'],
                batch_size=batch_size,
                joint_instruction=JOINT_WIKI_PREFIX
            )
        else:  # dual model
            train_doc, train_wiki, train_labels, train_seen = encode_dual_model(
                nv_embed_model,
                datasets['train'],
                batch_size=batch_size,
                doc_instruction=DOC_PREFIX,
                wiki_instruction=WIKI_PREFIX
            )
            dev_doc, dev_wiki, dev_labels, dev_seen = encode_dual_model(
                nv_embed_model,
                datasets['dev'],
                batch_size=batch_size,
                doc_instruction=DOC_PREFIX,
                wiki_instruction=WIKI_PREFIX
            )
            # Concatenate the dual embeddings along the feature dimension.
            train_embeddings = torch.cat([train_doc, train_wiki], dim=1)
            dev_embeddings = torch.cat([dev_doc, dev_wiki], dim=1)

    # Also retrieve the original documents and topics for the EmbeddingDataset.
    train_documents = [item["document"] for item in datasets['train']]
    train_topics = [item["topic"] for item in datasets['train']]
    dev_documents = [item["document"] for item in datasets['dev']]
    dev_topics = [item["topic"] for item in datasets['dev']]

    # Create datasets with the encoded embeddings.
    train_dataset = EmbeddingDataset(train_embeddings, train_labels, train_documents, train_topics, train_seen)
    dev_dataset = EmbeddingDataset(dev_embeddings, dev_labels, dev_documents, dev_topics, dev_seen)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the classifier using the provided hyperparameters.
    if model_type == "single":
        classifier = WikiSingleClassifier(DEFAULT_EMBED_DIM, hidden_dim, 3, dropout_rate, num_layers).to(DEVICE)
    else:
        # For concatenated dual embeddings, the input dimension becomes 2 * DEFAULT_EMBED_DIM.
        classifier = WikiDualClassifier(2 * DEFAULT_EMBED_DIM, hidden_dim, 3, dropout_rate, num_layers).to(DEVICE)
    
    # Train and evaluate.
    classifier, dev_loss, dev_acc, dev_f1 = train_classifier(
        classifier, train_loader, dev_loader,
        learning_rate=lr_value, patience=PATIENCE
    )
    
    return dev_loss, dev_acc, dev_f1, classifier


# ---------------------------
# D. Main Hyperparameter Loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Wiki-Enhanced Stance Detection")
    parser.add_argument("--model", type=str, choices=["single", "dual"], default="single",
                        help="Which classifier to use")
    parser.add_argument("--wiki_mode", type=str, choices=["none", "summary", "content"], default="none",
                        help="Use Wikipedia summary, full content, or none")
    args = parser.parse_args()

    print(f"Using Model: {args.model}, Wiki Mode: {args.wiki_mode}")

    # Candidate hyperparameters.
    candidate_lrs = [1e-4, 1e-5]
    candidate_batch_sizes = [8, 16]
    candidate_dropout_rates = [0.25, 0.5]
    candidate_hidden_dims = [1024, 2048]
    candidate_num_layers = [4,6,8]

    datasets = prepare_datasets(args.wiki_mode)

    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()

    # Loop over hyperparameter configurations.
    for lr in candidate_lrs:
        for bs in candidate_batch_sizes:
            for dropout in candidate_dropout_rates:
                for hidden_dim in candidate_hidden_dims:
                    for num_layers in candidate_num_layers:
                        print(f"\nRunning Config: LR={lr}, BS={bs}, Dropout={dropout}, Hidden Dim={hidden_dim}, Layers={num_layers}, Wiki Mode={args.wiki_mode}")
                        dev_loss, dev_acc, dev_f1, classifier = run_pipeline(
                            nv_embed_model, lr, args.model, bs, datasets,
                            dropout_rate=dropout, hidden_dim=hidden_dim, num_layers=num_layers
                        )
                        print(f"Config Result -> Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}, Dev Macro F1: {dev_f1:.4f}")

    # Test evaluation using the best classifier found.
    print("Evaluating on the official VAST test set...")
    test_ds = WikiDataset(csv_file=TEST_FILE, wiki_mode=args.wiki_mode)
    with torch.no_grad():
        if args.model == "single":
            test_embeddings, test_labels, test_seen = encode_single_model(
                nv_embed_model, test_ds, batch_size=best_batch_size, joint_instruction=JOINT_WIKI_PREFIX
            )
        else:
            test_doc, test_wiki, test_labels, test_seen = encode_dual_model(
                nv_embed_model, test_ds, batch_size=best_batch_size, doc_instruction=DOC_PREFIX, wiki_instruction=WIKI_PREFIX
            )
            test_embeddings = torch.cat([test_doc, test_wiki], dim=1)
    test_embeddings = test_embeddings.to(DEVICE)

    best_classifier.eval()
    vast_preds = []
    with torch.no_grad():
        for start_idx in range(0, test_embeddings.size(0), best_batch_size):
            end_idx = start_idx + best_batch_size
            batch_embed = test_embeddings[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            vast_preds.extend(preds.cpu().numpy().tolist())

    vast_preds = [LABEL_MAP[pred] for pred in vast_preds]
    true_labels = [LABEL_MAP[label] for label in test_labels.numpy().tolist()]
    acc = accuracy_score(true_labels, vast_preds)
    f1 = f1_score(true_labels, vast_preds, average="macro")
    print(f"VAST Test Set Accuracy: {acc:.4f}")
    print(f"VAST Test Set Macro F1: {f1:.4f}")

    # Partition samples based on the 'seen' flag.
    test_seen_list = test_seen.numpy().tolist()
    zero_shot_true = [label for label, seen in zip(true_labels, test_seen_list) if seen == 0]
    zero_shot_pred = [pred for pred, seen in zip(vast_preds, test_seen_list) if seen == 0]
    few_shot_true = [label for label, seen in zip(true_labels, test_seen_list) if seen == 1]
    few_shot_pred = [pred for pred, seen in zip(vast_preds, test_seen_list) if seen == 1]

    f1_zero_shot = f1_score(zero_shot_true, zero_shot_pred, average="macro")
    f1_few_shot = f1_score(few_shot_true, few_shot_pred, average="macro")
    print(f"VAST Test Set Zero-shot Macro F1: {f1_zero_shot:.4f}")
    print(f"VAST Test Set Few-shot Macro F1: {f1_few_shot:.4f}")

if __name__ == "__main__":
    main()
