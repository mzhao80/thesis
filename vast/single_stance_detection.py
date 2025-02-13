#!/usr/bin/env python
"""
run.py

This is the main runner script.
It:
  1. Loads VAST train/dev/test data and a separate inference file.
  2. Uses NV-Embed-v2 to separately encode documents and topics.
  3. Trains a classifier on top of these frozen embeddings (using early stopping based on validation loss and reporting F1).
  4. Runs inference on the inference file and saves predictions.
  5. Evaluates the final selected model on the official VAST test set.
  
You can choose which model architecture to use via a command-line argument:
  --model linear       # Uses the simple linear classifier.
  --model attention    # Uses the self-attention classifier.
  --model cross        # Uses the cross-attention classifier.
  --model tga          # Uses the topic-grouped attention classifier.
  
Additionally, this script performs a hyperparameter search over learning rate, batch size, and dropout rate.
It also allows you to choose whether to use the query prefix for NV-Embed by setting the flag --use_prefix.
CUDA is used if available.
"""

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

from dataset import VASTTrainDataset, InferenceDataset, EmbeddingDataset
from transformers import AutoModel
from models import StanceClassifier, AttentionClassifier, CrossAttentionClassifier, TGAClassifier

# -----------------
# A. Configuration
# -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TRAIN_FILE = "zero-shot-stance/data/VAST/vast_train.csv"
DEV_FILE = "zero-shot-stance/data/VAST/vast_dev.csv"
TEST_FILE = "zero-shot-stance/data/VAST/vast_test.csv"  # Official test set for evaluation
INFERENCE_FILE = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv"
OUTPUT_PREDICTIONS = "predictions.csv"
MODEL_PATH = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models"

# Default prefixes
DOC_PREFIX = ("Instruct: Encode the following speech from Congress for stance detection of policy topics.\n"
              "Speech: ")
TOPIC_PREFIX = ("Instruct: Encode the following topic for use in stance detection of Congressional speeches.\n"
                "Topic: ")

DEFAULT_EMBED_DIM = 4096
NUM_EPOCHS = 50
PATIENCE = 10
LABEL_MAP = {0: "con", 1: "pro", 2: "neutral"}

# ---------------------------
# C. Utility to encode documents and topics separately and then concatenate
# ---------------------------
def encode_documents_topics_in_batches(model, documents, topics, batch_size=32,
                                         doc_instruction="", topic_instruction=""):
    all_embeddings = []
    for start_idx in range(0, len(documents), batch_size):
        end_idx = start_idx + batch_size
        batch_docs = documents[start_idx:end_idx]
        batch_topics = topics[start_idx:end_idx]
        doc_embeddings = model.encode(
            batch_docs,
            instruction=doc_instruction,
        )
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        topic_embeddings = model.encode(
            batch_topics,
            instruction=topic_instruction,
        )
        topic_embeddings = F.normalize(topic_embeddings, p=2, dim=1)
        combined_embeddings = torch.cat([doc_embeddings, topic_embeddings], dim=1)
        all_embeddings.append(combined_embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# ---------------------------
# D. Training Loop with Early Stopping and Linear LR Scheduler
# ---------------------------
def train_classifier(classifier, train_loader, dev_loader, learning_rate, print_every=1000, patience=PATIENCE):
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    total_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps)
    classifier.train()
    best_dev_loss = float("inf")
    patience_counter = 0
    best_state = None
    iteration = 0
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        for batch in train_loader:
            embeddings, labels, combined_text = batch
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
            iteration += 1
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
            for embeddings, labels, _ in dev_loader:
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
    final_train_acc = total_correct / total_samples
    print("\nFinal Training Metrics:")
    print(f"Average Loss: {total_loss / total_samples:.4f}")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Best Dev Accuracy: {dev_acc:.4f}, Best Dev Weighted F1: {dev_f1:.4f}")
    return classifier, dev_loss, dev_acc, dev_f1

# ---------------------------
# E. Run Pipeline for a Given Hyperparameter Configuration
# ---------------------------
def run_pipeline(lr_value, model_type, batch_size, dropout_rate, use_prefix):
    print(f"\n=== Running experiment: LR={lr_value}, BS={batch_size}, Dropout={dropout_rate}, Model={model_type}, Use Prefix={use_prefix} ===")
    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()

    train_ds = VASTTrainDataset(csv_file=TRAIN_FILE)
    dev_ds = VASTTrainDataset(csv_file=DEV_FILE)

    train_documents = [item["document"] for item in train_ds]
    train_topics = [item["topic"] for item in train_ds]
    train_labels = [item["label"] for item in train_ds]
    train_seen = [item["seen"] for item in train_ds]

    dev_documents = [item["document"] for item in dev_ds]
    dev_topics = [item["topic"] for item in dev_ds]
    dev_labels = [item["label"] for item in dev_ds]
    dev_seen = [item["seen"] for item in dev_ds]

    # Decide on instruction based on use_prefix flag.
    doc_inst = DOC_PREFIX if use_prefix else ""
    topic_inst = TOPIC_PREFIX if use_prefix else ""

    print("Encoding training data...")
    train_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=train_documents,
        topics=train_topics,
        batch_size=batch_size,
        doc_instruction=doc_inst,
        topic_instruction=topic_inst
    )
    print("Encoding dev data...")
    dev_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=dev_documents,
        topics=dev_topics,
        batch_size=batch_size,
        doc_instruction=doc_inst,
        topic_instruction=topic_inst
    )

    train_dataset = EmbeddingDataset(train_embeddings, tensor(train_labels, dtype=torch.long), train_documents, train_topics, train_seen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = EmbeddingDataset(dev_embeddings, tensor(dev_labels, dtype=torch.long), dev_documents, dev_topics, dev_seen)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    if model_type.lower() == "linear":
        classifier = StanceClassifier(input_dim=2*DEFAULT_EMBED_DIM, num_classes=3).to(DEVICE)
    elif model_type.lower() == "attention":
        classifier = AttentionClassifier(embed_dim=DEFAULT_EMBED_DIM, num_classes=3, num_heads=4, dropout_rate=dropout_rate).to(DEVICE)
    elif model_type.lower() == "cross":
        classifier = CrossAttentionClassifier(embed_dim=DEFAULT_EMBED_DIM, num_classes=3, num_heads=4, dropout_rate=dropout_rate).to(DEVICE)
    elif model_type.lower() == "tga":
        from models import TGAClassifier  # Ensure TGAClassifier is imported.
        classifier = TGAClassifier(embed_dim=DEFAULT_EMBED_DIM, num_classes=3, num_prototypes=50, hidden_dim=512, dropout_rate=dropout_rate).to(DEVICE)
    else:
        raise ValueError("Invalid model type. Choose 'linear', 'attention', 'cross', or 'tga'.")
    
    classifier, dev_loss, dev_acc, dev_f1 = train_classifier(classifier, train_loader, dev_loader, learning_rate=lr_value, print_every=1000, patience=PATIENCE)
    return dev_loss, dev_acc, dev_f1, classifier

# ---------------------------
# F. Main Hyperparameter Search, Inference, and Test Evaluation
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Stance Detection Training and Evaluation")
    parser.add_argument("--model", type=str, default="linear",
                        help="Which model to use: 'linear', 'attention', 'cross', or 'tga'")
    parser.add_argument("--use_prefix", dest="use_prefix", action="store_true",
                        help="Use query prefix in encoding (default)")
    parser.add_argument("--no_prefix", dest="use_prefix", action="store_false",
                        help="Do not use query prefix in encoding")
    parser.set_defaults(use_prefix=True)
    args = parser.parse_args()
    model_type = args.model
    use_prefix = args.use_prefix

    # print flags
    print(f"Using model type: {model_type}")
    print(f"Using query prefix: {use_prefix}")

    candidate_lrs = [1e-3, 1e-4, 1e-5]
    candidate_batch_sizes = [8, 16, 32, 64, 128]
    candidate_dropout_rates = [0, 0.25, 0.5]

    best_config = None
    best_dev_loss = float('inf')
    best_classifier = None

    for lr_value in candidate_lrs:
        for bs in candidate_batch_sizes:
            for dropout in candidate_dropout_rates:
                dev_loss, dev_acc, dev_f1, classifier_candidate = run_pipeline(lr_value, model_type, bs, dropout, use_prefix)
                print(f"Config LR={lr_value}, BS={bs}, Dropout={dropout} -> Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_config = (lr_value, bs, dropout)
                    best_classifier = classifier_candidate

    print(f"\n=== Best configuration for model {model_type}: LR={best_config[0]}, Batch Size={best_config[1]}, Dropout={best_config[2]} with Dev Loss: {best_dev_loss:.4f}, Dev Accuracy: {best_dev_acc:.4f}, Dev F1: {best_dev_f1:.4f} ===")

    # Inference on training_data.csv.
    print("Running inference on training_data.csv...")
    test_ds_inference = InferenceDataset(csv_file=INFERENCE_FILE)
    test_documents = [item["document"] for item in test_ds_inference]
    test_topics = [item["topic"] for item in test_ds_inference]
    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()
    doc_inst = DOC_PREFIX if use_prefix else ""
    topic_inst = TOPIC_PREFIX if use_prefix else ""
    test_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=test_documents,
        topics=test_topics,
        batch_size=best_config[1],
        doc_instruction=doc_inst,
        topic_instruction=topic_inst
    )
    test_embeddings = test_embeddings.to(DEVICE)
    best_classifier.eval()
    all_preds = []
    with torch.no_grad():
        for start_idx in range(0, test_embeddings.size(0), best_config[1]):
            end_idx = start_idx + best_config[1]
            batch_embed = test_embeddings[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
    stance_predictions = [LABEL_MAP[pred] for pred in all_preds]
    df_inference = test_ds_inference.df.copy()
    df_inference["pred"] = stance_predictions
    df_inference.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Predictions written to {OUTPUT_PREDICTIONS}.")

    # Evaluate on the official VAST test set.
    print("Evaluating on the official VAST test set...")
    test_ds_vast = VASTTrainDataset(csv_file=TEST_FILE)
    test_documents_vast = [item["document"] for item in test_ds_vast]
    test_topics_vast = [item["topic"] for item in test_ds_vast]
    test_labels_vast = [item["label"] for item in test_ds_vast]
    test_seen_vast = [item["seen"] for item in test_ds_vast]
    test_embeddings_vast = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=test_documents_vast,
        topics=test_topics_vast,
        batch_size=best_config[1],
        doc_instruction=doc_inst,
        topic_instruction=topic_inst
    )
    test_embeddings_vast = test_embeddings_vast.to(DEVICE)
    best_classifier.eval()
    vast_preds = []
    with torch.no_grad():
        for start_idx in range(0, test_embeddings_vast.size(0), best_config[1]):
            end_idx = start_idx + best_config[1]
            batch_embed = test_embeddings_vast[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            vast_preds.extend(preds.cpu().numpy().tolist())
    vast_preds = [LABEL_MAP[pred] for pred in vast_preds]
    true_labels = [LABEL_MAP[label] for label in test_labels_vast]
    acc = accuracy_score(true_labels, vast_preds)
    f1 = f1_score(true_labels, vast_preds, average="macro")
    print(f"VAST Test Set Accuracy: {acc:.4f}")
    print(f"VAST Test Set Macro F1: {f1:.4f}")

    # Partition the samples based on the 'seen' flag.
    zero_shot_true = [label for label, seen in zip(true_labels, test_seen_vast) if seen == 0]
    zero_shot_pred = [pred for pred, seen in zip(vast_preds, test_seen_vast) if seen == 0]

    few_shot_true = [label for label, seen in zip(true_labels, test_seen_vast) if seen == 1]
    few_shot_pred = [pred for pred, seen in zip(vast_preds, test_seen_vast) if seen == 1]

    # Compute the macro F1 scores for each group.
    f1_zero_shot = f1_score(zero_shot_true, zero_shot_pred, average="macro")
    f1_few_shot = f1_score(few_shot_true, few_shot_pred, average="macro")

    print(f"VAST Test Set Zero-shot Macro F1: {f1_zero_shot:.4f}")
    print(f"VAST Test Set Few-shot Macro F1: {f1_few_shot:.4f}")

if __name__ == "__main__":
    main()
