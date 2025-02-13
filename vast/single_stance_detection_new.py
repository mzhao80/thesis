"""
single_stance_detection_new.py

This is the main runner script for joint document-topic stance detection.
It:
  1. Loads VAST train/dev/test data and an inference file.
  2. Uses NV-Embed-v2 to jointly encode each document and topic pair into a single 4096-dim embedding.
     The document and topic strings are concatenated using "||" with a query prefix.
  3. Trains a classifier on top of these frozen joint embeddings (with early stopping and F1 reporting).
  4. Runs inference on an input file and saves predictions.
  5. Evaluates the final model on the official VAST test set.
  
You can choose which classifier architecture to use via a command-line argument:
  --model linear         # Uses a simple linear classifier.
  --model mlp            # Uses a two-layer MLP classifier.
  
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
from models_new import JointStanceClassifier, JointMLPClassifier

# -----------------
# A. Configuration
# -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TRAIN_FILE = "zero-shot-stance/data/VAST/vast_train.csv"
DEV_FILE = "zero-shot-stance/data/VAST/vast_dev.csv"
TEST_FILE = "zero-shot-stance/data/VAST/vast_test.csv"
INFERENCE_FILE = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv"
OUTPUT_PREDICTIONS = "predictions.csv"
MODEL_PATH = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models"

# Default joint query prefix and separator.
JOINT_PREFIX = ("Instruct: Encode the following document and topic, separated by ||, jointly "
                "for use in stance detection.\nDocument and topic: ")
SEPARATOR = " || "

DEFAULT_EMBED_DIM = 4096
NUM_EPOCHS = 100
PATIENCE = 20
LABEL_MAP = {0: "con", 1: "pro", 2: "neutral"}

# ---------------------------
# C. Joint Encoding Utility Function
# ---------------------------
def encode_joint_documents_topics_in_batches(model, documents, topics, batch_size=32, joint_instruction=""):
    """
    Given lists of documents and topics, create joint strings for each pair by concatenating
    them with a separator and an optional joint_instruction prefix. Then, encode the joint strings using the provided model.
    """
    all_embeddings = []
    for start_idx in range(0, len(documents), batch_size):
        end_idx = start_idx + batch_size
        batch_docs = documents[start_idx:end_idx]
        batch_topics = topics[start_idx:end_idx]
        # Concatenate each document-topic pair.
        joint_texts = [f"{joint_instruction}{doc}{SEPARATOR}{topic}" for doc, topic in zip(batch_docs, batch_topics)]
        # Note: Pass the joint_instruction if your NV-Embed-v2 supports a prompt.
        joint_embeddings = model.encode(joint_texts, instruction=joint_instruction)
        joint_embeddings = F.normalize(joint_embeddings, p=2, dim=1)
        all_embeddings.append(joint_embeddings)
    return torch.cat(all_embeddings, dim=0)

# ---------------------------
# D. Training Loop with Early Stopping and LR Scheduler
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
# E. Prepare Datasets
# ---------------------------
def prepare_datasets():
    """
    Load datasets to be reused across experiments.
    Returns train and dev datasets.
    """
    print("\n=== Loading datasets ===")
    
    # Load datasets
    print("Loading datasets...")
    train_ds = VASTTrainDataset(csv_file=TRAIN_FILE)
    dev_ds = VASTTrainDataset(csv_file=DEV_FILE)
    test_ds = VASTTrainDataset(csv_file=TEST_FILE)
    inference_ds = InferenceDataset(csv_file=INFERENCE_FILE)
    
    return {
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds,
        'inference': inference_ds
    }

# ---------------------------
# F. Run Pipeline for a Hyperparameter Configuration
# ---------------------------
def run_pipeline(nv_embed_model, lr_value, model_type, batch_size, datasets, dropout_rate=None, hidden_dim=None, num_layers=None):
    """Run a single experiment with given hyperparameters."""
    print(f"\nRunning experiment with lr={lr_value}, model={model_type}, batch_size={batch_size}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # Create embeddings with the current batch size
    print("\n=== Encoding datasets with batch_size", batch_size, "===")
    with torch.no_grad():
        # Extract data for train set
        train_documents = [item["document"] for item in datasets['train']]
        train_topics = [item["topic"] for item in datasets['train']]
        train_labels = tensor([item["label"] for item in datasets['train']], dtype=torch.long)
        train_seen = [item["seen"] for item in datasets['train']]
        
        # Encode train set
        train_embeddings = encode_joint_documents_topics_in_batches(
            nv_embed_model,
            train_documents,
            train_topics,
            batch_size=batch_size,
            joint_instruction=JOINT_PREFIX
        )
        
        # Extract data for dev set
        dev_documents = [item["document"] for item in datasets['dev']]
        dev_topics = [item["topic"] for item in datasets['dev']]
        dev_labels = tensor([item["label"] for item in datasets['dev']], dtype=torch.long)
        dev_seen = [item["seen"] for item in datasets['dev']]
        
        # Encode dev set
        dev_embeddings = encode_joint_documents_topics_in_batches(
            nv_embed_model,
            dev_documents,
            dev_topics,
            batch_size=batch_size,
            joint_instruction=JOINT_PREFIX
        )
    
    # Create datasets with the encoded embeddings
    train_dataset = EmbeddingDataset(train_embeddings, train_labels, train_documents, train_topics, train_seen)
    dev_dataset = EmbeddingDataset(dev_embeddings, dev_labels, dev_documents, dev_topics, dev_seen)
    
    # Create data loaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    if model_type == "mlp":
        classifier = JointMLPClassifier(
            input_dim=DEFAULT_EMBED_DIM,
            hidden_dim=hidden_dim,
            num_classes=3,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
    else:
        classifier = JointStanceClassifier(input_dim=DEFAULT_EMBED_DIM, num_classes=3)
    classifier.to(DEVICE)
    
    # Train and evaluate
    classifier, dev_loss, dev_acc, dev_f1 = train_classifier(
        classifier, train_loader, dev_loader,
        learning_rate=lr_value, patience=PATIENCE
    )
    
    return dev_loss, dev_acc, dev_f1, classifier

# ---------------------------
# G. Main Hyperparameter Search, Inference, and Test Evaluation
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Joint Document-Topic Stance Detection Training and Evaluation")
    parser.add_argument("--model", type=str, default="linear",
                        help="Which classifier to use: 'linear' or 'mlp'")
    args = parser.parse_args()
    model_type = args.model

    print(f"Using classifier type: {model_type}")

    # Load datasets once
    datasets = prepare_datasets()

    candidate_lrs = [1e-4]
    candidate_batch_sizes = [8]
    candidate_dropout_rates = [0.1,0.25]
    candidate_hidden_dims = [1024,2048]
    candidate_num_layers = [6,8,10]
    
    best_config = None
    best_dev_loss = float('inf')
    best_classifier = None
    best_dev_acc = 0.0
    best_dev_f1 = 0.0
    best_bs = None
    best_dropout = None
    best_hidden_dim = None
    best_num_layers = None

    # Load NV-Embed model
    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()

    if model_type.lower() == "mlp":
        for lr_value in candidate_lrs:
            for bs in candidate_batch_sizes:
                for dropout in candidate_dropout_rates:
                    for hidden_dim in candidate_hidden_dims:
                        for num_layers in candidate_num_layers:
                            dev_loss, dev_acc, dev_f1, classifier_candidate = run_pipeline(nv_embed_model,
                                lr_value, model_type, bs, datasets,
                                dropout_rate=dropout, hidden_dim=hidden_dim, num_layers=num_layers
                            )
                            print(f"Config LR={lr_value}, BS={bs}, Dropout={dropout}, Hidden Dim={hidden_dim}, Num Layers={num_layers} -> Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")
                            if dev_loss < best_dev_loss:
                                best_dev_loss = dev_loss
                                best_config = (lr_value, bs, dropout, hidden_dim, num_layers)
                                best_classifier = classifier_candidate
                                best_dev_acc = dev_acc
                                best_dev_f1 = dev_f1
                                best_bs = bs
                                best_dropout = dropout
                                best_hidden_dim = hidden_dim
                                best_num_layers = num_layers
    # else:  # Linear classifier
    #     for lr_value in candidate_lrs:
    #         for bs in candidate_batch_sizes:
    #             dev_loss, dev_acc, dev_f1, classifier_candidate = run_pipeline(
    #                 lr_value, model_type, bs, datasets
    #             )
    #             print(f"Config LR={lr_value}, BS={bs} -> Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")
    #             if dev_loss < best_dev_loss:
    #                 best_dev_loss = dev_loss
    #                 best_config = (lr_value, bs)
    #                 best_classifier = classifier_candidate
    #                 best_dev_acc = dev_acc
    #                 best_dev_f1 = dev_f1
    #                 best_bs = bs

    print(f"\n=== Best configuration for model {model_type}: {best_config} with Dev Loss: {best_dev_loss:.4f}, Dev Accuracy: {best_dev_acc:.4f}, Dev F1: {best_dev_f1:.4f} ===")

    # Run inference using best model
    print("Running inference on training_data.csv...")
    inference_embeddings = encode_joint_documents_topics_in_batches(
        AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True),
        [item["document"] for item in datasets['inference']],
        [item["topic"] for item in datasets['inference']],
        batch_size=best_bs,
        joint_instruction=JOINT_PREFIX
    )
    print("Embeddings created.")
    inference_embeddings = inference_embeddings.to(DEVICE)
    best_classifier.eval()
    all_preds = []
    with torch.no_grad():
        for start_idx in range(0, inference_embeddings.size(0), best_bs):
            end_idx = start_idx + best_bs
            batch_embed = inference_embeddings[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
    stance_predictions = [LABEL_MAP[pred] for pred in all_preds]
    df_inference = datasets['inference'].df.copy()
    df_inference["pred"] = stance_predictions
    df_inference.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Predictions written to {OUTPUT_PREDICTIONS}.")

    # Evaluate on test set
    print("Evaluating on the official VAST test set...")
    test_embeddings = encode_joint_documents_topics_in_batches(
        AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True),
        [item["document"] for item in datasets['test']],
        [item["topic"] for item in datasets['test']],
        batch_size=best_bs,
        joint_instruction=JOINT_PREFIX
    )
    test_embeddings = test_embeddings.to(DEVICE)
    best_classifier.eval()
    test_preds = []
    with torch.no_grad():
        for start_idx in range(0, test_embeddings.size(0), best_bs):
            end_idx = start_idx + best_bs
            batch_embed = test_embeddings[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy().tolist())
    test_preds = [LABEL_MAP[pred] for pred in test_preds]
    true_labels = [LABEL_MAP[label] for label in [item["label"] for item in datasets['test']]]
    acc = accuracy_score(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds, average="macro")
    print(f"VAST Test Set Accuracy: {acc:.4f}")
    print(f"VAST Test Set Macro F1: {f1:.4f}")

    # Partition predictions based on the 'seen' flag
    zero_shot_true = [label for label, seen in zip(true_labels, [item["seen"] for item in datasets['test']]) if seen == 0]
    zero_shot_pred = [pred for pred, seen in zip(test_preds, [item["seen"] for item in datasets['test']]) if seen == 0]
    few_shot_true = [label for label, seen in zip(true_labels, [item["seen"] for item in datasets['test']]) if seen == 1]
    few_shot_pred = [pred for pred, seen in zip(test_preds, [item["seen"] for item in datasets['test']]) if seen == 1]

    f1_zero_shot = f1_score(zero_shot_true, zero_shot_pred, average="macro")
    f1_few_shot = f1_score(few_shot_true, few_shot_pred, average="macro")

    print(f"VAST Test Set Zero-shot Macro F1: {f1_zero_shot:.4f}")
    print(f"VAST Test Set Few-shot Macro F1: {f1_few_shot:.4f}")

if __name__ == "__main__":
    main()
