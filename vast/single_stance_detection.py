#!/usr/bin/env python
"""
single_stance_detection.py

Script that:
1) Loads VAST train/dev data from CSV.
2) Separately encodes the document and topic using nvidia/NV-Embed-v2 to obtain 4096-d embeddings,
   then concatenates them into an 8192-d representation.
3) Trains a simple classifier (a linear layer) on top of these frozen embeddings for stance detection.
4) Evaluates on the dev set.
5) Runs inference on training_data.csv and writes predictions to predictions.csv,
   outputting the document, subtopic, and predicted stance.
   
Debug output is printed periodically (every 1000 iterations) showing a snippet of the document||topic,
the target label, and the predicted label.
CUDA is used if available.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import gc

from dataset import VASTTrainDataset, InferenceDataset

# Import the NV-Embed model (which follows the usage format described in its docs)
from transformers import AutoModel

# -----------------
# A. Configuration
# -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TRAIN_FILE = "zero-shot-stance/data/VAST/vast_train.csv"
DEV_FILE = "zero-shot-stance/data/VAST/vast_dev.csv"
INFERENCE_FILE = "training_data.csv"
OUTPUT_PREDICTIONS = "predictions.csv"
MODEL_PATH = "../models"

# We now use separate instructions for document and topic.
DOC_PREFIX = ("Instruct: Encode the following speech from Congress for stance detection of policy topics.\n"
              "Speech: ")
TOPIC_PREFIX = ("Instruct: Encode the following topic for use in stance detection of Congressional speeches.\n"
                "Topic: ")

# Other configuration (some values will be overridden during hyperparameter search)
EMBED_DIM = 4096  # NV-Embed output dimension per encoding
BATCH_SIZE = 8    # Adjusted batch size (can modify as needed)
NUM_EPOCHS = 200
PATIENCE = 20     # Early stopping patience: stop if dev performance doesn't improve for this many epochs
LABEL_MAP = {0: "con", 1: "pro", 2: "neutral"}

# ---------------------------
# B. Modified Classification Head
# ---------------------------
class StanceClassifier(nn.Module):
    def __init__(self, input_dim=2*EMBED_DIM, num_classes=3):
        super().__init__()
        # The input dimension is doubled (document and topic embeddings concatenated)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        logits = self.linear(x)
        return logits

# ---------------------------
# Custom Dataset to hold embeddings, labels, and the original document/topic (for debug printing)
# ---------------------------
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, documents, topics):
        self.embeddings = embeddings
        self.labels = labels
        self.documents = documents
        self.topics = topics

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        combined_text = self.documents[idx] + " || " + self.topics[idx]
        return self.embeddings[idx], self.labels[idx], combined_text

# ---------------------------
# C. Utility to encode documents and topics separately and then concatenate
# ---------------------------
def encode_documents_topics_in_batches(model, documents, topics, batch_size=32,
                                         doc_instruction="", topic_instruction=""):
    """
    Separately encodes a list of documents and a list of topics using the NV-Embed model.
    Each is normalized and then concatenated to form a final embedding.
    
    Args:
        model: NV-Embed model (from AutoModel.from_pretrained(..., trust_remote_code=True))
        documents (List[str]): List of document strings.
        topics (List[str]): List of topic strings.
        batch_size (int): Batch size.
        doc_instruction (str): Instruction for encoding documents.
        topic_instruction (str): Instruction for encoding topics.
        
    Returns:
        torch.FloatTensor of shape [len(documents), 2 * EMBED_DIM]
    """
    all_embeddings = []
    for start_idx in range(0, len(documents), batch_size):
        end_idx = start_idx + batch_size
        batch_docs = documents[start_idx:end_idx]
        batch_topics = topics[start_idx:end_idx]
        # Encode documents:
        doc_embeddings = model.encode(
            batch_docs,
            instruction=doc_instruction,
        )
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        # Encode topics:
        topic_embeddings = model.encode(
            batch_topics,
            instruction=topic_instruction,
        )
        topic_embeddings = F.normalize(topic_embeddings, p=2, dim=1)
        # Concatenate embeddings along the feature dimension.
        combined_embeddings = torch.cat([doc_embeddings, topic_embeddings], dim=1)
        all_embeddings.append(combined_embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# ---------------------------
# D. Training Loop with Debug Output, Early Stopping, and Linear LR Scheduler
# ---------------------------
def train_classifier(classifier, train_loader, dev_loader, learning_rate, print_every=1000, patience=PATIENCE):
    """
    Train the stance classifier on top of frozen concatenated embeddings.
    Debug output is printed every 'print_every' iterations showing a snippet of the document||topic,
    its target label, and the predicted label.
    
    Early stopping is applied based on dev set accuracy.
    A linear learning rate scheduler is used to decay the learning rate to 0 over training.
    """
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Compute total training steps and create a linear scheduler that decays lr to zero.
    total_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps)
    
    classifier.train()

    best_dev_acc = 0.0
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
            scheduler.step()  # Update learning rate linearly

            total_loss += loss.item() * embeddings.size(0)
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += embeddings.size(0)
            
            iteration += 1
            # Uncomment the following block for per-iteration debug output:
            # if iteration % print_every == 0:
            #     for i in range(len(combined_text)):
            #         debug_text = combined_text[i]
            #         target_label = LABEL_MAP[int(labels[i].cpu().item())]
            #         predicted_label = LABEL_MAP[int(preds[i].cpu().item())]
            #         print(f"Iteration {iteration}: {debug_text[:50]}... | Target: {target_label} | Predicted: {predicted_label}")

        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Evaluate on the dev set after each epoch.
        classifier.eval()
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for embeddings, labels, _ in dev_loader:
                embeddings = embeddings.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = classifier(embeddings)
                _, preds = torch.max(logits, dim=1)
                dev_correct += (preds == labels).sum().item()
                dev_total += embeddings.size(0)
        dev_acc = dev_correct / dev_total
        print(f"Epoch {epoch+1} - Dev Accuracy: {dev_acc:.4f}")

        # Early stopping check.
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = classifier.state_dict()
            patience_counter = 0
            print("Dev accuracy improved; saving best model.")
        else:
            patience_counter += 1
            print(f"No improvement on dev set. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        classifier.train()  # Switch back to training mode for next epoch

    if best_state is not None:
        classifier.load_state_dict(best_state)
    else:
        print("Warning: No improvement on dev set was observed during training.")

    final_train_acc = total_correct / total_samples
    print("\nFinal Training Metrics:")
    print(f"Average Loss: {total_loss / total_samples:.4f}")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Best Dev Accuracy: {best_dev_acc:.4f}")

    return classifier, best_dev_acc

# ---------------------------
# F. Run Pipeline for a Given Learning Rate
# ---------------------------
def run_pipeline(lr_value):
    """
    Runs the complete pipeline (loading data, encoding, training with early stopping) for a given learning rate.
    Returns the dev set accuracy and the trained classifier.
    """
    print(f"\n=== Running experiment for learning rate = {lr_value} ===")
    # 1) Load the NV-Embed model.
    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()

    # 2) Load the train & dev datasets.
    train_ds = VASTTrainDataset(csv_file=TRAIN_FILE)
    dev_ds = VASTTrainDataset(csv_file=DEV_FILE)

    # Retrieve documents, topics, and labels.
    train_documents = [item["document"] for item in train_ds]
    train_topics = [item["topic"] for item in train_ds]
    train_labels = [item["label"] for item in train_ds]

    dev_documents = [item["document"] for item in dev_ds]
    dev_topics = [item["topic"] for item in dev_ds]
    dev_labels = [item["label"] for item in dev_ds]

    # 3) Encode training & dev data with NV-Embed using separate instructions.
    print("Encoding training data...")
    train_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=train_documents,
        topics=train_topics,
        batch_size=BATCH_SIZE,
        doc_instruction=DOC_PREFIX,
        topic_instruction=TOPIC_PREFIX
    )
    print("Encoding dev data...")
    dev_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=dev_documents,
        topics=dev_topics,
        batch_size=BATCH_SIZE,
        doc_instruction=DOC_PREFIX,
        topic_instruction=TOPIC_PREFIX
    )

    # 4) Build datasets for training.
    train_embeddings_tensor = train_embeddings  # already a torch.Tensor
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = EmbeddingDataset(train_embeddings_tensor, train_labels_tensor, train_documents, train_topics)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dev_embeddings_tensor = dev_embeddings
    dev_labels_tensor = torch.tensor(dev_labels, dtype=torch.long)
    dev_dataset = EmbeddingDataset(dev_embeddings_tensor, dev_labels_tensor, dev_documents, dev_topics)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Initialize and train the classifier with early stopping and LR scheduler.
    classifier = StanceClassifier(input_dim=2*EMBED_DIM, num_classes=3).to(DEVICE)
    classifier, dev_acc = train_classifier(classifier, train_loader, dev_loader, learning_rate=lr_value, print_every=1000, patience=PATIENCE)
    
    return dev_acc, classifier

# ---------------------------
# G. Main Hyperparameter Search and Inference
# ---------------------------
def main():
    candidate_lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    best_lr = None
    best_dev_acc = -1
    best_classifier = None

    # Hyperparameter search over candidate learning rates.
    for lr_value in candidate_lrs:
        dev_acc, classifier_candidate = run_pipeline(lr_value)
        print(f"Learning rate {lr_value} achieved Dev Accuracy: {dev_acc:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_lr = lr_value
            best_classifier = classifier_candidate

    print(f"\n=== Best learning rate: {best_lr} with Dev Accuracy: {best_dev_acc:.4f} ===")

    # Inference on training_data.csv using the best classifier.
    print("Running inference on training_data.csv...")
    test_ds = InferenceDataset(csv_file=INFERENCE_FILE)
    test_documents = [item["document"] for item in test_ds]
    test_topics = [item["topic"] for item in test_ds]
    # Use the same NV-Embed model to encode the test data.
    nv_embed_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    nv_embed_model.to(DEVICE)
    nv_embed_model.eval()
    test_embeddings = encode_documents_topics_in_batches(
        model=nv_embed_model,
        documents=test_documents,
        topics=test_topics,
        batch_size=BATCH_SIZE,
        doc_instruction=DOC_PREFIX,
        topic_instruction=TOPIC_PREFIX
    )
    test_embeddings = test_embeddings.to(DEVICE)
    best_classifier.eval()
    all_preds = []
    with torch.no_grad():
        for start_idx in range(0, test_embeddings.size(0), BATCH_SIZE):
            end_idx = start_idx + BATCH_SIZE
            batch_embed = test_embeddings[start_idx:end_idx]
            logits = best_classifier(batch_embed)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
    stance_predictions = [LABEL_MAP[pred] for pred in all_preds]

    # 7) For the final spreadsheet, output the original document, subtopic, and predicted stance.
    df_test = test_ds.df.copy()
    df_test["pred"] = stance_predictions
    df_test.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Predictions written to {OUTPUT_PREDICTIONS}.")

if __name__ == "__main__":
    main()
