# engine.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

from model import NVEmbedStanceClassifier
from dataset import StanceDataset

# Import PEFT for LoRA
from peft import get_peft_model, LoraConfig, TaskType

class Engine:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize datasets and data loaders.
        self.train_dataset = StanceDataset(args.train_file, doc_prefix=args.doc_prefix, wiki_prefix=args.wiki_prefix)
        self.dev_dataset = StanceDataset(args.dev_file, doc_prefix=args.doc_prefix, wiki_prefix=args.wiki_prefix)
        self.test_dataset = StanceDataset(args.test_file, doc_prefix=args.doc_prefix, wiki_prefix=args.wiki_prefix)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Initialize our NVEmbed-based classifier.
        self.model = NVEmbedStanceClassifier(
            model_path=args.model_path,
            num_labels=args.num_labels,
            doc_instruction=args.doc_instruction,
            wiki_instruction=args.wiki_instruction,
            dropout_rate=args.dropout_rate
        )
        
        # Apply PEFT with LoRA to adapt the model efficiently.
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["encoder"]  # Apply LoRA to the encoder module.
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            doc_texts = batch["doc_text"]
            wiki_texts = batch["wiki_text"]
            labels = batch["label"].to(self.device)
            
            logits = self.model(doc_texts, wiki_texts)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                doc_texts = batch["doc_text"]
                wiki_texts = batch["wiki_text"]
                labels = batch["label"].to(self.device)
                logits = self.model(doc_texts, wiki_texts)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average='macro')
        return f1

    def train(self):
        best_dev_f1 = 0.0
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            dev_f1 = self.evaluate(self.dev_loader)
            print(f"Epoch {epoch+1}/{self.args.epochs}, Loss: {train_loss:.4f}, Dev Macro F1: {dev_f1:.4f}")
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save(self.model.state_dict(), self.args.save_path)
                print("Best model saved!")
        test_f1 = self.evaluate(self.test_loader)
        print(f"Test Macro F1: {test_f1:.4f}")
