import torch
import torch.nn as nn
import os
import numpy as np
import wandb
from he_datasets import data_loader
from he_models import BERTSeqClf
import gc

class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Preparing data....')
        if args.inference == 0:
            print('Training data....')
            train_loader = data_loader(args.data, 'train', args.topic, args.batch_size, 
                                       model=args.model, wiki_model=args.wiki_model, n_workers=args.n_workers)
            print('Val data....')
            val_loader = data_loader(args.data, 'val', args.topic, 2 * args.batch_size, 
                                     model=args.model, wiki_model=args.wiki_model, n_workers=args.n_workers)
        else:
            train_loader = None
            val_loader = None

        print('Test data....')
        test_loader = data_loader(args.data, 'test', args.topic, 2 * args.batch_size, 
                                  model=args.model, wiki_model=args.wiki_model, n_workers=args.n_workers)
        print('Done\n')

        print('Initializing model....')
        num_labels = 3
        model = BERTSeqClf(num_labels=num_labels, model=args.model, n_layers_freeze=args.n_layers_freeze,
                           wiki_model=args.wiki_model, n_layers_freeze_wiki=args.n_layers_freeze_wiki)
        model = nn.DataParallel(model)
        if args.inference == 1:
            model_name = f"ckp/model_{args.data}_{args.topic}.pt"
            print('\nLoading checkpoint....')
            state_dict = torch.load(model_name, map_location='cpu')
            model.load_state_dict(state_dict)
            print('Done\n')
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = nn.CrossEntropyLoss(ignore_index=3)

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args

    def train(self):
        gc.collect()
        torch.cuda.empty_cache()
        if self.args.inference == 0:
            import copy
            best_epoch = 0
            best_epoch_f1 = 0
            best_state_dict = copy.deepcopy(self.model.state_dict())
            for epoch in range(self.args.epochs):
                print(f"{'*' * 30} Epoch: {epoch + 1} {'*' * 30}")
                train_loss = self.train_epoch()
                val_f1 = self.eval('val')
                if val_f1 > best_epoch_f1:
                    best_epoch = epoch
                    best_epoch_f1 = val_f1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"Epoch: {epoch + 1}\tTrain Loss: {train_loss:.3f}\tVal F1: {val_f1:.3f}")
                print(f"Best Epoch: {best_epoch + 1}\tBest Val F1: {best_epoch_f1:.3f}\n")
                
                # Log epoch-level metrics to wandb
                # wandb.log({
                #     "epoch": epoch + 1,
                #     "train_loss": train_loss,
                #     "val_f1": val_f1,
                #     "best_epoch": best_epoch + 1,
                #     "best_val_f1": best_epoch_f1
                # })

                if epoch - best_epoch >= self.args.patience:
                    break

            print('Saving the best checkpoint....')
            self.model.load_state_dict(best_state_dict)
            model_name = f"ckp/model_{self.args.data}_{self.args.topic}.pt"
            torch.save(best_state_dict, model_name)

        print('Inference...')
        test_f1, test_f1_few, test_f1_zero = self.eval('test')
        print(f"Test F1: {test_f1:.3f}\tTest F1_Few: {test_f1_few:.3f}\tTest F1_Zero: {test_f1_zero:.3f}")
        
        # Log test metrics to wandb
        # wandb.log({
        #     "test_f1": test_f1,
        #     "test_f1_few": test_f1_few,
        #     "test_f1_zero": test_f1_zero
        # })
        gc.collect()
        torch.cuda.empty_cache()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            stances = batch['stances'].to(self.device)
            if self.args.wiki_model and self.args.wiki_model != self.args.model:
                input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
            else:
                input_ids_wiki = None
                attention_mask_wiki = None

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
            loss = self.criterion(logits, stances)
            loss.backward()
            self.optimizer.step()

            # Log batch loss occasionally
            interval = max(len(self.train_loader) // 10, 1)
            if i % interval == 0 or i == len(self.train_loader) - 1:
                batch_loss = loss.item()
                print(f"Batch: {i + 1}/{len(self.train_loader)}\tLoss: {batch_loss:.3f}")
                # wandb.log({"batch_loss": batch_loss})

            epoch_loss += loss.item()
        
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_loss / len(self.train_loader)

    def eval(self, phase='val'):
        gc.collect()
        torch.cuda.empty_cache()
        self.model.eval()
        all_preds = []
        all_labels = []
        all_valid = []
        all_few_shot = []   # Accumulate few_shot values for each batch
        val_loader = self.val_loader if phase == 'val' else self.test_loader
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['stances']  # shape: (batch, 3)
                valid_mask = batch['valid_mask']  # shape: (batch, 3)
                few_shot = batch['few_shot']      # shape: (batch, 3)

                if self.args.wiki_model and self.args.wiki_model != self.args.model:
                    input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                    attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
                else:
                    input_ids_wiki = None
                    attention_mask_wiki = None

                logits = self.model(input_ids, attention_mask, token_type_ids,
                                    input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
                # logits shape: (batch, 3, num_labels)
                preds = logits.argmax(dim=2)  # shape: (batch, 3)
                all_preds.append(preds.detach().to('cpu').numpy())
                all_labels.append(labels.detach().to('cpu').numpy())
                all_valid.append(valid_mask.detach().to('cpu').numpy())
                all_few_shot.append(few_shot.detach().to('cpu').numpy())

        y_pred = np.concatenate(all_preds, axis=0)   # (N, 3)
        y_true = np.concatenate(all_labels, axis=0)    # (N, 3)
        valid_mask = np.concatenate(all_valid, axis=0) # (N, 3)
        few_shot_all = np.concatenate(all_few_shot, axis=0)  # (N, 3)

        # Compute F1 per target ignoring padded duplicates
        f1_scores = []
        for i in range(3):
            mask = valid_mask[:, i].astype(bool)
            if mask.sum() > 0:
                f1 = f1_score(y_true[:, i][mask], y_pred[:, i][mask], average="macro")
                f1_scores.append(f1)
        f1_avg = np.mean(f1_scores) if f1_scores else 0.0

        if phase == 'test':
            few_scores = []
            zero_scores = []
            for i in range(3):
                mask_valid = valid_mask[:, i].astype(bool)
                mask_few = (few_shot_all[:, i].astype(bool)) & mask_valid
                if mask_few.sum() > 0:
                    f1_few = f1_score(y_true[:, i][mask_few], y_pred[:, i][mask_few], average="macro")
                else:
                    f1_few = 0.0
                few_scores.append(f1_few)
                mask_zero = (~few_shot_all[:, i].astype(bool)) & mask_valid
                if mask_zero.sum() > 0:
                    f1_zero = f1_score(y_true[:, i][mask_zero], y_pred[:, i][mask_zero], average="macro")
                else:
                    f1_zero = 0.0
                zero_scores.append(f1_zero)
            f1_avg_few = np.mean(few_scores) if few_scores else 0.0
            f1_avg_zero = np.mean(zero_scores) if zero_scores else 0.0
            gc.collect()
            torch.cuda.empty_cache()
            return f1_avg, f1_avg_few, f1_avg_zero

        gc.collect()
        torch.cuda.empty_cache()
        return f1_avg
