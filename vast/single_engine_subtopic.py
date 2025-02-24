import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from single_datasets_subtopic import subtopic_data_loader
from he_models import BERTSeqClf
from collections import defaultdict

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def get_prediction_weight(probs, method='entropy'):
    """Calculate prediction weight using either max probability or entropy.
    
    Args:
        probs: Array of probabilities [prob_against, prob_neutral, prob_favor]
        method: Either 'max_prob' or 'entropy'
    
    Returns:
        Weight between 0 and 1, where higher means more confident
    """
    # Extract only against and favor probabilities
    pro_con_probs = np.array([probs[0], probs[1]])  # [against, favor]
    
    # Normalize to sum to 1
    pro_con_probs = pro_con_probs / (pro_con_probs.sum() + 1e-10)
    
    if method == 'max_prob':
        # Simple maximum probability
        return np.max(pro_con_probs)
    else:
        # Entropy-based weight
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(pro_con_probs * np.log(pro_con_probs + 1e-10))
        max_entropy = -np.log(1/2)  # Maximum entropy for 2 classes (uniform distribution)
        # Convert to confidence score (1 = most confident, 0 = least confident)
        return 1 - (entropy / max_entropy)

def calculate_stance_confidence(probs_array, weights):
    """Calculate stance confidence based only on weighted mean probabilities.
    
    Args:
        probs_array: Array of probabilities for each document [n_docs, n_classes]
        weights: Array of weights for each document [n_docs]
        
    Returns:
        Confidence score between 0 and 1
    """
    # Extract and normalize only against and favor probabilities
    pro_con_probs = probs_array[:, [0, 1]]  # Keep only against and favor
    pro_con_probs = pro_con_probs / (pro_con_probs.sum(axis=1, keepdims=True) + 1e-10)
    
    # Calculate weighted mean of pro-con probabilities
    weighted_mean_probs = np.average(pro_con_probs, axis=0, weights=weights)
    
    # Average difference between against and favor probabilities
    confidence = abs(weighted_mean_probs[0] - weighted_mean_probs[1])
    
    return confidence

class SubtopicEngine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using {torch.cuda.device_count()} GPUs!")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Preparing data....')
        self.test_loader = subtopic_data_loader(args.batch_size, model=args.model,
                                              wiki_model=args.wiki_model, n_workers=args.n_workers)
        print('Done\n')

        print('Initializing model....')
        num_labels = 3  # Stance labels: against (0), neutral (1), favor (2)
        self.model = BERTSeqClf(num_labels=num_labels, model=args.model, n_layers_freeze=args.n_layers_freeze,
                               wiki_model=args.wiki_model, n_layers_freeze_wiki=args.n_layers_freeze_wiki)
        self.model = nn.DataParallel(self.model)
        
        # Load the trained model
        print('\nLoading checkpoint....')
        state_dict = torch.load(args.save_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print('Done\n')
        self.model.to(self.device)

    def predict(self):
        self.model.eval()
        
        # Store predictions and probs for each speaker and subtopic
        speaker_subtopic_preds = defaultdict(lambda: defaultdict(list))
        speaker_subtopic_parents = defaultdict(lambda: defaultdict(str))
        speaker_subtopic_probs = defaultdict(lambda: defaultdict(list))
        speaker_subtopic_docs = defaultdict(lambda: defaultdict(list))
        speaker_subtopic_chambers = defaultdict(lambda: defaultdict(str))
        
        print('Running inference...')
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                logits = outputs.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                
                # Store predictions and logits by speaker and subtopic
                speakers = batch['speaker']
                subtopics_2 = batch['subtopics_2']
                for i in range(len(speakers)):
                    speaker = speakers[i]
                    subtopic_2 = subtopics_2[i]
                    if subtopic_2 and subtopic_2 != "Misc.":  # Skip empty subtopics
                        speaker_subtopic_preds[speaker][subtopic_2].append(preds[i])
                        probs = softmax(logits[i])
                        weight = get_prediction_weight(probs, method='entropy')  # Use entropy-based weighting
                        speaker_subtopic_probs[speaker][subtopic_2].append((probs, weight))
                        speaker_subtopic_docs[speaker][subtopic_2].append((batch['document'][i], weight))
                        speaker_subtopic_chambers[speaker][subtopic_2]=batch['chamber'][i]
                        speaker_subtopic_parents[speaker][subtopic_2]=batch['subtopics_1'][i]
        
        # Calculate weighted average probabilities for each speaker-subtopic pair
        results = []
        for speaker in speaker_subtopic_preds:
            for subtopic_2 in speaker_subtopic_preds[speaker]:
                preds = speaker_subtopic_preds[speaker][subtopic_2]
                probs_and_weights = speaker_subtopic_probs[speaker][subtopic_2]
                docs_and_weights = speaker_subtopic_docs[speaker][subtopic_2]
                chamber = speaker_subtopic_chambers[speaker][subtopic_2]
                parent = speaker_subtopic_parents[speaker][subtopic_2]
                
                # Separate probs and weights
                probs = np.array([p[0] for p in probs_and_weights])
                weights = np.array([p[1] for p in probs_and_weights])
                
                # Normalize weights to sum to 1
                weights = weights / (weights.sum() + 1e-10)
                
                # Calculate weighted average
                avg_probs = np.sum(probs * weights[:, np.newaxis], axis=0)
                
                # Calculate stance confidence considering only weighted mean probabilities
                stance_confidence = calculate_stance_confidence(probs, weights)
                
                # Find most important doc (highest weight)
                most_important_idx = np.argmax(weights)
                most_important_doc = docs_and_weights[most_important_idx][0]
                
                results.append({
                    'speaker': speaker,
                    'chamber': chamber,
                    'subtopic_1': parent,
                    'subtopic_2': subtopic_2,
                    'document_count': len(docs_and_weights),
                    'stance': ['against', 'favor', 'neutral'][np.argmax(avg_probs)],
                    'stance_confidence': stance_confidence,
                    'prob_against': avg_probs[0],
                    'prob_favor': avg_probs[1],
                    'prob_neutral': avg_probs[2],
                    'max_doc_weight': weights[most_important_idx],
                    'most_important_doc': most_important_doc
                })
        
        # Create and save results DataFrame
        results_df = pd.DataFrame(results)
        output_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_stance_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f'\nResults saved to: {output_path}')
