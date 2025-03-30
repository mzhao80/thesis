import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.auto import tqdm
import ast

def parse_subjects_or_committees(value, field_type):
    """
    Parse the legislative subjects or committees string into a list of names.
    
    Parameters:
    -----------
    value : str
        The string representation of legislative subjects or committees
    field_type : str
        Either 'legislative_subjects' or 'committees' to determine parsing method
    
    Returns:
    --------
    list: List of subject/committee names
    """
    if pd.isna(value):
        return []
    
    try:
        data = ast.literal_eval(value)
        #data = json.loads(r"{}".format(value.replace("'", '"')))
        return [item['name'] for item in data if 'name' in item]
    except Exception as e:
        print(f"Error parsing {field_type}: {e}")
        print(value)
        return []

def create_gold_standard_mapping(df, field_type='legislative_subjects'):
    """
    Create a mapping of document indices to their gold standard labels.
    Each document can have multiple gold labels based on policy_area and topics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing taxonomy data
    field_type : str
        'legislative_subjects' or 'committees'
    
    Returns:
    --------
    dict: Mapping from document index to set of gold labels
    """
    doc_to_gold_labels = {}
    
    for idx, row in df.iterrows():
        policy_area = row['policy_area']
        
        if pd.isna(policy_area) or not policy_area:
            continue
            
        topics = parse_subjects_or_committees(row[field_type], field_type)
        
        # Create gold labels as (policy_area, topic) pairs
        gold_labels = set()
        for topic in topics:
            if topic:  # Only add if topic is not empty
                gold_labels.add(f"{policy_area}_{topic}")
                
        doc_to_gold_labels[idx] = gold_labels
        
    return doc_to_gold_labels

def create_generated_mapping(df):
    """
    Create a mapping of document indices to their generated taxonomy labels.
    Each document has one generated label based on policy_area and subtopic_1.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing taxonomy data
    
    Returns:
    --------
    dict: Mapping from document index to generated label
    """
    doc_to_gen_label = {}
    
    for idx, row in df.iterrows():
        policy_area = row['policy_area']
        subtopic_1 = row['subtopic_1']
        
        if not policy_area or not subtopic_1:
            continue
            
        # Create generated label as (policy_area, subtopic_1) pair
        gen_label = f"{policy_area}_{subtopic_1}"
        doc_to_gen_label[idx] = gen_label
        
    return doc_to_gen_label

def compute_pairwise_metrics(doc_to_gold_labels, doc_to_gen_label, all_docs=None):
    """
    Compute pairwise precision, recall, and F1 score.
    Two documents are considered correctly clustered if they have the same generated label
    and they share at least one gold label.
    
    Parameters:
    -----------
    doc_to_gold_labels : dict
        Mapping from document index to set of gold standard labels
    doc_to_gen_label : dict
        Mapping from document index to generated label
    all_docs : list or None
        Complete list of document indices to consider for coverage penalty
        
    Returns:
    --------
    tuple: (precision, recall, f1)
    """
    # Get all document indices that have both gold and generated labels
    doc_indices = sorted(set(doc_to_gold_labels.keys()) & set(doc_to_gen_label.keys()))
    
    # Missing documents - documents that have gold labels but no generated labels
    missing_docs = len(set(doc_to_gold_labels.keys()) - set(doc_to_gen_label.keys()))
    
    # Prepare metrics counters
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        # Create mappings for GPU processing
        idx_to_pos = {idx: i for i, idx in enumerate(doc_indices)}
        n = len(doc_indices)
        
        # Create tensors for generated labels
        gen_labels_list = [doc_to_gen_label[idx] for idx in doc_indices]
        gen_label_to_id = {label: i for i, label in enumerate(set(gen_labels_list))}
        gen_ids = torch.tensor([gen_label_to_id[label] for label in gen_labels_list], device='cuda')
        
        # Pre-compute gold label overlaps
        gold_overlap = torch.zeros((n, n), device='cuda', dtype=torch.bool)
        
        for i, idx_i in enumerate(doc_indices):
            gold_i = doc_to_gold_labels[idx_i]
            
            for j, idx_j in enumerate(doc_indices[i+1:], i+1):
                gold_j = doc_to_gold_labels[idx_j]
                
                # Check if they share any gold label
                if gold_i & gold_j:
                    gold_overlap[i, j] = True
                    gold_overlap[j, i] = True
        
        # Compute same generated label
        same_gen = gen_ids.unsqueeze(1) == gen_ids.unsqueeze(0)
        
        # Compute metrics
        upper_tri = torch.triu(torch.ones_like(same_gen, dtype=torch.bool), diagonal=1)
        true_positive = torch.sum((same_gen & gold_overlap & upper_tri)).item()
        false_positive = torch.sum((same_gen & ~gold_overlap & upper_tri)).item()
        false_negative = torch.sum((~same_gen & gold_overlap & upper_tri)).item()
        
    else:
        # CPU implementation
        for i, idx_i in enumerate(doc_indices):
            gen_i = doc_to_gen_label[idx_i]
            gold_i = doc_to_gold_labels[idx_i]
            
            for idx_j in doc_indices[i+1:]:
                gen_j = doc_to_gen_label[idx_j]
                gold_j = doc_to_gold_labels[idx_j]
                
                # Same generated label
                same_gen = gen_i == gen_j
                
                # Share at least one gold label
                share_gold = bool(gold_i & gold_j)
                
                if same_gen and share_gold:
                    true_positive += 1
                elif same_gen and not share_gold:
                    false_positive += 1
                elif not same_gen and share_gold:
                    false_negative += 1
    
    # Add penalty for missing documents
    # Each missing document contributes to false negatives as we failed to label it
    if missing_docs > 0:
        false_negative += missing_docs
    
    # Calculate metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def compute_clustering_metrics(df, field_type='legislative_subjects'):
    """
    Compute clustering evaluation metrics comparing the generated taxonomy against
    the gold standard defined by either legislative subjects or committees.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing taxonomy data
    field_type : str
        'legislative_subjects' or 'committees'
    
    Returns:
    --------
    dict: Evaluation metrics (pairwise precision, recall, F1, AMI)
    """
    print(f"\nEvaluating using {field_type}...")
    
    # Create mappings
    doc_to_gold_labels = create_gold_standard_mapping(df, field_type)
    doc_to_gen_label = create_generated_mapping(df)
    
    # Get documents that have both gold and generated labels
    common_docs = sorted(set(doc_to_gold_labels.keys()) & set(doc_to_gen_label.keys()))
    print(f"Total documents: {len(df)}")
    
    # Get all documents in the gold standard
    all_gold_docs = set(doc_to_gold_labels.keys())
    # Get all documents with generated labels
    all_gen_docs = set(doc_to_gen_label.keys())
    
    # Report coverage
    missing_docs = len(all_gold_docs - all_gen_docs)
    print(f"Generated taxonomy documents: {len(all_gen_docs)} ({100 * len(all_gen_docs) / len(df):.2f}%)")
    print(f"Documents not in our taxonomy: {missing_docs} ({100 * missing_docs / len(df):.2f}%)")
    
    # Compute pairwise metrics with missing document penalty
    pair_precision, pair_recall, pair_f1 = compute_pairwise_metrics(doc_to_gold_labels, doc_to_gen_label)
    print(f"Pairwise Precision: {pair_precision:.4f}")
    print(f"Pairwise Recall: {pair_recall:.4f}")
    print(f"Pairwise F1: {pair_f1:.4f}")
    
    # Prepare for AMI
    # Create numeric cluster assignments
    
    # For gold: assign each document to multiple clusters (one for each gold label)
    gold_label_to_id = {}
    gold_doc_clusters = defaultdict(list)
    
    for idx in common_docs:
        gold_labels = doc_to_gold_labels[idx]
        for label in gold_labels:
            if label not in gold_label_to_id:
                gold_label_to_id[label] = len(gold_label_to_id)
            gold_doc_clusters[idx].append(gold_label_to_id[label])
    
    # For generated: assign each document to one cluster
    gen_label_to_id = {}
    gen_doc_clusters = {}
    
    for idx in common_docs:
        gen_label = doc_to_gen_label[idx]
        if gen_label not in gen_label_to_id:
            gen_label_to_id[gen_label] = len(gen_label_to_id)
        gen_doc_clusters[idx] = gen_label_to_id[gen_label]
    
    # Flatten gold labels for comparison
    # For each document-gold cluster pair, create an entry
    gold_flat = []
    gen_flat = []
    
    for idx in common_docs:
        for gold_id in gold_doc_clusters[idx]:
            gold_flat.append(gold_id)
            gen_flat.append(gen_doc_clusters[idx])
    
    # Compute AMI
    ami = adjusted_mutual_info_score(gold_flat, gen_flat)
    
    print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
    
    # If there are missing documents, calculate coverage-adjusted metrics
    if missing_docs > 0:
        coverage = len(common_docs) / len(all_gold_docs)
        coverage_adjusted_ami = ami * coverage
        
        # Note: Precision is not adjusted by coverage as missing documents don't affect precision
        # Recall and F1 are already affected by missing documents in compute_pairwise_metrics
        # through the addition of missing documents to false negatives
        
        print(f"Coverage: {coverage:.4f}")
        print(f"Coverage-Adjusted AMI: {coverage_adjusted_ami:.4f}")
        
        return {
            'pairwise': {'precision': pair_precision, 'recall': pair_recall, 'f1': pair_f1},
            'ami': ami,
            'coverage': coverage,
            'coverage_adjusted_ami': coverage_adjusted_ami
        }
    
    return {
        'pairwise': {'precision': pair_precision, 'recall': pair_recall, 'f1': pair_f1},
        'ami': ami,
        'coverage': 1.0  # Perfect coverage
    }

def main():
    # Load the taxonomy data
    print("Loading taxonomy data...")
    for file, suffix in zip(['taxonomy_data.csv', 'taxonomy_data_bertopic.csv', 'taxonomy_data_hlda.csv'], ['','_bertopic','_hlda']):
        df = pd.read_csv(file)
        print(len(df['subtopic_1'].unique()))
        print(len(df['subtopic_2'].unique()))
        print(f"Loaded {len(df)} documents ({file})")
        
        # Output device information
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")
        
        # Evaluate using legislative subjects
        metrics_leg = compute_clustering_metrics(df, 'legislative_subjects')
        
        # Evaluate using committees
        metrics_com = compute_clustering_metrics(df, 'committees')
        
        # Export results to CSV
        results_df = pd.DataFrame({
            'Metric': [
                'Pairwise Precision', 'Pairwise Recall', 'Pairwise F1',
                'AMI', 'Coverage', 'Coverage-Adjusted AMI'
            ],
            'Legislative Subjects': [
                metrics_leg['pairwise']['precision'], 
                metrics_leg['pairwise']['recall'], 
                metrics_leg['pairwise']['f1'],
                metrics_leg['ami'], 
                metrics_leg['coverage'],
                metrics_leg.get('coverage_adjusted_ami', 0)
            ],
            'Committees': [
                metrics_com['pairwise']['precision'], 
                metrics_com['pairwise']['recall'], 
                metrics_com['pairwise']['f1'],
                metrics_com['ami'],
                metrics_com['coverage'],
                metrics_com.get('coverage_adjusted_ami', 0)
            ]
        })
        
        results_df.to_csv(f'taxonomy_evaluation_results{suffix}.csv', index=False)
        print(f"\nResults saved to taxonomy_evaluation_results{suffix}.csv")

if __name__ == "__main__":
    main()
