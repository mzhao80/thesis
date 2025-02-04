import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm.auto import tqdm
import json
import plotly.graph_objects as go
import re

OUTPUT_DIR = "./topic_taxonomies_outputs"

def preprocess_text(text):
    """Preprocess text by removing speaker references and procedural phrases"""
    if pd.isna(text):
        return ""
        
    # Replace Madam Speaker, Mr. President, Madam President with Mr. Speaker
    text = re.sub(r'Mr\. President', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chair', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Speakerman', 'Mr. Speaker', text)
    text = re.sub(r'Madam President', 'Mr. Speaker', text)
    text = re.sub(r'Madam Speaker', 'Mr. Speaker', text)
    text = re.sub(r'Madam Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chair', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairwoman', 'Mr. Speaker', text)

    # strip out the following phrases from the beginning of each text and leave the remainder:
    # "Mr. Speaker, " 
    text = re.sub(r'^Mr\. Speaker, ', '', text)
    # "Mr. Speaker, I yield myself the balance of my time. "
    text = re.sub(r'^I yield myself the balance of my time\. ', '', text)
    # "I yield myself such time as I may consume. "
    text = re.sub(r'^I yield myself such time as I may consume\. ', '', text)
    
    return text.strip()

class HierarchicalBERTopic:
    def __init__(self, embedder="all-MiniLM-L6-v2", min_topic_size=5):
        self.embedder = SentenceTransformer(embedder, device="cuda")
        self.min_topic_size = min_topic_size
        # Adjust UMAP parameters for small datasets
        self.umap_model = UMAP(
            n_neighbors=3,  # Reduced from 15
            n_components=2,  # Reduced from 5
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
    def fit_transform(self, documents, levels=3):
        """
        Fit hierarchical topic model with specified number of levels
        Returns hierarchical structure of topics
        """
        if len(documents) < 10:
            print(f"Warning: Small dataset ({len(documents)} documents). Adjusting parameters.")
            return self._process_small_dataset(documents)
            
        print(f"Embedding documents...")
        # Preprocess texts
        documents = [preprocess_text(text) for text in documents]
        
        embeddings = self.embedder.encode(
            documents, 
            show_progress_bar=True, 
            batch_size=32
        )
        
        # Initialize storage for hierarchical results
        hierarchy = {
            "levels": [],
            "document_hierarchy": [[] for _ in range(len(documents))]
        }
        
        # Create topic hierarchy
        try:
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP reduction failed: {str(e)}. Falling back to fewer components.")
            # Fallback to simpler dimensionality reduction
            self.umap_model.n_components = min(2, len(documents) - 1)
            self.umap_model.n_neighbors = min(3, len(documents) - 1)
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
        
        # Store topic models for each level
        topic_models = []
        topic_lists = []
        
        for level in range(levels):
            n_clusters = min(
                max(2, int(np.sqrt(len(documents)) / (level + 1))),
                len(documents) - 1
            )
            print(f"\nProcessing level {level + 1} with {n_clusters} clusters...")
            
            # Cluster at current level
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(reduced_embeddings)
            
            # Create BERTopic model for this level
            umap_components = min(2, len(documents) - 1)
            umap_neighbors = min(3, len(documents) - 1)
            
            topic_model = BERTopic(
                embedding_model=self.embedder,
                umap_model=UMAP(
                    n_neighbors=umap_neighbors,
                    n_components=umap_components,
                    min_dist=0.0,
                    metric='cosine'
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=min(self.min_topic_size, max(2, len(documents) // 4)),
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                ),
                min_topic_size=min(self.min_topic_size, max(2, len(documents) // 4)),
                verbose=True
            )
            
            # Fit the model with pre-computed embeddings and clusters
            topics, probs = topic_model.fit_transform(
                documents,
                embeddings=embeddings,
                y=cluster_labels
            )
            
            topic_models.append(topic_model)
            topic_lists.append(topics)
            
            # Get topic info and representative documents
            topic_info = topic_model.get_topic_info()
            topic_docs = topic_model.get_representative_docs()
            
            # Store level information
            level_info = {
                "n_clusters": n_clusters,
                "topics": {},
                "topic_sizes": topic_info['Count'].to_dict(),
                "topic_names": topic_info['Name'].to_dict()
            }
            
            # Store topic words and representative docs
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # Skip outlier topic
                    words = [word for word, _ in topic_model.get_topic(topic_id)][:10]
                    level_info["topics"][str(topic_id)] = {
                        "words": words,
                        "representative_docs": topic_docs.get(topic_id, [])[:3]
                    }
            
            hierarchy["levels"].append(level_info)
            
            # Store document assignments for this level
            for i, topic in enumerate(topics):
                hierarchy["document_hierarchy"][i].append(int(topic))
            
            # Update embeddings for next level using UMAP
            if level < levels - 1:
                try:
                    reduced_embeddings = UMAP(
                        n_neighbors=min(3, len(documents) - 1),
                        n_components=min(2, len(documents) - 1),
                        min_dist=0.0,
                        metric='cosine',
                        random_state=42
                    ).fit_transform(reduced_embeddings)
                except Exception as e:
                    print(f"UMAP reduction failed at level {level}: {str(e)}")
                    break
        
        # Create hierarchical topics dataframe for visualization
        if len(topic_models) > 1:
            hierarchical_topics = []
            for level in range(1, len(topic_lists)):
                parent_topics = topic_lists[level-1]
                child_topics = topic_lists[level]
                
                # Get topic embeddings for distance calculation
                parent_model = topic_models[level-1]
                child_model = topic_models[level]
                
                for i, (parent, child) in enumerate(zip(parent_topics, child_topics)):
                    if parent != -1 and child != -1:
                        # Get topic words
                        parent_words = [word for word, _ in parent_model.get_topic(parent)][:5]
                        child_words = [word for word, _ in child_model.get_topic(child)][:5]
                        
                        # Create topic entries
                        hierarchical_topics.append({
                            'Parent_ID': int(parent),
                            'Child_ID': int(child),
                            'Parent_Name': f"Topic {parent}",
                            'Child_Name': f"Topic {child}",
                            'Topics': parent_words,
                            'Parent_Keywords': " | ".join(parent_words),
                            'Child_Keywords': " | ".join(child_words),
                            'Distance': 1.0  # Use constant distance to avoid negative values
                        })
            
            # Convert to DataFrame and save visualization
            if hierarchical_topics:
                df_hierarchy = pd.DataFrame(hierarchical_topics)
                
                # Save the hierarchy data for debugging
                df_hierarchy.to_csv(os.path.join(OUTPUT_DIR, "hierarchy_debug.csv"), index=False)
                
                try:
                    # Create custom figure for hierarchy visualization
                    fig = go.Figure()
                    
                    # Add edges (connections between parent and child topics)
                    for _, row in df_hierarchy.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['Parent_ID'], row['Child_ID']],
                            y=[0, 1],  # Fixed vertical positions for levels
                            mode='lines',
                            line=dict(color='gray', width=1),
                            showlegend=False
                        ))
                    
                    # Add parent nodes
                    parent_nodes = df_hierarchy[['Parent_ID', 'Parent_Name', 'Parent_Keywords']].drop_duplicates()
                    fig.add_trace(go.Scatter(
                        x=parent_nodes['Parent_ID'],
                        y=[0] * len(parent_nodes),
                        mode='markers+text',
                        marker=dict(size=20, color='lightblue'),
                        text=parent_nodes['Parent_Name'],
                        hovertext=parent_nodes['Parent_Keywords'],
                        showlegend=False
                    ))
                    
                    # Add child nodes
                    child_nodes = df_hierarchy[['Child_ID', 'Child_Name', 'Child_Keywords']].drop_duplicates()
                    fig.add_trace(go.Scatter(
                        x=child_nodes['Child_ID'],
                        y=[1] * len(child_nodes),
                        mode='markers+text',
                        marker=dict(size=20, color='lightgreen'),
                        text=child_nodes['Child_Name'],
                        hovertext=child_nodes['Child_Keywords'],
                        showlegend=False
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Topic Hierarchy",
                        showlegend=False,
                        hovermode='closest',
                        plot_bgcolor='white',
                        width=1200,
                        height=800
                    )
                    
                    # Save visualization
                    fig.write_html(os.path.join(OUTPUT_DIR, f"topic_hierarchy.html"))
                    
                except Exception as e:
                    print(f"Failed to create visualization: {str(e)}")
                    print("Hierarchy data saved to CSV for inspection")
                    df_hierarchy.to_csv(os.path.join(OUTPUT_DIR, "hierarchy_debug.csv"), index=False)
        
        return hierarchy
    
    def _process_small_dataset(self, documents):
        """Handle very small datasets differently"""
        print("Processing small dataset with simplified approach...")
        # Preprocess texts
        documents = [preprocess_text(text) for text in documents]
        
        embeddings = self.embedder.encode(documents, show_progress_bar=True, batch_size=32)
        
        # Use simpler clustering for small datasets
        n_clusters = min(2, len(documents) - 1)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Create simple hierarchy
        hierarchy = {
            "levels": [{
                "n_clusters": n_clusters,
                "topics": {},
                "topic_sizes": {},
                "topic_names": {}
            }],
            "document_hierarchy": [[label] for label in labels]
        }
        
        return hierarchy

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def process_policy_area(policy_area, texts):
    """Process a single policy area with hierarchical BERTopic"""
    print(f"\nProcessing policy area: {policy_area}")
    print(f"Number of documents: {len(texts)}")
    
    if len(texts) < 5:
        return None
        
    model = HierarchicalBERTopic()
    hierarchy = model.fit_transform(texts, levels=3)
    
    # Convert numpy types to Python native types
    hierarchy = convert_to_serializable(hierarchy)
    
    return {
        'num_documents': len(texts),
        'hierarchy': hierarchy
    }

def main():
    # Read the data
    print("Reading data...")
    df = pd.read_csv("df_bills.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each policy area
    print("\nProcessing policy areas...")
    policy_area_results = {}
    
    for policy_area in df['policy_area'].unique():
        if pd.isna(policy_area):
            continue
            
        area_texts = df[df['policy_area'] == policy_area]['speech'].tolist()
        results = process_policy_area(policy_area, area_texts)
        
        if results:
            policy_area_results[policy_area] = results
    
    # Save policy area results
    print("\nSaving policy area results...")
    with open(os.path.join(OUTPUT_DIR, "bertopic_hierarchy.json"), "w") as f:
        json.dump(convert_to_serializable(policy_area_results), f, indent=2)
    
    # Process global hierarchy
    print("\nProcessing global hierarchy...")
    all_texts = df['speech'].tolist()
    model = HierarchicalBERTopic()
    global_hierarchy = model.fit_transform(all_texts, levels=4)
    
    # Save global results
    print("\nSaving global results...")
    with open(os.path.join(OUTPUT_DIR, "bertopic_global_hierarchy.json"), "w") as f:
        json.dump(convert_to_serializable(global_hierarchy), f, indent=2)
    
    print("BERTopic analysis complete!")

if __name__ == "__main__":
    main()
