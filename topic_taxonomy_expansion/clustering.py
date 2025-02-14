import os
import pandas as pd
import numpy as np
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from hdbscan.validity import validity_index
from openai import OpenAI
import api_keys

def compute_medoid(texts, embeddings):
    """
    Given a list of texts and their embeddings, compute the medoid text.
    The medoid is the text whose embedding is closest to the cluster centroid.
    """
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    similarities = cosine_similarity(embeddings, centroid).flatten()
    medoid_index = np.argmax(similarities)
    return texts[medoid_index]

def generate_cluster_label(subtopics, client, max_attempts=5):
    """
    Uses the OpenAI API (gpt-4o-mini) to generate a succinct label that represents
    the common theme among the given list of subtopics.
    """
    prompt = (
        "Given the following list of subtopics, provide a succinct label that captures "
        "the common theme they represent.\n\n"
        "Subtopics: " + "; ".join(subtopics) + "\n\n"
        "Answer only with a short topic (3-7 words)."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that generates a concise topic label for a cluster of subtopics."
                "Respond only with a short phrase (3-7 words) that represents the common theme."
            )
        },
        {"role": "user", "content": prompt}
    ]
    
    attempt = 0
    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            attempt += 1
    return "Label not generated"


def main():
    # Create directory for visualizations if it doesn't exist.
    if not os.path.exists('cluster_viz'):
        os.makedirs('cluster_viz')

    client = OpenAI(api_key=api_keys.OPENAI_API_KEY)

    # Load the predictions CSV (which contains the original columns).
    df = pd.read_csv('all_data_predictions.csv')

    # Initialize the Sentence Transformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

    source_mapping = {}
    for idx, row in df.iterrows():
        # We convert the index to a string for later concatenation.
        source_mapping[idx] = row['document']

    # List to store cluster taxonomy information.
    taxonomy = []

    # Loop over each parent topic (policy_area)
    for parent_topic in df['policy_area'].unique():
        print(f"\nProcessing parent topic: {parent_topic}")
        
        # Filter by current parent topic.
        sub_df = df[df['policy_area'] == parent_topic]
        
        # Get unique predicted subtopics.
        subtopics = sub_df['pred_subtopic'].unique().tolist()
        print(f"Total subtopics: {len(subtopics)}")
        
        # Skip if too few subtopics.
        if len(subtopics) < 10:
            print("Too few subtopics, skipping...")
            continue

        # Compute original embeddings.
        original_embeddings = model.encode(subtopics, convert_to_numpy=True)
        
        # --- Dimensionality Reduction with UMAP for clustering & DBCV ---
        umap_model = umap.UMAP(n_components=8)
        reduced_embeddings = umap_model.fit_transform(original_embeddings)
        d = reduced_embeddings.shape[1]  # explicitly set d = number of features after reduction
        
        # Cluster using HDBSCAN on the reduced embeddings.
        min_cluster_size = 10
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Print clustering summary.
        unique_labels = set(cluster_labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = list(cluster_labels).count(-1)
        print(f"Clusters (excluding noise): {num_clusters}, Noise points: {noise_count}")
        
        # Compute DBCV score using the reduced embeddings and explicitly set d.
        try:
            dbcv_score = validity_index(
                reduced_embeddings.astype(np.float64),
                cluster_labels,
                metric='euclidean',
                d=d
            )
            print(f"DBCV Score: {dbcv_score:.3f}")
        except Exception as e:
            print("Error computing DBCV score:", e)
        
        # --- Visualization using UMAP to 2 dimensions ---
        # We run a separate UMAP reduction for visualization.
        umap_vis = umap.UMAP(n_components=2)
        vis_embeddings = umap_vis.fit_transform(original_embeddings)
        vis_df = pd.DataFrame({
            "subtopic": subtopics,
            "cluster": cluster_labels,
            "umap_x": vis_embeddings[:, 0],
            "umap_y": vis_embeddings[:, 1]
        })
        vis_df_no_noise = vis_df[vis_df["cluster"] != -1]
        if not vis_df_no_noise.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=vis_df_no_noise, x="umap_x", y="umap_y",
                            hue="cluster", palette="viridis", s=100)
            plt.title(f"UMAP Visualization for {parent_topic}")
            plt.savefig(f"cluster_viz/{parent_topic}_umap.png")
            plt.close()
            print(f"UMAP visualization saved for {parent_topic}")
        
        # --- Organize texts by cluster label (drop noise points) ---
        clusters = {}
        for label, text in zip(cluster_labels, subtopics):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(text)
        
        # For each cluster, compute the medoid
        for label, texts in clusters.items():
            cluster_original_embeddings = model.encode(texts, convert_to_numpy=True)   
            cluster_reduced_embeddings = umap_model.transform(cluster_original_embeddings)
            new_topic_label = compute_medoid(np.array(texts), cluster_original_embeddings)
            new_reduced_topic_label = compute_medoid(np.array(texts), cluster_reduced_embeddings)

            source_indices = []
            for t in texts:
                if t in source_mapping:
                    source_indices.extend(source_mapping[t])
            # Remove duplicates if desired:
            source_indices = list(dict.fromkeys(source_indices))
            
            taxonomy.append({
                'policy_area': parent_topic,
                'cluster_length': len(texts),
                'medoid_text': new_topic_label,
                'reduced_medoid_text': new_reduced_topic_label,
                'gpt_label': generate_cluster_label(texts, client),
                'pred_subtopics': ";".join(texts),
                'source_indices': ";".join(source_indices)
            })

    # Write taxonomy to CSV.
    taxonomy_df = pd.DataFrame(taxonomy)
    taxonomy_df.to_csv('taxonomy_output.csv', index=False)
    print("\nTaxonomy saved to taxonomy_output.csv")

if __name__ == "__main__":
    main()