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
from collections import Counter, defaultdict
from tqdm.auto import tqdm

def enforce_majority_cluster(embeddings, cluster_labels, misc_embedding):
    """
    Ensures identical embeddings are assigned to the most frequent cluster they appeared in.
    """
    embedding_dict = defaultdict(lambda: Counter())  # Maps embedding -> cluster frequency counter
    corrected_labels = np.copy(cluster_labels)

    # First pass: Count occurrences of each embedding in different clusters
    for i, emb in enumerate(embeddings):
        emb_tuple = tuple(emb)  # Convert to a hashable type
        embedding_dict[emb_tuple][cluster_labels[i]] += 1  # Count occurrences

    # Second pass: Reassign labels based on majority occurrence
    for i, emb in enumerate(embeddings):
        emb_tuple = tuple(emb)
        if len(embedding_dict[emb_tuple]) > 1:
            # Find the most frequent cluster for this embedding
            most_frequent_cluster = max(embedding_dict[emb_tuple], key=embedding_dict[emb_tuple].get)
            corrected_labels[i] = most_frequent_cluster  # Assign the most common cluster
        # remove misc
        if emb_tuple == misc_embedding:
            corrected_labels[i] = -1

    return corrected_labels

def compute_medoid(texts, embeddings):
    """
    Given a list of texts and their embeddings, compute the medoid text.
    The medoid is the text whose embedding is closest to the cluster centroid.
    """
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    similarities = cosine_similarity(embeddings, centroid).flatten()
    medoid_index = np.argmax(similarities)
    return texts[medoid_index]

def generate_cluster_label(subtopics, parent_topic, client, max_attempts=5):
    """
    Uses the OpenAI API (gpt-4o-mini) to generate a succinct label that represents
    the common theme among the given list of subtopics.
    """
    system_message = (
        "You are a helpful assistant constructing a three-level topic taxonomy. "
        "When given a first-level parent topic and a list of topics of documents clustered under the parent topic, you must generate a concise and representative second-level topic (3-7 words) "
        "that describes the common topic of the document topics clustered under the parent topic.\n"
        "The generated second-level subtopic should be a distinct child topic and more specific than the parent topic, but still be as broad as possible otherwise because we will later generate a third level of the taxonomy. "
        "For example, a good three-level taxonomy would be 1. Defense; 2. Naval Expansion; 3. Asia-Pacific Naval Buildup. "
        "The output must be inherently stanced as something one could take a for or against position on. For example, you should output Budget Cuts instead of Budget Policy and Expanding National Landmarks instead of National Landmarks Controversy."
    )

    prompt = (
        "Given the following parent topic and list of topics from documents clustered under the parent topic, generate a second-level subtopic that is representative of "
        "the common topic the document topics represent. It must be a policy controversy one can take a stance on (for or against). Output the topic with an inherent stance, such as Expanding Refugee Visa Programs instead of Refugee Visa Policy or Refugee Controversy.\n\n"
        "First-Level Parent Topic: " + parent_topic + "\n\n"
        "Clustered Document Topics: " + "; ".join(subtopics) + "\n\n"
        "Answer only with the generated second-level subtopic (2-5 words):\n\n"
    )

    messages = [
        {"role": "system", "content": system_message},
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
    n_components = 5
    n_neighbors = 50
    min_cluster_size = 25
    weighted_score = 0
    doc_count = 0
    # Create directory for visualizations if it doesn't exist.
    if not os.path.exists('cluster_viz'):
        os.makedirs('cluster_viz')

    client = OpenAI(api_key=api_keys.OPENAI_API_KEY)

    # Load the predictions CSV (which contains the original columns).
    df = pd.read_csv('step_1.csv')
    df['idx'] = df['idx'].astype(str)
    original_length = len(df)
    # drop rows with no subtopic predicted
    # df = df.dropna(subset=['pred_subtopic', 'policy_area'])
    # print(f"{original_length - len(df)} rows dropped ({(original_length - len(df)) / original_length * 100:.2f}%) due to no predicted subtopic or policy area.")

    # Initialize the Sentence Transformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

    source_mapping = {}
    for _, row in df.iterrows():
        path = (row['policy_area'], row['pred_subtopic'])
        if path not in source_mapping:
            source_mapping[path] = []
        source_mapping[path].append(row['idx'])
    
    embeddings = {}

    for subtopic in tqdm(df['pred_subtopic'].unique(), desc="Generating embeddings"):
        embeddings[subtopic] = model.encode(subtopic, convert_to_numpy=True)
    embeddings["Misc."] = model.encode("Misc.", convert_to_numpy=True)
    misc_embedding = tuple(embeddings["Misc."])

    # List to store cluster taxonomy information.
    taxonomy = []

    # Loop over each parent topic (policy_area)
    for parent_topic in tqdm(df['policy_area'].unique(), desc="Clustering"):
        curr_doc_count = 0
        print(f"\nProcessing parent topic: {parent_topic}")
        
        # Filter by current parent topic.
        sub_df = df[df['policy_area'] == parent_topic]
        
        # Get unique predicted subtopics.
        subtopics = sub_df['pred_subtopic'].tolist()
        print(f"Total subtopics: {len(subtopics)}")
        
        # Skip if too few subtopics.
        if len(subtopics) < n_components + 2:
            print("Too few subtopics, skipping...")
            source_indices = set()
            for t in subtopics:
                source_indices.update(source_mapping[(parent_topic, t)])
            curr_doc_count += len(source_indices)
            
            taxonomy.append({
                'policy_area': parent_topic,
                'cluster_length': len(subtopics),
                'medoid_text': "Misc.",
                'reduced_medoid_text': "Misc.",
                'gpt_label': "Misc.",
                'pred_subtopics': "Misc.",
                'source_indices': ";".join(source_indices)
            })
            doc_count += curr_doc_count
            print(f"Documents processed: {curr_doc_count}")
            continue

        # Compute original embeddings.
        # original_embeddings = model.encode(subtopics, convert_to_numpy=True)
        # get original_embeddings from embeddings dict instead
        original_embeddings = np.array([embeddings[subtopic] for subtopic in subtopics])
        
        # --- Dimensionality Reduction with UMAP for clustering & DBCV ---
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, metric='cosine')
        reduced_embeddings = umap_model.fit_transform(original_embeddings)
        d = reduced_embeddings.shape[1]  # explicitly set d = number of features after reduction
        
        # Cluster using HDBSCAN on the reduced embeddings.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        cluster_labels = enforce_majority_cluster(original_embeddings, cluster_labels, misc_embedding)
        
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
            weighted_score += dbcv_score * (len(subtopics) - noise_count)
            print(f"Weighted DBCV Score: {weighted_score:.3f}")
        except Exception as e:
            print("Error computing DBCV score:", e)
        
        # --- Visualization using UMAP to 2 dimensions ---
        # We run a separate UMAP reduction for visualization.
        umap_vis = umap.UMAP(n_components=2, metric='cosine')
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
        for label, texts in zip(cluster_labels, subtopics):
            clusters.setdefault(label, []).append(texts)

        # if there is one cluster (one unique value in cluster_labels), set all values of cluster_labels to 1. this will cause them to form one cluster.
        if len(clusters) == 1:
            clusters = {1: subtopics}
        
        # For each cluster, compute the medoid
        for label, texts in clusters.items():
            seen = set()
            newTexts = []
            for text in texts:
                if text not in seen:
                    newTexts.append(text)
                    seen.add(text)
            oldTexts = texts
            texts = newTexts
            
            if label != -1:
                cluster_original_embeddings = np.array([embeddings[t] for t in texts])
                cluster_reduced_embeddings = umap_model.transform(cluster_original_embeddings)
                new_topic_label = compute_medoid(np.array(texts), cluster_original_embeddings)
                new_reduced_topic_label = compute_medoid(np.array(texts), cluster_reduced_embeddings)
                gpt_label = generate_cluster_label(texts, parent_topic, client)
            else:
                new_topic_label = "Misc."
                new_reduced_topic_label = "Misc."
                gpt_label = "Misc."
                
            source_indices = set()
            for t in texts:
                source_indices.update(source_mapping[(parent_topic, t)])
            curr_doc_count += len(source_indices)
            
            taxonomy.append({
                'policy_area': parent_topic,
                'cluster_length': len(oldTexts),
                'medoid_text': new_topic_label,
                'reduced_medoid_text': new_reduced_topic_label,
                'gpt_label': gpt_label,
                'pred_subtopics': ";".join(texts),
                'source_indices': ";".join(source_indices)
            })
        print(f"Documents processed: {curr_doc_count}")
        doc_count += curr_doc_count

    # Write taxonomy to CSV.
    taxonomy_df = pd.DataFrame(taxonomy)
    taxonomy_df.to_csv(f'step_2.csv', index=False)
    
    print(f"\nTaxonomy saved to step_2.csv with weighted DBCV score: {weighted_score:.3f}")
    print(f"Total doc count processed: {doc_count}. Original doc count: {len(df)}")

if __name__ == "__main__":
    main()