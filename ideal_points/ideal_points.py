"""
StrideStance Ideal Points: PCA-Based Legislator Scaling

This script implements PCA-based ideal point estimation for legislators based on their stance 
scores across various political topics. The approach treats each topic as a feature and each 
legislator as an observation, seeking to find orthogonal axes that capture maximum variance 
in the stance data.

Key features:
1. Preprocessing stance detection outputs into a consistent format
2. Aggregating multiple speeches into single stance scores per speaker-topic pair
3. Creating speaker-topic matrices at different granularity levels (topic, subtopic1, subtopic2)
4. Applying hierarchical propagation to handle missing values
5. Reducing dimensionality through PCA or alternative methods
6. Visualizing results with party-based coloring
7. Saving processed data for downstream analysis

The script addresses data sparsity (legislators not speaking on all topics) through a
hierarchical propagation approach that leverages the topic taxonomy structure to fill in
missing values from parent topics when appropriate.

Methodology:
- Speaker-Topic Matrix Creation: Aggregates stance scores across speeches using weighted
  averaging, max pooling, or simple averaging.
- Hierarchical Propagation: Propagates stance scores from subtopic1 to subtopic2 when needed.
- Dimensionality Reduction: Applies PCA (default), UMAP, or TruncatedSVD.
- Visualization: Creates scatter plots of legislators in the reduced dimensional space.

Usage:
  python ideal_points.py --data_path speaker_stance_results.csv
                        --subtopic_level subtopic1
                        --method pca
                        --n_components 2
                        --speaker_percentile 0.75
                        --topic_percentile 0.75
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import umap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_and_preprocess_data(data_path):
    """Load and preprocess the stance data."""
    results_df = pd.read_csv(data_path)
    
    # Compute stance score
    results_df['stance_score'] = results_df['prob_favor'] - results_df['prob_against']
    
    # Use document_count as combined weight if word_count isn't available
    if 'document_word_count' in results_df.columns:
        print("Using document word count for weighting")
        results_df['combined_weight'] = results_df['document_word_count']
    else:
        print("Using document count for weighting (word count not available)")
        results_df['combined_weight'] = results_df['document_count']
    
    # Create mapping of subtopic_2 to their parent subtopic_1
    subtopic_mapping = results_df[['subtopic_1', 'subtopic_2']].drop_duplicates()
    subtopic_mapping = subtopic_mapping.set_index('subtopic_2')['subtopic_1'].to_dict()
    
    return results_df, subtopic_mapping

def create_topic_matrix(results_df, subtopic_mapping, aggregation_method='weighted_averaging'):
    """
    Create the topic matrices with aggregated scores.
    
    Args:
        results_df: DataFrame with stance prediction results
        subtopic_mapping: Dictionary mapping subtopic_2 to parent subtopic_1
        aggregation_method: Method to aggregate scores ('simple_averaging', 'weighted_averaging', or 'max_pooling')
        
    Returns:
        speaker_subtopic2_filled: DataFrame with filled scores for subtopic_2
        speaker_subtopic1_filled: DataFrame with scores for subtopic_1
        speaker_subtopic2_raw: Raw DataFrame with unfilled scores for subtopic_2
    """
    print(f"Using '{aggregation_method}' method for subtopic aggregation")
    
    # Get direct stance scores for subtopic_2
    speaker_subtopic2 = results_df.pivot_table(
        index='speaker',
        columns='subtopic_2',
        values='stance_score',
        aggfunc='first'
    )
    
    # Calculate scores for each speaker-subtopic_1 combination using the specified method
    speaker_subtopic1 = pd.DataFrame(index=speaker_subtopic2.index)
    for subtopic1 in results_df['subtopic_1'].unique():
        subtopic1_data = results_df[results_df['subtopic_1'] == subtopic1]
        
        if aggregation_method == 'weighted_averaging':
            # Weighted average by document word count or document count
            weighted_avg = subtopic1_data.groupby('speaker').apply(
                lambda x: np.average(x['stance_score'], weights=x['combined_weight'])
                if len(x) > 0 and sum(x['combined_weight']) > 0 else np.nan
            )
            speaker_subtopic1[subtopic1] = weighted_avg
            
        elif aggregation_method == 'max_pooling':
            # Use the stance score with the highest absolute value
            max_score = subtopic1_data.groupby('speaker').apply(
                lambda x: x.loc[x['stance_score'].abs().idxmax(), 'stance_score'] 
                if len(x) > 0 else np.nan
            )
            speaker_subtopic1[subtopic1] = max_score
            
        else:  # simple_averaging or fallback
            # Simple average of stance scores
            simple_avg = subtopic1_data.groupby('speaker')['stance_score'].mean()
            speaker_subtopic1[subtopic1] = simple_avg
    
    # Fill missing values with parent topic scores
    speaker_subtopic2_filled = speaker_subtopic2.copy()
    for subtopic2 in speaker_subtopic2.columns:
        parent_subtopic1 = subtopic_mapping[subtopic2]
        mask = speaker_subtopic2[subtopic2].isna()
        speaker_subtopic2_filled.loc[mask, subtopic2] = speaker_subtopic1.loc[mask, parent_subtopic1]
    
    # Fill remaining missing values with 0
    speaker_subtopic2_filled = speaker_subtopic2_filled.fillna(0)
    speaker_subtopic1_filled = speaker_subtopic1.fillna(0)
    
    return speaker_subtopic2_filled, speaker_subtopic1_filled, speaker_subtopic2

def reduce_dimensions(matrix, method='pca', n_components=2):
    """Reduce dimensions of input matrix using PCA, UMAP, or Truncated SVD."""
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(matrix)
        
        # Get variance explained
        variance_ratios = reducer.explained_variance_ratio_ * 100
        
        # Get component loadings and create a DataFrame
        loadings = pd.DataFrame(
            reducer.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=matrix.columns
        )
        
        # Print top contributors for each component
        print("\nTop contributors to each principal component:")
        for i in range(n_components):
            print(f"\nPrincipal Component {i+1} (explains {variance_ratios[i]:.1f}% of variance)")
            # Get absolute loadings and sort
            top_contributors = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(10)
            for topic, loading in top_contributors.items():
                # Show original loading (with sign) for the top contributors
                original_loading = loadings.loc[topic, f'PC{i+1}']
                print(f"{topic}: {original_loading:.3f}")
        
        return embedding, reducer, variance_ratios
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(matrix)
        
        return embedding, reducer, None
    else:  # Truncated SVD
        n_components_initial = min(20, min(matrix.shape) - 1)  # Use more components initially
        svd_initial = TruncatedSVD(n_components=n_components_initial, random_state=42)
        matrix_transformed_initial = svd_initial.fit_transform(matrix)
        
        # Get the explained variance ratio
        explained_variance_ratio = svd_initial.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Now reduce to n_components for final ideal points
        svd_final = TruncatedSVD(n_components=n_components, random_state=42)
        embedding = svd_final.fit_transform(matrix)
        
        return embedding, svd_final, explained_variance_ratio[:n_components]

def visualize_embedding(embedding, speaker_chamber, method, n_components, subtopic_level, variance_ratios=None):
    """Visualize the embedding with party-based coloring."""
    # Load party data 
    party_df = load_party_data()
    
    # Define colors for each chamber-party combination
    colors = {
        ('S', 'D'): '#0015BC',  # Senate Democrat - Dark Blue
        ('S', 'R'): '#FF0000',  # Senate Republican - Red
        ('S', 'I'): '#006400',  # Senate Independent - Dark Green
        ('H', 'D'): '#6495ED',  # House Democrat - Light Blue
        ('H', 'R'): '#FF6B6B',  # House Republican - Light Red
        ('H', 'I'): '#90EE90',  # House Independent - Light Green
        ('S', 'UNK'): '#808080',  # Senate Unknown - Gray
        ('H', 'UNK'): '#D3D3D3'  # House Unknown - Light Gray
    }
    
    # Create markers for each chamber
    markers = {'S': '^', 'H': 'o'}  # triangle for Senate, circle for House
    
    # List of legislators to label (exact names as they appear in the data)
    legislators_to_label = ["Mr. SCHUMER", "Ms. PELOSI", "Ms. NORTON", "Mrs. MILLER-MEEKS", "Mr. McCONNELL", "Mr. SCALISE"]
    
    if n_components == 1:
        # Create figure for 1D visualization - make it simpler
        print("\nCreating 1D visualization...")
        plt.figure(figsize=(15, 8))
        
        # Process data into a clean dataframe for plotting
        plot_data = []
        
        for i, speaker in enumerate(speaker_chamber.index):
            chamber_code = speaker_chamber.loc[speaker]
            matched_chamber, matched_party = match_speaker_to_party(speaker, chamber_code, party_df)
            # Flip the dimension so positive values are on the left
            ideal_point = -1 * embedding[i, 0]  # Multiply by -1 to flip the dimension
            
            plot_data.append({
                'speaker': speaker,
                'chamber': matched_chamber,
                'party': matched_party,
                'ideal_point': ideal_point,
                'position': i  # Just use index as position
            })
        
        plot_df = pd.DataFrame(plot_data)
        print(f"Plot data shape: {plot_df.shape}")
        
        # Plot different chamber-party combinations
        for chamber in ['S', 'H']:
            for party in ['D', 'R', 'I', 'UNK']:
                subset = plot_df[(plot_df['chamber'] == chamber) & (plot_df['party'] == party)]
                
                if len(subset) == 0:
                    continue
                    
                print(f"Plotting {len(subset)} {chamber}-{party} points")
                
                color = colors.get((chamber, party), "gray")
                marker = markers.get(chamber, "o")
                
                plt.scatter(
                    subset['position'], 
                    subset['ideal_point'],
                    c=color,
                    marker=marker,
                    alpha=0.7,
                    s=100,
                    label=f"{chamber} - {party}"
                )
                
                # Add speaker labels only for specified legislators
                for _, row in subset.iterrows():
                    if row['speaker'] in legislators_to_label:
                        speaker_name = " ".join(row['speaker'].split(".", 1)[1:]) if "." in row['speaker'] else row['speaker']
                        plt.annotate(
                            speaker_name,
                            (row['position'], row['ideal_point']),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=48,
                            rotation=90
                        )
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set labels and title
        if variance_ratios is not None:
            plt.ylabel(f"{method.upper()} Dimension 1 ({variance_ratios[0]:.1f}% var)")
        else:
            plt.ylabel(f"{method.upper()} Dimension 1")
            
        title = f'1D {method.upper()} Projection of Congressional {subtopic_level.upper()} Stance Patterns'
        if variance_ratios is not None:
            title += f'\nTotal variance explained: {variance_ratios[0]:.1f}%'
        plt.title(title, fontsize=24, pad=20)
        
        # Hide x-axis ticks
        plt.xticks([])
        
        # Add legend
        plt.legend(markerscale=0.7, fontsize=16, loc='upper right')
        
        # Add explanatory text
        plt.figtext(
            0.08, 0.02,
            'Points represent speakers positioned by their stance patterns across topics.\n'
            'Only vertical position matters.\n'
            'Triangles = Senate, Circles = House\n'
            'Colors: Dark Blue = Senate (D), Red = Senate (R), Dark Green = Senate (I)\n'
            '        Light Blue = House (D), Light Red = House (R), Light Green = House (I)',
            fontsize=14,
            alpha=1
        )
        
        # Save plot
        output_prefix = f'stance_{subtopic_level}_{method.lower()}_{n_components}d'
        plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/{output_prefix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"1D plot saved to {output_prefix}.png")
        
        # Return early since we've handled the plotting
        return
    
    elif n_components == 3:
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points for each chamber-party combination
        for (chamber, party), color in colors.items():
            # Find speakers of this chamber and party using the new matching function
            mask = []
            for speaker in speaker_chamber.index:
                speaker_chamber_party = match_speaker_to_party(speaker, speaker_chamber.loc[speaker], party_df)
                mask.append(speaker_chamber_party[0] == chamber and speaker_chamber_party[1] == party)
            
            mask = np.array(mask)
            if any(mask):
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    embedding[mask, 2],
                    c=color,
                    marker=markers[chamber],
                    alpha=0.7,
                    label=f"{chamber} - {party}",
                    s=100
                )
        
        # Add labels only for specified legislators
        for i, speaker in enumerate(speaker_chamber.index):
            if speaker in legislators_to_label:
                ax.text(
                    embedding[i, 0],
                    embedding[i, 1],
                    embedding[i, 2],
                    speaker,
                    fontsize=16,
                    alpha=0.7
                )
        
        if variance_ratios is not None:
            ax.set_xlabel(f'{method.upper()}1 ({variance_ratios[0]:.1f}% var)')
            ax.set_ylabel(f'{method.upper()}2 ({variance_ratios[1]:.1f}% var)')
            ax.set_zlabel(f'{method.upper()}3 ({variance_ratios[2]:.1f}% var)')
        else:
            ax.set_xlabel(f'{method.upper()} Dimension 1')
            ax.set_ylabel(f'{method.upper()} Dimension 2')
            ax.set_zlabel(f'{method.upper()} Dimension 3')
            
    else:  # 2D visualization
        plt.figure(figsize=(20, 15))
        
        # Plot points for each chamber-party combination
        for (chamber, party), color in colors.items():
            # Find speakers of this chamber and party using the new matching function
            mask = []
            for speaker in speaker_chamber.index:
                speaker_chamber_party = match_speaker_to_party(speaker, speaker_chamber.loc[speaker], party_df)
                mask.append(speaker_chamber_party[0] == chamber and speaker_chamber_party[1] == party)
            
            mask = np.array(mask)
            if any(mask):
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=color,
                    marker=markers[chamber],
                    alpha=0.7,
                    label=f"{chamber} - {party}",
                    s=100
                )
        
        # Add labels only for specified legislators
        for i, speaker in enumerate(speaker_chamber.index):
            if speaker in legislators_to_label:
                # Get chamber information for correct marker shape
                chamber_code = speaker_chamber.loc[speaker]
                speaker_chamber_party = match_speaker_to_party(speaker, chamber_code, party_df)
                chamber = speaker_chamber_party[0]
                marker_shape = markers.get(chamber, 'o')  # Use triangle for Senate, circle for House
                
                # Add text label
                speaker_name = " ".join(speaker.split(".", 1)[1:]) if "." in speaker else speaker
                plt.annotate(
                    speaker_name,
                    (embedding[i, 0], embedding[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=16,
                    alpha=1
                )
                
                # Add bold outline with correct marker shape
                plt.scatter(
                    embedding[i, 0],
                    embedding[i, 1],
                    s=120,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=2,
                    marker=marker_shape  # This will use the correct shape (triangle or circle)
                )

        
        if variance_ratios is not None:
            plt.xlabel(f'{method.upper()}1 ({variance_ratios[0]:.1f}% var)')
            plt.ylabel(f'{method.upper()}2 ({variance_ratios[1]:.1f}% var)')
        else:
            plt.xlabel(f'{method.upper()} Dimension 1')
            plt.ylabel(f'{method.upper()} Dimension 2')
    
    title = f'{n_components}D {method.upper()} Projection of Stance-Based Ideal Points'
    if variance_ratios is not None:
        total_var = sum(variance_ratios[:n_components])
        title += f'\nTotal variance explained: {total_var:.1f}%'
    
    if n_components == 3:
        ax.set_title(title, fontsize=24, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    elif n_components == 2:  # Only apply this for 2D, not 1D
        plt.title(title, fontsize=24, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    # 1D legend is handled in the 1D visualization block
    plt.gca().invert_xaxis()
    
    # Save plot with tight layout to accommodate legend
    output_prefix = f'stance_{subtopic_level}_{method.lower()}_{n_components}d'
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(embedding, speaker_topic_matrices, method='pca', n_components=2, subtopic_level='subtopic1', filtered_speakers=None):
    """
    Save embedding and topic matrices to CSV files.
    
    Args:
        embedding: numpy array of embeddings
        speaker_topic_matrices: tuple of (speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw)
        method: dimensionality reduction method
        n_components: number of components in embedding
        subtopic_level: subtopic level used
        filtered_speakers: indices of speakers after filtering (if any)
    """
    print("\nSaving results...")
    
    # Unpack the matrices 
    speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw = speaker_topic_matrices
    
    # If we've filtered the speakers, use the filtered indices
    if filtered_speakers is not None:
        embedding_index = filtered_speakers
    else:
        embedding_index = speaker_topic_std.index
    
    # Save embedding
    embedding_df = pd.DataFrame(embedding, index=embedding_index)
    embedding_df.columns = [f'dim{i+1}' for i in range(embedding_df.shape[1])]
    embedding_df.index.name = 'speaker'
    
    # Get chamber information
    chamber_info = get_speaker_chambers()
    
    # Add to embedding_df
    embedding_df = embedding_df.reset_index()
    embedding_df = pd.merge(embedding_df, chamber_info, on='speaker', how='left')
    embedding_df = embedding_df.set_index('speaker')
    
    # Add metadata
    embedding_df['method'] = method
    embedding_df['n_components'] = n_components
    embedding_df['subtopic_level'] = subtopic_level
    
    # Save embedding files
    output_prefix = f'stance_{subtopic_level}_{method.lower()}_{n_components}d'
    embedding_df.to_csv(f'{output_prefix}.csv')
    embedding_df.to_csv(f'/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/{output_prefix}.csv')
    
    # Save topic matrices
    # 1. Final processed matrix (with filled values)
    speaker_topic_std.to_csv('speaker_topic_matrix.csv')
    speaker_topic_std.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_topic_matrix.csv')
    
    # 2. Subtopic 1 matrix (weighted averages)
    # Add chamber information to the matrix
    speaker_subtopic1 = speaker_subtopic1.reset_index()
    speaker_subtopic1 = pd.merge(speaker_subtopic1, chamber_info, on='speaker', how='left')
    speaker_subtopic1.to_csv('speaker_subtopic1_matrix.csv', index=False)
    speaker_subtopic1.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_subtopic1_matrix.csv', index=False)
    
    # 3. Raw subtopic 2 matrix (before filling missing values)
    speaker_subtopic2_raw.to_csv('speaker_subtopic2_raw_matrix.csv')
    speaker_subtopic2_raw.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_subtopic2_raw_matrix.csv')
    
    print(f"\nResults saved as:")
    print(f"- Embedding: 'stance_{subtopic_level}_{method.lower()}_{n_components}d.csv'")
    print("- Topic matrices:")
    print("  - Final processed matrix: 'speaker_topic_matrix.csv'")
    print("  - Subtopic 1 matrix: 'speaker_subtopic1_matrix.csv'")
    print("  - Raw subtopic 2 matrix: 'speaker_subtopic2_raw_matrix.csv'")

def get_speaker_chambers():
    """
    Get chamber information for each speaker from the results dataframe.
    
    Returns:
        DataFrame with speaker and chamber columns
    """
    # Load data path
    data_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/stance_predictions.csv'
    # Load just the speaker and chamber columns
    df = pd.read_csv(data_path)
    # Get first chamber for each speaker
    chamber_info = df[['speaker', 'chamber']].drop_duplicates('speaker')
    return chamber_info

def load_party_data():
    """Load party information for congress members."""
    party_df = pd.read_csv('congress_117_party.csv')
    return party_df

def match_speaker_to_party(speaker_name, chamber, party_df):
    """
    Match speaker names from stance predictions to the party dataframe.
    
    Args:
        speaker_name (str): Name in format "Mr. SCHUMER" or "Mr. JOHNSON of OHIO"
        chamber (str): Chamber ('H' or 'S')
        party_df (pd.DataFrame): DataFrame with party information
    
    Returns:
        tuple: (chamber, party) or (chamber, 'UNK') if not found
    """
    # Special case for Mr. RYAN, who is in the House with Democratic party
    if speaker_name == "Mr. RYAN":
        return ('H', 'D')
    
    # Handle empty or None values
    if not speaker_name or pd.isna(speaker_name):
        return (chamber, 'UNK')
    
    # Split the name first by period (to remove salutations)
    name_parts = speaker_name.split('.', 1)
    if len(name_parts) > 1:
        # Take the part after the period and strip whitespace
        name = name_parts[1].strip()
    else:
        name = name_parts[0].strip()
    
    # Check if there's a state mentioned (format: "JOHNSON of OHIO")
    state_parts = name.split(' of ')
    lastname = state_parts[0].strip()
    state = state_parts[1].strip() if len(state_parts) > 1 else None
    
    # Try to match by lastname and chamber
    match_by_name = party_df[(party_df['speaker'] == lastname) & (party_df['chamber'] == chamber)]
    if len(match_by_name) == 1:
        # If we have exactly one match, return it
        row = match_by_name.iloc[0]
        return (row['chamber'], row['party'])
    elif len(match_by_name) > 1 and state is not None:
        # If multiple matches and we have a state, use state to disambiguate
        match_by_state = match_by_name[match_by_name['state'] == state]
        if len(match_by_state) >= 1:
            # If we have at least one match by state, return the first one
            row = match_by_state.iloc[0]
            return (row['chamber'], row['party'])
    
    # If no match is found, print the speaker information
    print(f"No match found for speaker: '{speaker_name}', extracted lastname: '{lastname}', chamber: '{chamber}', state: '{state if state else 'N/A'}'")
    
    # Default to unknown party if no match is found
    return (chamber, 'UNK')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate stance visualizations')
    parser.add_argument('--method', type=str, choices=['pca', 'umap', 'svd'], default='pca',
                      help='Dimensionality reduction method (pca, umap, or svd)')
    parser.add_argument('--dimensions', type=int, choices=[1, 2, 3], default=2,
                      help='Number of dimensions for the projection (1, 2, or 3)')
    parser.add_argument('--subtopic-level', type=str, choices=['subtopic1', 'subtopic2'], default='subtopic1',
                      help='Which subtopic level to use for dimensionality reduction')
    parser.add_argument('--aggregation-method', type=str, choices=['simple_averaging', 'weighted_averaging', 'max_pooling'], default='simple_averaging',
                      help='Method to aggregate scores for subtopics')
    parser.add_argument('--speaker-percentile', type=float, default=0.75,
                      help='Filter to include only speakers above this percentile (0.0 = include all, 0.9 = top 10% of speakers)')
    parser.add_argument('--topic-percentile', type=float, default=0.75,
                      help='Filter to include only topics above this percentile (0.0 = include all, 0.9 = top 10% of topics)')
    args = parser.parse_args()
    
    # Set plot parameters
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['font.size'] = 16
    
    # Load and process data
    data_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/stance_predictions.csv'
    results_df, subtopic_mapping = load_and_preprocess_data(data_path)
    
    # Create topic matrices
    global speaker_topic_std  # needed for visualization function
    speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw = create_topic_matrix(results_df, subtopic_mapping, aggregation_method=args.aggregation_method)
    
    # Get chamber information
    speaker_chamber = results_df.groupby('speaker')['chamber'].first()
    
    # Select matrix based on subtopic level
    if args.subtopic_level == 'subtopic1':
        reduction_matrix = speaker_subtopic1
        print("\nUsing Subtopic 1 matrix for dimensionality reduction")
    else:  # subtopic2
        reduction_matrix = speaker_topic_std
        print("\nUsing Subtopic 2 matrix for dimensionality reduction")
    
    # Filter speakers and topics by percentile if needed
    original_shape = reduction_matrix.shape
    n_speakers_total = len(reduction_matrix)
    
    # Filter speakers if percentile threshold > 0
    if args.speaker_percentile > 0:
        # Calculate number of non-zero topics per speaker
        speaker_topic_counts = (reduction_matrix != 0).sum(axis=1)
        
        # Calculate the threshold for number of topics
        topic_threshold = speaker_topic_counts.quantile(args.speaker_percentile)
        print(f"Speaker percentile threshold {args.speaker_percentile*100:.1f}% corresponds to {topic_threshold:.0f}+ topics")
        
        # Filter to speakers with at least this many topics
        valid_speakers = speaker_topic_counts[speaker_topic_counts >= topic_threshold].index
        
        # Filter matrix to include only these speakers
        reduction_matrix = reduction_matrix.loc[valid_speakers]
        
        # Filter speaker_chamber to match the filtered speakers
        speaker_chamber = speaker_chamber[valid_speakers]
        
        print(f"Filtered from {n_speakers_total} to {len(reduction_matrix)} speakers ({len(reduction_matrix)/n_speakers_total*100:.1f}%)")
    else:
        print(f"Including all {n_speakers_total} speakers")
    
    # Calculate speaker coverage by topic
    if args.topic_percentile > 0:
        # Count number of non-zero values per topic
        topic_speaker_counts = (reduction_matrix != 0).sum(axis=0)
        
        # Calculate threshold
        speaker_threshold = topic_speaker_counts.quantile(args.topic_percentile)
        print(f"Topic percentile threshold {args.topic_percentile*100:.1f}% corresponds to {speaker_threshold:.0f}+ speakers")
        
        # Get valid topics (columns with enough speakers)
        valid_topics = topic_speaker_counts[topic_speaker_counts >= speaker_threshold].index
        
        # Filter the matrix
        original_topic_count = reduction_matrix.shape[1]
        reduction_matrix = reduction_matrix[valid_topics]
        
        print(f"Filtered from {original_topic_count} to {len(valid_topics)} topics ({len(valid_topics)/original_topic_count*100:.1f}%)")
        for curr in valid_topics:
            print(curr)
    
    # Reduce dimensions
    embedding, reducer, variance_ratios = reduce_dimensions(reduction_matrix, method=args.method, n_components=args.dimensions)
    
    # Make sure speaker_chamber has the same number of entries as the embedding
    if len(speaker_chamber) != embedding.shape[0]:
        print(f"Warning: speaker_chamber has {len(speaker_chamber)} entries but embedding has {embedding.shape[0]} rows")
        print("This may happen if speakers were filtered based on percentile. Using only the speakers in the embedding.")
        # Only keep the speakers that are in the embedding
        embedding_df = pd.DataFrame(embedding, index=reduction_matrix.index)
        shared_speakers = set(speaker_chamber.index).intersection(set(embedding_df.index))
        speaker_chamber = speaker_chamber.loc[list(shared_speakers)]
    
    # Visualize results
    visualize_embedding(embedding, speaker_chamber, args.method, args.dimensions, args.subtopic_level, variance_ratios)
    
    # Save results - pass the filtered indices if we've filtered speakers
    filtered_speakers = reduction_matrix.index if args.speaker_percentile > 0 else None
    save_results(embedding, (speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw), 
                args.method, args.dimensions, args.subtopic_level, filtered_speakers)
    
    print(f"\nGenerated {args.dimensions}D {args.method.upper()} visualization for {args.subtopic_level}")
    if variance_ratios is not None:
        print("\nVariance explained by each component:")
        for i, var in enumerate(variance_ratios):
            print(f"Component {i+1}: {var:.1f}%")
        print(f"Total: {sum(variance_ratios):.1f}%")
    
    print(f"\nResults saved as:")
    print(f"- Embedding: 'stance_{args.subtopic_level}_{args.method.lower()}_{args.dimensions}d.csv'")
    print("- Topic matrices:")
    print("  - Final processed matrix: 'speaker_topic_matrix.csv'")
    print("  - Subtopic 1 matrix: 'speaker_subtopic1_matrix.csv'")
    print("  - Raw subtopic 2 matrix: 'speaker_subtopic2_raw_matrix.csv'")

if __name__ == '__main__':
    main()
