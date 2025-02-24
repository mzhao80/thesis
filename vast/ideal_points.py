import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
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
    
    # Create combined weight for subtopic_1 aggregation
    results_df['combined_weight'] = results_df['document_count'] * results_df['stance_confidence']
    
    # Create mapping of subtopic_2 to their parent subtopic_1
    subtopic_mapping = results_df[['subtopic_1', 'subtopic_2']].drop_duplicates()
    subtopic_mapping = subtopic_mapping.set_index('subtopic_2')['subtopic_1'].to_dict()
    
    return results_df, subtopic_mapping

def create_topic_matrix(results_df, subtopic_mapping):
    """Create the topic matrices with weighted scores."""
    # Get direct stance scores for subtopic_2
    speaker_subtopic2 = results_df.pivot_table(
        index='speaker',
        columns='subtopic_2',
        values='stance_score',
        aggfunc='first'
    )
    
    # Calculate weighted average scores for each speaker-subtopic_1 combination
    speaker_subtopic1 = pd.DataFrame(index=speaker_subtopic2.index)
    for subtopic1 in results_df['subtopic_1'].unique():
        subtopic1_data = results_df[results_df['subtopic_1'] == subtopic1]
        weighted_avg = subtopic1_data.groupby('speaker').apply(
            lambda x: np.average(x['stance_score'], weights=x['combined_weight'])
            if len(x) > 0 else np.nan
        )
        speaker_subtopic1[subtopic1] = weighted_avg
    
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
    """Reduce dimensions of input matrix using PCA or UMAP."""
    # Scale the data to [-1, 1] range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(matrix)
        
        # Scale the embedding
        embedding_scaled = scaler.fit_transform(embedding)
        
        # Calculate variance explained
        variance_ratios = reducer.explained_variance_ratio_ * 100
        
        return embedding_scaled, reducer, variance_ratios
    else:  # umap
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(matrix)
        
        # Scale the embedding
        embedding_scaled = scaler.fit_transform(embedding)
        
        return embedding_scaled, reducer, None

def visualize_embedding(embedding, speaker_data, method, n_components, subtopic_level, variance_ratios=None):
    """Visualize the embedding with party-based coloring."""
    # Skip visualization for 1D
    if n_components == 1:
        return
        
    party_info = load_party_data()
    
    # Define colors for each chamber-party combination
    colors = {
        ('S', 'D'): '#0015BC',  # Senate Democrat - Dark Blue
        ('S', 'R'): '#FF0000',  # Senate Republican - Red
        ('S', 'I'): '#006400',  # Senate Independent - Dark Green
        ('H', 'D'): '#6495ED',  # House Democrat - Light Blue
        ('H', 'R'): '#FF6B6B',  # House Republican - Light Red
        ('H', 'I'): '#90EE90'   # House Independent - Light Green
    }
    
    # Create markers for each chamber
    markers = {'S': 'o', 'H': 's'}  # circle for Senate, square for House
    
    if n_components == 3:
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points for each chamber-party combination
        for (chamber, party), color in colors.items():
            # Find speakers of this chamber and party
            mask = [party_info.get(speaker, (chamber, 'Unknown'))[0] == chamber and 
                   party_info.get(speaker, (chamber, 'Unknown'))[1] == party 
                   for speaker in speaker_data.index]
            if any(mask):
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    embedding[mask, 2],
                    c=color,
                    marker=markers[chamber],
                    alpha=0.6,
                    label=f"{chamber} - {party}",
                    s=100
                )
        
        # Add labels for each point
        for i, speaker in enumerate(speaker_data.index):
            speaker_name = speaker.replace('Mr. ', '').replace('Ms. ', '').replace('Mrs. ', '')
            ax.text(
                embedding[i, 0],
                embedding[i, 1],
                embedding[i, 2],
                speaker_name,
                fontsize=8,
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
            # Find speakers of this chamber and party
            mask = [party_info.get(speaker, (chamber, 'Unknown'))[0] == chamber and 
                   party_info.get(speaker, (chamber, 'Unknown'))[1] == party 
                   for speaker in speaker_data.index]
            if any(mask):
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=color,
                    marker=markers[chamber],
                    alpha=0.6,
                    label=f"{chamber} - {party}",
                    s=100
                )
        
        # Add labels for each point
        for i, speaker in enumerate(speaker_data.index):
            speaker_name = speaker.replace('Mr. ', '').replace('Ms. ', '').replace('Mrs. ', '')
            plt.annotate(
                speaker_name,
                (embedding[i, 0], embedding[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        if variance_ratios is not None:
            plt.xlabel(f'{method.upper()}1 ({variance_ratios[0]:.1f}% var)')
            plt.ylabel(f'{method.upper()}2 ({variance_ratios[1]:.1f}% var)')
        else:
            plt.xlabel(f'{method.upper()} Dimension 1')
            plt.ylabel(f'{method.upper()} Dimension 2')
    
    title = f'{n_components}D {method.upper()} Projection of Congressional {subtopic_level.upper()} Stance Patterns'
    if variance_ratios is not None:
        total_var = sum(variance_ratios[:n_components])
        title += f'\nTotal variance explained: {total_var:.1f}%'
    
    if n_components == 3:
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.title(title, fontsize=14, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    # Add explanation
    plt.figtext(
        0.02, 0.02,
        'Points represent speakers positioned by their stance patterns across topics.\n'
        'Proximity indicates similar voting/speaking patterns.\n'
        'Circles = Senate, Squares = House\n'
        'Colors: Dark Blue = Senate (D), Red = Senate (R), Dark Green = Senate (I)\n'
        '        Light Blue = House (D), Light Red = House (R), Light Green = House (I)',
        fontsize=10,
        alpha=0.7
    )
    
    # Save plot with tight layout to accommodate legend
    output_prefix = f'stance_{subtopic_level}_{method.lower()}_{n_components}d'
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(embedding, speaker_topic_matrices, method, n_components, subtopic_level):
    """Save embedding coordinates and topic matrices."""
    speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw = speaker_topic_matrices
    
    # Load party information
    party_info = load_party_data()
    
    # Create DataFrame with embedding coordinates
    cols = [f'{method.upper()}{i+1}' for i in range(n_components)]
    embedding_df = pd.DataFrame(
        embedding if n_components == 1 else embedding,
        columns=cols,
        index=speaker_topic_std.index
    )
    
    # Add chamber and party information
    embedding_df['chamber'] = [party_info.get(speaker, ('Unknown', 'Unknown'))[0] for speaker in embedding_df.index]
    embedding_df['party'] = [party_info.get(speaker, ('Unknown', 'Unknown'))[1] for speaker in embedding_df.index]
    
    # Sort by first dimension if 1D
    if n_components == 1:
        embedding_df = embedding_df.sort_values(by=cols[0])
    
    # Save embedding files
    output_prefix = f'stance_{subtopic_level}_{method.lower()}_{n_components}d'
    embedding_df.to_csv(f'{output_prefix}.csv')
    embedding_df.to_csv(f'/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/{output_prefix}.csv')
    
    # Save topic matrices
    # 1. Final processed matrix (with filled values)
    speaker_topic_std.to_csv('speaker_topic_matrix.csv')
    speaker_topic_std.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_topic_matrix.csv')
    
    # 2. Subtopic 1 matrix (weighted averages)
    speaker_subtopic1.to_csv('speaker_subtopic1_matrix.csv')
    speaker_subtopic1.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_subtopic1_matrix.csv')
    
    # 3. Raw subtopic 2 matrix (before filling missing values)
    speaker_subtopic2_raw.to_csv('speaker_subtopic2_raw_matrix.csv')
    speaker_subtopic2_raw.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_subtopic2_raw_matrix.csv')

def load_party_data():
    """Load party information for congress members."""
    party_df = pd.read_csv('congress_117_party.csv')
    # Create a dictionary mapping speaker to (chamber, party)
    return {row['speaker']: (row['chamber'], row['party']) 
            for _, row in party_df.iterrows()}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate stance visualizations')
    parser.add_argument('--method', type=str, choices=['pca', 'umap'], default='pca',
                      help='Dimensionality reduction method (pca or umap)')
    parser.add_argument('--dimensions', type=int, choices=[1, 2, 3], default=2,
                      help='Number of dimensions for the projection (1, 2, or 3)')
    parser.add_argument('--subtopic-level', type=str, choices=['subtopic1', 'subtopic2'], default='subtopic2',
                      help='Which subtopic level to use for dimensionality reduction')
    args = parser.parse_args()
    
    # Set plot parameters
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['font.size'] = 8
    
    # Load and process data
    data_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_stance_results.csv'
    results_df, subtopic_mapping = load_and_preprocess_data(data_path)
    
    # Create topic matrices
    global speaker_topic_std  # needed for visualization function
    speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw = create_topic_matrix(results_df, subtopic_mapping)
    
    # Select matrix based on subtopic level
    if args.subtopic_level == 'subtopic1':
        reduction_matrix = speaker_subtopic1
        print("\nUsing Subtopic 1 matrix for dimensionality reduction")
    else:  # subtopic2
        reduction_matrix = speaker_topic_std
        print("\nUsing Subtopic 2 matrix for dimensionality reduction")
    
    # Get chamber information
    speaker_chamber = results_df.groupby('speaker')['chamber'].first()
    
    # Reduce dimensions
    embedding, reducer, variance_ratios = reduce_dimensions(reduction_matrix, method=args.method, n_components=args.dimensions)
    
    # Visualize results
    visualize_embedding(embedding, speaker_chamber, args.method, args.dimensions, args.subtopic_level, variance_ratios)
    
    # Save results
    save_results(embedding, (speaker_topic_std, speaker_subtopic1, speaker_subtopic2_raw), 
                args.method, args.dimensions, args.subtopic_level)
    
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
