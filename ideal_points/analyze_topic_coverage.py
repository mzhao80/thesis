import pandas as pd
import numpy as np

def analyze_topic_coverage(matrix_path):
    """
    Analyze what proportion of speakers have stances on each topic.
    
    Args:
        matrix_path: Path to the speaker-subtopic matrix CSV
    """
    # Read the matrix
    df = pd.read_csv(matrix_path, index_col=0)
    
    # Calculate proportion of non-zero entries for each topic (column)
    coverage = {}
    for topic in df.columns:
        col = df[topic]
        non_zero = (col != 0).sum()
        total = len(col)
        coverage[topic] = non_zero / total
    
    # Convert to dataframe and sort
    coverage_df = pd.DataFrame.from_dict(coverage, orient='index', columns=['coverage'])
    coverage_df = coverage_df.sort_values('coverage', ascending=False)
    
    # Print results
    print("\nTopic Coverage Analysis")
    print("=" * 100)
    print(f"Total number of topics: {len(df.columns)}")
    print(f"Total number of speakers: {len(df.index)}")
    print("\nCoverage by topic (proportion of speakers with non-zero stance):")
    print("-" * 100)
    
    for topic, row in coverage_df.iterrows():
        coverage_pct = row['coverage'] * 100
        num_speakers = int((df[topic] != 0).sum())
        print(f"{topic:<70} {coverage_pct:>6.1f}% ({num_speakers:>3} speakers)")
    
    print("\nSummary Statistics")
    print("-" * 100)
    print(f"Mean coverage: {coverage_df['coverage'].mean()*100:.1f}%")
    print(f"Median coverage: {coverage_df['coverage'].median()*100:.1f}%")
    print(f"Std deviation: {coverage_df['coverage'].std()*100:.1f}%")
    
if __name__ == "__main__":
    matrix_path = "speaker_subtopic1_matrix.csv"
    analyze_topic_coverage(matrix_path)
