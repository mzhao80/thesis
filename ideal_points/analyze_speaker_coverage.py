import pandas as pd
import numpy as np

def analyze_speaker_coverage(matrix_path):
    """
    Analyze what proportion of subtopics each speaker has stances on.
    
    Args:
        matrix_path: Path to the speaker-subtopic matrix CSV
    """
    # Read the matrix
    df = pd.read_csv(matrix_path, index_col=0)
    
    # Calculate total coverage statistics
    total_cells = df.shape[0] * df.shape[1]
    zero_cells = (df == 0).sum().sum()
    missing_percentage = (zero_cells / total_cells) * 100
    covered_percentage = 100 - missing_percentage
    
    # Calculate proportion of non-zero entries for each speaker
    coverage = {}
    for speaker in df.index:
        row = df.loc[speaker]
        non_zero = (row != 0).sum()
        total = len(row)
        coverage[speaker] = non_zero / total
    
    # Convert to dataframe and sort
    coverage_df = pd.DataFrame.from_dict(coverage, orient='index', columns=['coverage'])
    coverage_df = coverage_df.sort_values('coverage', ascending=False)
    
    # Print results
    print("\nSpeaker Coverage Analysis")
    print("=" * 80)
    print(f"Total number of subtopics: {len(df.columns)}")
    print(f"Total number of speakers: {len(df.index)}")
    print(f"Total possible speaker-topic pairs: {total_cells}")
    print(f"Total zero-value pairs: {zero_cells}")
    print(f"Percentage of speaker-topic pairs lacking coverage: {missing_percentage:.1f}%")
    print(f"Percentage of speaker-topic pairs with coverage: {covered_percentage:.1f}%")
    print("\nCoverage by speaker (proportion of subtopics with non-zero stance):")
    print("-" * 80)
    
    for speaker, row in coverage_df.iterrows():
        coverage_pct = row['coverage'] * 100
        num_topics = int((df.loc[speaker] != 0).sum())
        print(f"{speaker:<50} {coverage_pct:>6.1f}% ({num_topics:>3} topics)")
    
    print("\nSummary Statistics")
    print("-" * 80)
    print(f"Mean coverage: {coverage_df['coverage'].mean()*100:.1f}%")
    print(f"Median coverage: {coverage_df['coverage'].median()*100:.1f}%")
    print(f"Std deviation: {coverage_df['coverage'].std()*100:.1f}%")
    
if __name__ == "__main__":
    matrix_path = "speaker_subtopic1_matrix.csv"
    analyze_speaker_coverage(matrix_path)
