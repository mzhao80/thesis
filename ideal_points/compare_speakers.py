#!/usr/bin/env python3
import pandas as pd

# Define file paths
stance_predictions_path = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/stance_predictions.csv'
congress_party_path = 'congress_117_party.csv'

# Read the CSV files
try:
    stance_df = pd.read_csv(stance_predictions_path)
    congress_df = pd.read_csv(congress_party_path)
    
    print(f"Successfully loaded files:")
    print(f"  - {stance_predictions_path}: {len(stance_df)} rows")
    print(f"  - {congress_party_path}: {len(congress_df)} rows")
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Process speaker names in stance_predictions file - remove the first word (salutation)
def remove_salutation(speaker_name):
    if isinstance(speaker_name, str):
        # Split by space and join all parts except the first one
        parts = speaker_name.split()
        if len(parts) > 1:
            return ' '.join(parts[1:])  # Skip the first word (salutation)
    return speaker_name  # Return as is if not a string or has no spaces

# Apply the transformation to get processed speaker names
stance_df['processed_speaker'] = stance_df['speaker'].apply(remove_salutation)

# Get unique processed speaker names from stance predictions
stance_speakers = set(stance_df['processed_speaker'].dropna().unique())

# Get unique speaker names from congress party file
congress_speakers = set(congress_df['speaker'].dropna().unique())

# Find speakers that are in stance predictions but not in congress party file
missing_speakers = stance_speakers - congress_speakers

# Output the results
print(f"\nFound {len(missing_speakers)} speakers in stance predictions that are not in congress party file:")
for speaker in sorted(missing_speakers):
    print(f"  - {speaker}")

print(f"\nMissing speakers have been saved to 'missing_speakers.txt'")
