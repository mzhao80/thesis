import pandas as pd

# Read the taxonomy data
taxonomy_df = pd.read_csv('step_4.csv')

# Read the training data
training_df = pd.read_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data_2023.csv')

# Create a mapping dictionary from source_indices to taxonomy information
taxonomy_mapping = {}
for idx, row in taxonomy_df.iterrows():
    indices = [int(i.strip()) for i in row['source_indices'].split(';')]
    for index in indices:
        taxonomy_mapping[index] = {
            'policy_area': row['policy_area'],
            'subtopic_1': row['subtopic_1'],
            'subtopic_2': row['subtopic_2']
        }

# Add new columns to training_df
training_df['policy_area'] = training_df.index.map(lambda x: taxonomy_mapping.get(x, {}).get('policy_area', ''))
training_df['subtopic_1'] = training_df.index.map(lambda x: taxonomy_mapping.get(x, {}).get('subtopic_1', ''))
training_df['subtopic_2'] = training_df.index.map(lambda x: taxonomy_mapping.get(x, {}).get('subtopic_2', ''))

# Define target column logic
def determine_target(row):
    if row['subtopic_2'] and row['subtopic_2'] != 'Misc.':
        return row['subtopic_2']
    elif row['subtopic_1'] and row['subtopic_1'] != 'Misc.':
        return row['subtopic_1']
    else:
        return row['policy_area']

training_df['target'] = training_df.apply(determine_target, axis=1)

# Save the merged dataset
training_df.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv', index=False)
print("Merged dataset saved successfully with 'target' column added!")
