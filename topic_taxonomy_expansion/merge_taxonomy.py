import pandas as pd

# Read the taxonomy data
taxonomy_df = pd.read_csv('step_4.csv')

# Read the training data
training_df = pd.read_csv('step_1.csv')

# Create a mapping dictionary from source_indices to taxonomy information
taxonomy_mapping = {}
for idx, row in taxonomy_df.iterrows():
    indices = [int(i.strip()) for i in row['source_indices'].split(';')]
    for index in indices:
        taxonomy_mapping[index] = {
            'idx': index,
            'policy_area': row['policy_area'],
            'subtopic_1': row['subtopic_1'],
            'subtopic_2': row['subtopic_2']
        }

# Merge taxonomy_mapping dict into training_df on idx key
final_df = training_df.merge(
    pd.DataFrame.from_dict(taxonomy_mapping, orient='index'),
    left_index=True,
    right_on='idx',
    how='left'
)

# Define target column logic
def determine_target(row):
    if row['subtopic_2']:
        if row['subtopic_2'] != 'Misc.':
            return row['subtopic_2']
    # elif row['subtopic_1'] != 'Misc.':
    #     return row['subtopic_1']
    return ""

final_df['target'] = final_df.apply(determine_target, axis=1)

# sort final_df by idx
final_df = final_df.sort_values(by='idx')

# Save the merged dataset
final_df.to_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv', index=False)
print("Merged dataset saved successfully with 'target' column added!")
