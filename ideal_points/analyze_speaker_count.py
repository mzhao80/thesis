import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv')

# Drop rows that don't have a 'policy_area'
df = df.dropna(subset=['policy_area'])

# Count the number of speeches for each chamber
count_H = df[df['chamber'] == 'H'].shape[0]
count_S = df[df['chamber'] == 'S'].shape[0]
print("Number of speeches with chamber H:", count_H)
print("Number of speeches with chamber S:", count_S)

# Group by speaker and chamber to count the speeches
speaker_counts = df.groupby(['speaker', 'chamber']).size().unstack(fill_value=0)

# Create a 'total' column to sort speakers by overall speech count (descending)
speaker_counts['total'] = speaker_counts.sum(axis=1)
speaker_counts = speaker_counts.sort_values('total', ascending=False)
# Remove the total column now that sorting is done
speaker_counts = speaker_counts.drop(columns='total')

# Prepare data for the grouped bar chart:
# For each speaker group, we want the two bars (H and S) to touch. 
# We'll define each group to have a width of 1 unit and each bar will be 0.5.
n = len(speaker_counts)
group_width = 1.0  
bar_width = group_width / 2  # equals 0.5

# Create x positions for each speaker group. Adding a small gap (e.g., 0.1) between groups.
x = np.arange(n) * (group_width + 0.1)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for chamber H and chamber S with specified colors.
bars_H = ax.bar(x, speaker_counts.get('H', pd.Series([0]*n)), 
                width=bar_width, color='orange', label='H')
bars_S = ax.bar(x + bar_width, speaker_counts.get('S', pd.Series([0]*n)), 
                width=bar_width, color='green', label='S')

ax.set_ylabel('Number of Speeches')
ax.set_title('Distribution of Speeches by Speaker and Chamber')
# Set xticks at the center of each group; but leave them blank as requested.
ax.set_xticks(x + bar_width/2)
ax.set_xticklabels([''] * n)
ax.legend()

# Annotate the top 5 speakers (by total speeches) on the plot.
top5 = speaker_counts.head(5)
top5_total = top5.sum(axis=1)
top5_text = "Top 5 speakers:\n"
for i, (speaker, total) in enumerate(top5_total.items(), start=1):
    top5_text += f"{i}. {speaker}: {total}\n"

# Add the text box to the upper right corner of the plot
ax.text(0.95, 0.95, top5_text, transform=ax.transAxes, verticalalignment='top',
        horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('speaker_distribution.png')
plt.show()
