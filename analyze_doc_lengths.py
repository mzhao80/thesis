import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np

# Load data
data_path = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv"
df = pd.read_csv(data_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/nv-embed-v2")

# Tokenize documents and compute lengths
df["token_length"] = df["document"].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))

# Compute quantiles
quantiles = np.percentile(df["token_length"], [0, 5, 10, 25, 50, 75, 90, 95, 99, 100])
quantile_labels = ["0%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "100%"]

print("Document Length Quantiles:")
for label, value in zip(quantile_labels, quantiles):
    print(f"{label}: {value}")

# Identify outliers using the IQR method
q1 = np.percentile(df["token_length"], 25)
q3 = np.percentile(df["token_length"], 75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

outliers = df[df["token_length"] > upper_bound].sort_values(by="token_length", ascending=False)

# Compute fraction of dataset represented by outliers
outlier_fraction = len(outliers) / len(df)

print("\nOutliers (Top 10 by token length):")
print(outliers[["document", "token_length"]].head(10))
print(f"\nFraction of dataset represented by outliers: {outlier_fraction:.4f}")

# Remove documents above the 99% token length threshold
threshold_99 = np.percentile(df["token_length"], 99)
df_filtered = df[df["token_length"] <= threshold_99]

# Save the filtered dataset
output_path = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/new_training_data.csv"
df_filtered.to_csv(output_path, index=False)
print(f"Filtered dataset saved to {output_path}")

