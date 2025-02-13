import pandas as pd
import numpy as np

# Load data
data_path = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/df_bills.csv"
df = pd.read_csv(data_path)
# print out proportion of NAs in policy_area
print(df["policy_area"].isnull().sum() / len(df))
df = df.dropna(subset=["policy_area"])
df.to_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/df_bills.csv", index=False)