#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def sample_csv(input_file, output_file, sample_size=100, random_state=42):
    """
    Reads a CSV file, randomly samples a specified number of rows,
    and writes the sampled data to a new CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to the output CSV file
    sample_size : int, default=100
        Number of rows to randomly sample
    random_state : int, default=42
        Random seed for reproducibility
    """
    # Read the CSV file
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    # filter only df with non-empty target
    df = df[df['target'].notna()]
    
    # Get the total number of rows
    total_rows = len(df)
    print(f"Total rows in original file: {total_rows}")
    
    # Sample rows (if sample_size > total_rows, sample_frac=1 will return all rows)
    if sample_size >= total_rows:
        print(f"Warning: Requested sample size ({sample_size}) is greater than or equal to the total number of rows ({total_rows}). All rows will be kept.")
        sampled_df = df
    else:
        print(f"Randomly sampling {sample_size} rows")
        sampled_df = df.sample(n=sample_size, random_state=random_state)
    
    # Write to output file
    print(f"Writing sampled data to {output_file}")
    sampled_df.to_csv(output_file, index=False)
    print(f"Successfully wrote {len(sampled_df)} rows to {output_file}")

if __name__ == "__main__":
    input_file = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_2023.csv"
    output_file = "/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_2023_sample.csv"
    
    sample_csv(input_file, output_file)
