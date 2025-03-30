#!/usr/bin/env python3
"""
StrideStance Ideal Points: Stance-Based Item Response Theory (IRT) Implementation

This script implements a Bayesian Item Response Theory (IRT) model for estimating legislator 
ideal points from stance detection outputs. Unlike traditional vote-based IRT approaches,
this model uses continuous stance scores as the observed data, adapting the classical
IRT framework to accommodate the nuanced expression of political positions in speech.

The theoretical foundation extends Clinton, Jackman, and Rivers (2004) spatial voting 
theory to the domain of political speech. In this adaptation:
- Topics serve as "items" (analogous to roll-call votes)
- Speakers serve as "subjects" with latent ideal points (θ)
- Stance scores represent the observed data (y)
- Each topic has discrimination (a) and difficulty (b) parameters

Model specification:
  y_{j,i} = a_i * θ_j - b_i + ε_{j,i},
  where:
    θ_j       = latent ideal point for speaker j
    a_i, b_i  = discrimination & difficulty for topic i
    y_{j,i}   = stance score for speaker j on topic i
    ε_{j,i}   ~ Normal(0, σ²)

Key features:
1. Filtering topics and speakers based on coverage thresholds
2. Bayesian estimation using PyMC for uncertainty quantification
3. Visualization of ideal points with party-based coloring
4. Output of posterior means and credible intervals
5. Extraction of topic discrimination parameters for interpretation

The model addresses data sparsity through filtering criteria and robust priors, and
provides uncertainty estimates (95% credible intervals) for all parameters.

INPUT:
  speaker_subtopic1_matrix.csv
      - Wide format CSV with:
          - The first column named "speaker"
          - Subsequent columns named after each subtopic_1
          - Each cell has a numeric stance score for (speaker, subtopic_1).

OUTPUT:
  speaker_subtopic1_idealpoints.csv
      - CSV mapping each speaker to their estimated ideal point (posterior mean).
  speaker_subtopic1_discrimination.csv
      - CSV mapping each topic to its estimated discrimination parameter (posterior mean).
  speaker_subtopic1_idealpoints_uncertainty.csv
      - CSV with uncertainy estimates (lower/upper bounds of 95% credible intervals).

Usage:
  python irt.py --input_csv speaker_subtopic1_matrix.csv
               --output_csv speaker_subtopic1_idealpoints.csv
               --party_csv congress_117_party.csv
               --plot_png stance_1d_plot.png
               --speaker_percentile 0.75
               --topic_percentile 0.75
"""

import pandas as pd
import numpy as np
import pymc as pm
import jax
import arviz as az
import multiprocessing
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def run_continuous_irt_subtopic1(
    input_csv: str = "speaker_subtopic1_matrix.csv",
    output_csv: str = "speaker_subtopic1_idealpoints.csv",
    party_csv: str = "congress_117_party.csv",
    plot_png: str = "stance_1d_plot.png",
    coverage_cutoff: float = 0.10,  # 10% coverage threshold
    speaker_percentile: float = 0.0, # 0.0 = include all, 0.9 = top 10% of speakers
    topic_percentile: float = 0.0,   # 0.0 = include all, 0.9 = top 10% of topics
    party_filter: str = "",        # Filter by party (D, R, or "" for all)
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9
):
    """
    Reads a wide-format CSV of speaker–subtopic_1 stance data and fits a 
    continuous 1D Item Response Theory (IRT) model using PyMC.

    Args:
        input_csv: Path to input CSV (wide format, speakers in rows, topics in columns)
        output_csv: Path to save speaker ideal points to (CSV)
        party_csv: Path to party CSV mapping speakers to parties
        plot_png: Path to save 1D plot to (PNG)
        coverage_cutoff: Topic coverage threshold (0.1 = 10%)
        speaker_percentile: Filter to include only speakers above this percentile
                           based on number of non-zero topics (0 = include all)
        topic_percentile: Filter to include only topics above this percentile
                         based on number of non-zero speakers (0 = include all)
        party_filter: Filter by party (D, R, or "" for all)
        draws: Number of posterior samples to draw after tuning
        tune: Number of tuning steps for MCMC
        target_accept: Target acceptance rate for MCMC
    """
    # -------------------------------------------------------------------------
    # 1. Load and preprocess the data
    # -------------------------------------------------------------------------
    print(f"Loading data from {input_csv}...")
    
    # Load the wide-format data
    df_wide = pd.read_csv(input_csv)
    
    # Identify metadata columns (speaker, chamber) vs. subtopic columns
    metadata_cols = ["speaker", "chamber"] if "chamber" in df_wide.columns else ["speaker"]
    subtopic_cols = [c for c in df_wide.columns if c not in metadata_cols]
    
    # Melt the data to long format: each row has (speaker, subtopic_1, stance_score)
    df_long = df_wide.melt(
        id_vars=metadata_cols,
        value_vars=subtopic_cols,
        var_name="subtopic_1",
        value_name="stance_score"
    )
    
    # Drop rows with missing stance scores
    df_long = df_long.dropna(subset=["stance_score"])
    
    # Total number of speakers in the dataset
    all_speakers = df_long["speaker"].unique()
    n_speakers_total = len(all_speakers)
    
    # -------------------------------------------------------------------------
    # 2. Load Party Data and Apply Party Filter (if specified)
    # -------------------------------------------------------------------------
    if party_filter:
        print(f"Loading party data and filtering to only include {party_filter} party members...")
        party_info = load_party_data(party_csv=party_csv)
        
        # Create temporary chamber codes for matching
        df_long['chamber_code'] = 'UNK'
        if "chamber" in df_long.columns:
            df_long.loc[df_long['chamber'].str.lower().str.startswith('s'), 'chamber_code'] = 'S'
            df_long.loc[df_long['chamber'].str.lower().str.startswith('h'), 'chamber_code'] = 'H'
        
        # Get unique speakers for party matching
        unique_speakers = df_long[['speaker', 'chamber_code']].drop_duplicates()
        
        # Match each speaker to their party
        speaker_parties = {}
        for _, row in unique_speakers.iterrows():
            speaker = row['speaker']
            chamber_code = row['chamber_code']
            _, party = match_speaker_to_party(speaker, chamber_code, party_info)
            speaker_parties[speaker] = party
        
        # Filter df_long to only include speakers of the specified party
        filtered_speakers = [speaker for speaker, party in speaker_parties.items() 
                           if party == party_filter]
        
        if not filtered_speakers:
            print(f"Warning: No speakers found with party '{party_filter}'. Proceeding with all speakers.")
        else:
            n_before = len(df_long['speaker'].unique())
            df_long = df_long[df_long['speaker'].isin(filtered_speakers)]
            n_after = len(df_long['speaker'].unique())
            print(f"Filtered from {n_before} to {n_after} speakers based on party filter")
        
        # Modify output filenames to reflect party filter
        base_name, ext = os.path.splitext(output_csv)
        output_csv = f"{base_name}_{party_filter}{ext}"
        
        base_name, ext = os.path.splitext(plot_png)
        plot_png = f"{base_name}_{party_filter}{ext}"
    
    # -------------------------------------------------------------------------
    # 3. Filter Speakers and Topics based on Percentile Thresholds
    # -------------------------------------------------------------------------
    # Calculate topics with non-zero stance scores per speaker
    # Only count topics that have non-zero stance scores
    speakers_with_nonzero_topics = df_long[df_long["stance_score"] != 0].groupby("speaker")["subtopic_1"].nunique().reset_index(name="num_topics")
    
    # Merge with all speakers to include those with zero topics
    all_speaker_df = pd.DataFrame({"speaker": all_speakers})
    speaker_topic_counts = all_speaker_df.merge(speakers_with_nonzero_topics, how="left", on="speaker").fillna(0)
    
    # Debug: Print topic count statistics 
    print("\nDEBUG - Non-zero topic counts per speaker:")
    print(f"Min topics: {speaker_topic_counts['num_topics'].min()}")
    print(f"Max topics: {speaker_topic_counts['num_topics'].max()}")
    print(f"Median topics: {speaker_topic_counts['num_topics'].median()}")
    print(f"Mean topics: {speaker_topic_counts['num_topics'].mean():.1f}")
    print(f"25th percentile: {speaker_topic_counts['num_topics'].quantile(0.25)}")
    print(f"75th percentile: {speaker_topic_counts['num_topics'].quantile(0.75)}")
    print(f"90th percentile: {speaker_topic_counts['num_topics'].quantile(0.9)}")
    
    # Filter speakers if percentile threshold > 0
    if speaker_percentile > 0:
        # Calculate the threshold for number of topics
        topic_threshold = speaker_topic_counts["num_topics"].quantile(speaker_percentile)
        print(f"Speaker percentile threshold {speaker_percentile*100:.1f}% corresponds to {topic_threshold:.0f}+ topics")
        
        # Get list of speakers above percentile threshold
        valid_speakers = speaker_topic_counts.loc[
            speaker_topic_counts["num_topics"] >= topic_threshold, 
            "speaker"
        ]
        
        # Filter dataframe to include only these speakers
        df_long = df_long[df_long["speaker"].isin(valid_speakers)]
        
        print(f"Filtered from {n_speakers_total} to {len(valid_speakers)} speakers ({len(valid_speakers)/n_speakers_total*100:.1f}%)")
    else:
        print(f"Including all {n_speakers_total} speakers")
    
    # Calculate speaker coverage by topic, only counting speakers with non-zero stance scores
    nonzero_topic_coverage = df_long[df_long["stance_score"] != 0].groupby("subtopic_1")["speaker"].nunique().reset_index(name="num_speakers")
    
    # Merge with all topics to include those with zero speakers
    all_topics = df_long["subtopic_1"].unique()
    all_topic_df = pd.DataFrame({"subtopic_1": all_topics})
    topic_speaker_counts = all_topic_df.merge(nonzero_topic_coverage, how="left", on="subtopic_1").fillna(0)
    
    all_speakers_filtered = df_long["speaker"].unique()
    n_speakers_filtered = len(all_speakers_filtered)
    topic_speaker_counts["coverage"] = topic_speaker_counts["num_speakers"] / float(n_speakers_filtered)
    
    # Debug: Print speaker count statistics for topics
    print("\nDEBUG - Speaker counts per topic:")
    print(f"Min speakers: {topic_speaker_counts['num_speakers'].min()}")
    print(f"Max speakers: {topic_speaker_counts['num_speakers'].max()}")
    print(f"Median speakers: {topic_speaker_counts['num_speakers'].median()}")
    print(f"Mean speakers: {topic_speaker_counts['num_speakers'].mean():.1f}")
    print(f"25th percentile: {topic_speaker_counts['num_speakers'].quantile(0.25)}")
    print(f"75th percentile: {topic_speaker_counts['num_speakers'].quantile(0.75)}")
    print(f"90th percentile: {topic_speaker_counts['num_speakers'].quantile(0.9)}")
    
    # -------------------------------------------------------------------------
    # 4. Compute Topic Coverage & Filter Topics Below Coverage Threshold
    # -------------------------------------------------------------------------
    # Apply basic coverage cutoff filter
    valid_topics_by_coverage = topic_speaker_counts.loc[
        topic_speaker_counts["coverage"] >= coverage_cutoff, 
        "subtopic_1"
    ]
    
    # Apply percentile filter if needed
    if topic_percentile > 0:
        speaker_threshold = topic_speaker_counts["num_speakers"].quantile(topic_percentile)
        print(f"Topic percentile threshold {topic_percentile*100:.1f}% corresponds to {speaker_threshold:.0f}+ speakers")
        
        valid_topics_by_percentile = topic_speaker_counts.loc[
            topic_speaker_counts["num_speakers"] >= speaker_threshold, 
            "subtopic_1"
        ]
        
        # Combine both filters (intersection)
        valid_topics = set(valid_topics_by_coverage) & set(valid_topics_by_percentile)
        valid_topics = list(valid_topics)
        
        print(f"Filtered from {len(topic_speaker_counts)} to {len(valid_topics)} topics")
    else:
        valid_topics = valid_topics_by_coverage
        print(f"Retaining {len(valid_topics)} topics with coverage >= {coverage_cutoff*100:.1f}%")
    
    # Filter df_long to include only valid topics
    df_long = df_long[df_long["subtopic_1"].isin(valid_topics)]
    
    # -------------------------------------------------------------------------
    # 5. Setup Model Indices
    # -------------------------------------------------------------------------
    # Create integer indices for speakers and subtopic_1 items
    df_long["speaker_id"] = df_long["speaker"].astype("category").cat.codes
    df_long["item_id"] = df_long["subtopic_1"].astype("category").cat.codes

    # Extract the unique counts after filtering
    n_speakers = df_long["speaker_id"].nunique()
    n_items = df_long["item_id"].nunique()

    print(f"Number of speakers with valid data (at least 1 topic): {n_speakers}")
    print(f"Number of retained topics (items): {n_items}")

    # Build arrays for the model
    speaker_idx = df_long["speaker_id"].values
    item_idx = df_long["item_id"].values
    stance_scores = df_long["stance_score"].values

    # -------------------------------------------------------------------------
    # 6. Build and Fit Continuous 1D IRT Model in PyMC
    # -------------------------------------------------------------------------
    with pm.Model() as irt_model:

        # Latent ideal points for each speaker
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=n_speakers)

        # Discrimination & difficulty for each subtopic_1
        a = pm.Normal("a", mu=0.0, sigma=1.0, shape=n_items)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=n_items)

        # Residual standard deviation (common to all items)
        sigma = pm.Exponential("sigma", lam=1.0)

        # Expected stance: y_{j,i} = a_i * theta_j - b_i
        mu = a[item_idx] * theta[speaker_idx] - b[item_idx]

        # Likelihood
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=stance_scores)

        # Sample with NUTS
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=4,
            random_seed=42,
            nuts_sample="blackjax"
        )

    print(az.summary(trace, var_names=["theta", "a", "b", "sigma"]))

    # -------------------------------------------------------------------------
    # 7. Extract Posterior Means of Ideal Points
    # -------------------------------------------------------------------------
    # If n_speakers == 0 (extreme case after filtering), skip extraction
    if n_speakers > 0:
        # theta shape => (draws * chains, n_speakers)
        theta_means = trace.posterior["theta"].mean(dim=("chain", "draw")).values
        
        # Extract 95% credible intervals for theta
        theta_lower = trace.posterior["theta"].quantile(0.025, dim=("chain", "draw")).values
        theta_upper = trace.posterior["theta"].quantile(0.975, dim=("chain", "draw")).values

        # Map speaker_id -> speaker name
        speaker_map = (
            df_long[["speaker_id", "speaker"]]
            .drop_duplicates()
            .sort_values("speaker_id")
            .set_index("speaker_id")["speaker"]
        )

        results = pd.DataFrame({
            "speaker_id": range(n_speakers),
            "ideal_point_mean": theta_means,
            "ideal_point_lower": theta_lower,
            "ideal_point_upper": theta_upper
        })
        results["speaker"] = results["speaker_id"].map(speaker_map)

        # Sort by ideal point (descending)
        results = results.sort_values("ideal_point_mean", ascending=False).reset_index(drop=True)
        
        # Add chamber column to results if available in df_long
        if "chamber" in df_long.columns:
            speaker_chambers = df_long[['speaker', 'chamber']].drop_duplicates('speaker')
            speaker_chambers = speaker_chambers.set_index('speaker')['chamber']
            results["chamber"] = results["speaker"].map(speaker_chambers)
        
        # Extract and print the topics with highest absolute discrimination parameters
        # Map item_id -> topic name
        topic_map = (
            df_long[["item_id", "subtopic_1"]]
            .drop_duplicates()
            .sort_values("item_id")
            .set_index("item_id")["subtopic_1"]
        )
        
        # Get mean discrimination parameter values
        a_means = trace.posterior["a"].mean(dim=("chain", "draw")).values
        
        # Get mean difficulty parameter values
        b_means = trace.posterior["b"].mean(dim=("chain", "draw")).values
        
        # Extract 95% credible intervals for alpha (discrimination) and beta (difficulty)
        a_lower = trace.posterior["a"].quantile(0.025, dim=("chain", "draw")).values
        a_upper = trace.posterior["a"].quantile(0.975, dim=("chain", "draw")).values
        b_lower = trace.posterior["b"].quantile(0.025, dim=("chain", "draw")).values
        b_upper = trace.posterior["b"].quantile(0.975, dim=("chain", "draw")).values
        
        # Create DataFrame with topic names and parameter values
        topic_params_df = pd.DataFrame({
            "topic": [topic_map[i] for i in range(len(a_means))],
            "alpha_discrimination_mean": a_means,
            "alpha_discrimination_lower": a_lower,
            "alpha_discrimination_upper": a_upper,
            "beta_difficulty_mean": b_means,
            "beta_difficulty_lower": b_lower,
            "beta_difficulty_upper": b_upper,
            "abs_discrimination": np.abs(a_means)
        })
        
        # Create a copy for sorting by discrimination
        discrimination_df = topic_params_df[["topic", "alpha_discrimination_mean", "abs_discrimination"]].copy()
        discrimination_df = discrimination_df.sort_values("abs_discrimination", ascending=False)
        
        # Print the top topics by absolute discrimination parameter
        print("\nTopics with highest absolute discrimination parameter values:")
        print(discrimination_df.head(10).to_string(index=False))
        
        # Save the discrimination parameters to a CSV
        discrimination_csv = output_csv.replace("idealpoints.csv", "discrimination.csv")
        discrimination_df.to_csv(discrimination_csv, index=False)
        print(f"Saved topic discrimination parameters to {discrimination_csv}")
        
        # Save the complete topic parameter table (sorted alphabetically by topic name)
        topic_params_csv = output_csv.replace("idealpoints.csv", "topic_parameters.csv")
        topic_params_sorted = topic_params_df.sort_values("topic")
        topic_params_columns = ["topic", 
                              "alpha_discrimination_mean", "alpha_discrimination_lower", "alpha_discrimination_upper", 
                              "beta_difficulty_mean", "beta_difficulty_lower", "beta_difficulty_upper"]
        topic_params_sorted[topic_params_columns].to_csv(topic_params_csv, index=False)
        print(f"Saved topic IRT parameters to {topic_params_csv} (sorted alphabetically by topic name)")
        
        # ---------------------------------------------------------------------
        # 8. Save Results & Plot
        # ---------------------------------------------------------------------
        results.to_csv(output_csv, index=False)
        print(f"Saved speaker ideal points to {output_csv}")
        
        # Load party data for plot
        party_info = load_party_data(party_csv=party_csv)
        
        # Add party information to results for the uncertainty CSV
        results['party_code'] = 'UNK'
        for i, row in results.iterrows():
            speaker = row['speaker']
            chamber_code = 'UNK'
            if 'chamber' in results.columns:
                chamber = row['chamber'] 
                if chamber.lower().startswith('s'):
                    chamber_code = 'S'
                elif chamber.lower().startswith('h'):
                    chamber_code = 'H'
            
            # Match speaker to party using the matching function
            _, matched_party = match_speaker_to_party(
                speaker, chamber_code, party_info
            )
            
            results.at[i, 'party_code'] = matched_party
        
        # Create and save uncertainty CSV file
        uncertainty_csv = output_csv.replace("idealpoints.csv", "idealpoints_uncertainty.csv")
        uncertainty_df = results[['speaker', 'chamber', 'party_code', 
                                  'ideal_point_lower', 'ideal_point_mean', 'ideal_point_upper']]
        uncertainty_df.columns = ['speaker', 'chamber', 'party', 
                                 '95%_theta_lower', 'theta_mean', '95%_theta_upper']
        # set theta_mean column to be a float
        uncertainty_df['theta_mean'] = pd.to_numeric(uncertainty_df['theta_mean'], errors='coerce')
        # Sort by increasing theta_mean
        uncertainty_df = uncertainty_df.sort_values('theta_mean', ascending=True)
        uncertainty_df.to_csv(uncertainty_csv, index=False)
        print(f"Saved uncertainty estimates to {uncertainty_csv} (sorted by increasing theta_mean)")
        
        # Plot the 1D ideal points
        plot_1d_ideal_points(
            results=results,
            party_info=party_info,
            output_png=plot_png
        )
        
        # Create the uncertainty plot
        uncertainty_plot_png = plot_png.replace(".png", "_uncertainty.png")
        plot_ideal_points_with_uncertainty(
            results=results,
            party_info=party_info,
            output_png=uncertainty_plot_png
        )
    else:
        print("No speakers remained after filtering. Skipping output.")


def load_party_data(party_csv="congress_117_party.csv"):
    """
    Load party information for congress members from a CSV that has columns:
      - 'speaker'
      - 'chamber' (H or S)
      - 'party'   (D, R, or I)
    Returns a DataFrame with party information
    """
    try:
        party_df = pd.read_csv(party_csv)
        return party_df
    except Exception as e:
        print(f"Error loading party data: {e}")
        return pd.DataFrame(columns=["speaker", "chamber", "party"])

def match_speaker_to_party(speaker_name, chamber, party_df):
    """
    Match speaker names from stance predictions to the party dataframe.
    
    Args:
        speaker_name (str): Name in format "Mr. SCHUMER" or "Mr. JOHNSON of OHIO"
        chamber (str): Chamber ('H' or 'S')
        party_df (pd.DataFrame): DataFrame with party information
    
    Returns:
        tuple: (chamber, party) or (chamber, 'UNK') if not found
    """
    # Special case for Mr. RYAN, who is in the House with Democratic party
    if speaker_name == "Mr. RYAN":
        if chamber.upper().startswith('H'):
            return ('H', 'D')
        else:
            return ('S', 'R')
    
    # Handle empty or None values
    if not speaker_name or pd.isna(speaker_name):
        return (chamber[0].upper() if chamber else 'H', 'UNK')
    
    # Split the name first by period (to remove salutations)
    name_parts = speaker_name.split('.', 1)
    if len(name_parts) > 1:
        # Take the part after the period and strip whitespace
        name = name_parts[1].strip()
    else:
        name = name_parts[0].strip()
    
    # Check if there's a state mentioned (format: "JOHNSON of OHIO")
    state_parts = name.split(' of ')
    lastname = state_parts[0].strip()
    state = state_parts[1].strip() if len(state_parts) > 1 else None
    
    # Normalize chamber to single letter
    chamber_code = chamber[0].upper() if chamber else 'H'
    
    # Try to match by lastname and chamber
    match_by_name = party_df[(party_df['speaker'] == lastname) & (party_df['chamber'] == chamber_code)]
    if len(match_by_name) == 1:
        # If we have exactly one match, return it
        row = match_by_name.iloc[0]
        return (row['chamber'], row['party'])
    elif len(match_by_name) > 1 and state is not None:
        # If multiple matches and we have a state, use state to disambiguate
        match_by_state = match_by_name[match_by_name['state'] == state]
        if len(match_by_state) >= 1:
            # If we have at least one match by state, return the first one
            row = match_by_state.iloc[0]
            return (row['chamber'], row['party'])
    
    # If no match is found, print the speaker information
    print(f"No match found for speaker: '{speaker_name}', extracted lastname: '{lastname}', chamber: '{chamber_code}', state: '{state if state else 'N/A'}'")
    
    # Default to unknown party if no match is found
    return (chamber_code, 'UNK')

def plot_1d_ideal_points(results, party_info, output_png):
    """Generate a 1D line plot of ideal points."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    print(f"Generating 1D ideal point plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors for each chamber-party combination
    colors = {
        ('S', 'D'): '#0015BC',  # Senate Democrat - Dark Blue
        ('S', 'R'): '#FF0000',  # Senate Republican - Red
        ('S', 'I'): '#006400',  # Senate Independent - Dark Green
        ('H', 'D'): '#6495ED',  # House Democrat - Light Blue
        ('H', 'R'): '#FF6B6B',  # House Republican - Light Red
        ('H', 'I'): '#90EE90',  # House Independent - Light Green
        ('S', 'UNK'): '#808080',  # Senate Unknown - Gray
        ('H', 'UNK'): '#D3D3D3'  # House Unknown - Light Gray
    }
    
    # Define markers for each chamber
    markers = {'S': '^', 'H': 'o', 'UNK': 's'}
    
    # Get chamber code for each speaker
    results['chamber_code'] = 'UNK'
    for i, row in results.iterrows():
        if 'chamber' in results.columns:
            chamber = row['chamber']
            if chamber.lower().startswith('s'):
                results.at[i, 'chamber_code'] = 'S'
            elif chamber.lower().startswith('h'):
                results.at[i, 'chamber_code'] = 'H'
    
    # Assign parties and chambers to each speaker
    results['party_code'] = 'UNK'
    for i, row in results.iterrows():
        speaker = row['speaker']
        chamber_code = row['chamber_code']
        
        # Match speaker to party using the matching function
        matched_chamber, matched_party = match_speaker_to_party(
            speaker, chamber_code, party_info
        )
        
        results.at[i, 'chamber_code'] = matched_chamber
        results.at[i, 'party_code'] = matched_party
    
    # Get unique chamber-party pairs from results
    unique_cpairs = set(zip(results['chamber_code'], results['party_code']))
    
    # List of legislators to label (exact names as they appear in the data)
    legislators_to_label = ["Mr. SCHUMER", "Ms. PELOSI", "Ms. NORTON", "Mrs. MILLER-MEEKS", "Mr. McCONNELL", "Mr. SCALISE"]
    
    # Plot each group of speakers (by chamber and party)
    for (chamber, party) in unique_cpairs:
        # Select speakers of this party and chamber
        mask = (results['chamber_code'] == chamber) & (results['party_code'] == party)
        
        if any(mask):
            # Get positions and labels
            positions = np.arange(len(results))[mask]
            ideal_points = results.loc[mask, "ideal_point_mean"]
            labels = results.loc[mask, "speaker"]
            
            # Plot points
            color = colors.get((chamber, party), "gray")
            marker = markers.get(chamber, "o")
            ax.scatter(
                positions,
                ideal_points,
                c=color,
                marker=marker,
                alpha=0.6,
                s=100,
                label=f"{chamber} - {party}"
            )
            
            # Add names only for specified legislators
            for pos, point, label in zip(positions, ideal_points, labels):
                if label in legislators_to_label:
                    # Get marker shape based on chamber
                    marker_shape = markers.get(chamber, "o")
                    
                    # Add text label
                    speaker_name = " ".join(label.split(".", 1)[1:]) if "." in label else label
                    ax.annotate(
                        speaker_name,
                        (pos, point),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        rotation=90
                    )
                    
                    # Add bold outline with correct marker shape
                    ax.scatter(
                        pos,
                        point,
                        s=120,
                        facecolors='none',
                        edgecolors='black',
                        linewidths=2,
                        marker=marker_shape
                    )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_ylabel("Ideal Point (First Dimension)")
    ax.set_title("1D Ideal Points by Speaker")
    
    # Hide x-axis ticks and labels
    ax.set_xticks([])
    
    # Add legend with small point sizes
    ax.legend(markerscale=0.7, fontsize=8, loc='upper right')
    
    # Add explanatory text
    plt.figtext(
        0.08, 0.02,
        'Only vertical position matters.',
        fontsize=10,
        alpha=0.7
    )
    
    # Save the plot
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved to {output_png}")


def plot_ideal_points_with_uncertainty(results, party_info, output_png):
    """Generate a plot of ideal points with uncertainty estimates (95% credible intervals)."""
    print(f"Generating ideal points with uncertainty plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors for each chamber-party combination
    colors = {
        ('S', 'D'): '#0015BC',  # Senate Democrat - Dark Blue
        ('S', 'R'): '#FF0000',  # Senate Republican - Red
        ('S', 'I'): '#006400',  # Senate Independent - Dark Green
        ('H', 'D'): '#6495ED',  # House Democrat - Light Blue
        ('H', 'R'): '#FF6B6B',  # House Republican - Light Red
        ('H', 'I'): '#90EE90',  # House Independent - Light Green
        ('S', 'UNK'): '#808080',  # Senate Unknown - Gray
        ('H', 'UNK'): '#D3D3D3'  # House Unknown - Light Gray
    }
    
    # Define markers for each chamber
    markers = {'S': '^', 'H': 'o', 'UNK': 's'}
    
    # Ensure chamber_code and party_code are assigned
    if 'chamber_code' not in results.columns or 'party_code' not in results.columns:
        # Get chamber code for each speaker
        results['chamber_code'] = 'UNK'
        for i, row in results.iterrows():
            if 'chamber' in results.columns:
                chamber = row['chamber']
                if chamber.lower().startswith('s'):
                    results.at[i, 'chamber_code'] = 'S'
                elif chamber.lower().startswith('h'):
                    results.at[i, 'chamber_code'] = 'H'
        
        # Assign parties and chambers to each speaker
        results['party_code'] = 'UNK'
        for i, row in results.iterrows():
            speaker = row['speaker']
            chamber_code = row['chamber_code']
            
            # Match speaker to party using the matching function
            matched_chamber, matched_party = match_speaker_to_party(
                speaker, chamber_code, party_info
            )
            
            results.at[i, 'chamber_code'] = matched_chamber
            results.at[i, 'party_code'] = matched_party
    
    # Sort results by ideal point only (increasing order)
    results = results.sort_values('ideal_point_mean')
    
    # Add positional index
    results['position'] = np.arange(len(results))
    
    # Get unique chamber-party pairs from results
    unique_cpairs = set(zip(results['chamber_code'], results['party_code']))
    
    # List of legislators to label (exact names as they appear in the data)
    legislators_to_label = ["Mr. SCHUMER", "Ms. PELOSI", "Ms. NORTON", "Mrs. MILLER-MEEKS", "Mr. McCONNELL", "Mr. SCALISE"]
    
    # Plot each group of speakers (by chamber and party)
    for (chamber, party) in unique_cpairs:
        # Select speakers of this party and chamber
        mask = (results['chamber_code'] == chamber) & (results['party_code'] == party)
        
        if any(mask):
            # Get positions and values
            positions = results.loc[mask, 'position']
            ideal_points = results.loc[mask, "ideal_point_mean"]
            speakers = results.loc[mask, "speaker"]
            
            # Get uncertainty bounds
            lower_bounds = results.loc[mask, "ideal_point_lower"]
            upper_bounds = results.loc[mask, "ideal_point_upper"]
            
            # Plot points and error bars
            color = colors.get((chamber, party), "gray")
            marker = markers.get(chamber, "o")
            
            # Plot error bars first
            for pos, point, lower, upper in zip(positions, ideal_points, lower_bounds, upper_bounds):
                ax.plot([pos, pos], [lower, upper], color=color, alpha=0.3, linewidth=2)
            
            # Plot main points
            ax.scatter(
                positions,
                ideal_points,
                c=color,
                marker=marker,
                alpha=0.8,
                s=50,
                label=f"{chamber} - {party}"
            )
            
            # Add annotations only for specified legislators
            for pos, point, speaker in zip(positions, ideal_points, speakers):
                if speaker in legislators_to_label:
                    # Get marker shape based on chamber
                    marker_shape = markers.get(chamber, "o")
                    
                    # Add text label
                    speaker_name = " ".join(speaker.split(".", 1)[1:]) if "." in speaker else speaker
                    ax.annotate(
                        speaker_name,
                        (pos, point),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90
                    )
                    
                    # Add bold outline with correct marker shape
                    ax.scatter(
                        pos,
                        point,
                        s=120,
                        facecolors='none',
                        edgecolors='black',
                        linewidths=2,
                        marker=marker_shape
                    )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_ylabel("Ideal Point (First Dimension)")
    ax.set_title("1D Ideal Points with 95% Credible Intervals")
    
    # Hide x-axis ticks and labels
    ax.set_xticks([])
    
    # Add legend with small point sizes
    ax.legend(markerscale=0.7, fontsize=8, loc='upper right')
    
    # Add explanatory text
    plt.figtext(
        0.02, 0.02,
        'Vertical lines show 95% credible intervals.\n' + 
        'Larger intervals indicate greater uncertainty in speaker positions.\n' +
        'Speakers ordered by increasing ideal point value.\n',
        fontsize=10,
        alpha=0.7
    )
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Uncertainty plot saved to {output_png}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    print(jax.default_backend())
    print(jax.devices())
    # Example usage:
    #   coverage_cutoff=0.10 -> remove topics with coverage < 10%
    #   party_filter='D' -> show only Democratic party members
    run_continuous_irt_subtopic1(
        input_csv="speaker_subtopic1_matrix.csv",
        output_csv="speaker_subtopic1_idealpoints.csv",
        party_csv="congress_117_party.csv",
        plot_png="stance_1d_plot.png",
        coverage_cutoff=0,
        speaker_percentile=0.75,
        topic_percentile=0.75,
        party_filter='',  # Set to 'D' or 'R' to filter by party
        draws=4000,
        tune=2000,
        target_accept=0.9
    )
