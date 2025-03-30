#!/usr/bin/env python3
"""
Example Python script demonstrating a TWO-dimensional continuous IRT model
for aggregated speaker–subtopic_1 stance data using PyMC, with an added step
to remove topics that have coverage below a specified threshold (e.g., 10%).

INPUT:
  speaker_subtopic1_matric.csv
      - Wide format CSV with:
          - The first column named "speaker"
          - The second column named "chamber"
          - Subsequent columns named after each subtopic_1
          - Each cell has a numeric stance score for (speaker, subtopic_1).

  congress_117_party.csv
      - A CSV mapping 'speaker' to 'chamber' (H/S) and 'party' (D/R/I).

OUTPUT:
  speaker_subtopic1_idealpoints.csv
      - CSV mapping each speaker to their estimated 2D ideal point (dim1, dim2).

  stance_2d_plot.png
      - A scatter plot showing each speaker's position in 2D, colored by party
        and shaped by chamber.

MODEL (2D):
  y_{j,i} = dot(a_i, theta_j) - b_i + epsilon_{j,i},
      where:
        theta_j            = latent ideal point for speaker j (2D)
        a_i                = discrimination vector for item i (2D)
        b_i                = difficulty (scalar)
        y_{j,i}            = stance score
        epsilon_{j,i}      ~ Normal(0, sigma^2)

Steps:
  1. Load the wide-format CSV and melt to long format.
  2. Calculate coverage for each subtopic_1 (proportion of speakers with non-NA stance).
  3. Remove topics with coverage < 10%.
  4. Build and fit the 2D IRT model using PyMC (theta, a, b, sigma).
  5. Save each speaker's posterior mean 2D ideal point to a CSV.
  6. Generate a 2D scatter plot colored by (party, chamber).
"""

import pandas as pd
import numpy as np
import pymc as pm
import jax
import matplotlib.pyplot as plt
import arviz as az
import multiprocessing

def load_party_data(party_csv="congress_117_party.csv"):
    """
    Load party information for congress members from a CSV that has columns:
      - 'speaker'
      - 'chamber' (H or S)
      - 'party'   (D, R, or I)
    Returns a DataFrame with party information
    """
    party_df = pd.read_csv(party_csv)
    return party_df

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
        return ('H', 'D')
    
    # Handle empty or None values
    if not speaker_name or pd.isna(speaker_name):
        return (chamber, 'UNK')
    
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
    
    # Try to match by lastname and chamber
    match_by_name = party_df[(party_df['speaker'] == lastname) & (party_df['chamber'] == chamber)]
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
    print(f"No match found for speaker: '{speaker_name}', extracted lastname: '{lastname}', chamber: '{chamber}', state: '{state if state else 'N/A'}'")
    
    # Default to unknown party if no match is found
    return (chamber, 'UNK')

def plot_2d_ideal_points(
    speaker_coords: pd.DataFrame,
    party_info: pd.DataFrame,  # Changed from dict to pd.DataFrame
    output_png: str = "stance_2d_plot.png"
):
    """
    Plot a 2D scatter of legislators' ideal points. 
    speaker_coords: DataFrame with columns ['speaker', 'chamber', 'dim1', 'dim2'].
    party_info: DataFrame with party information
    output_png: filename for saving the figure.
    """
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

    # Create markers for each chamber
    markers = {'S': 'o', 'H': 's'}  # circle for Senate, square for House

    # Convert speaker_coords to a numeric array for easy indexing
    # We'll keep track of the speaker names in the same order
    X = speaker_coords[["dim1", "dim2"]].values
    speaker_names = speaker_coords["speaker"].values

    # Build a mask for each (chamber, party) combination
    # We'll do a loop and scatter them with the appropriate color + marker
    plt.figure(figsize=(18, 12))

    # Get unique chamber-party pairs and add unknown combinations
    unique_cpairs = set()
    for c in ['H', 'S']:
        for p in ['D', 'R', 'I', 'UNK']:
            unique_cpairs.add((c, p))

    for (chamber, party) in unique_cpairs:
        color = colors.get((chamber, party), "gray")
        marker = markers.get(chamber, "x")

        mask = []
        for i, speaker in enumerate(speaker_names):
            # Get chamber directly from the dataframe
            sp_chamber = speaker_coords.iloc[i]['chamber']
            
            # Match speaker to party using the matching function
            matched_chamber, matched_party = match_speaker_to_party(speaker, sp_chamber, party_info)
            
            if matched_chamber == chamber and matched_party == party:
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask, dtype=bool)

        if np.any(mask):
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=color,
                marker=marker,
                alpha=0.6,
                s=100,
                label=f"{chamber} - {party}"
            )

    # Add text labels for each point
    for i, speaker in enumerate(speaker_names):
        speaker_name = " ".join(speaker.split(".", 1)[1:])
        plt.annotate(
            speaker_name,
            (X[i, 0], X[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("2D Projection of Congressional Stance Patterns (2D IRT)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.figtext(
        0.02, 0.02,
        'Points represent speakers positioned by their stance patterns across topics.\n'
        'Proximity indicates similarity in stance.\n'
        'Circles = Senate, Squares = House\n'
        'Colors: Dark Blue = Senate (D), Red = Senate (R), Dark Green = Senate (I)\n'
        '        Light Blue = House (D), Light Red = House (R), Light Green = House (I)',
        fontsize=10,
        alpha=0.7
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved 2D stance plot to {output_png}")
    plt.close()


def run_continuous_irt_subtopic1(
    input_csv="speaker_subtopic1_matrix.csv",
    output_csv="ideal_points_2d.csv",
    party_csv="congress_117_party.csv",
    plot_png="stance_2d_plot.png",
    coverage_cutoff=0.1,
    speaker_percentile=0.75,
    topic_percentile=0.0,
    draws=500,
    tune=2000,
    target_accept=0.9
):
    """
    Run a 2D ideal point estimation using a continuous IRT model.
    
    Parameters
    ----------
    input_csv : str
        Path to the input CSV file (wide-format). Must have:
          - Column "speaker"
          - Column "chamber"
          - One column per subtopic_1
          - Numeric stance scores in each cell
        
    output_csv : str
        Path to the output CSV file containing:
          - speaker_id
          - speaker
          - dim1, dim2: coordinates in 2D space
        
    party_csv : str
        CSV containing speaker -> (chamber, party) for coloring the plot.
    
    plot_png : str
        Filename for saving the 2D scatter plot.
    
    coverage_cutoff : float
        Minimum proportion of speakers that must have a non-missing stance for
        a topic to be retained. For example, 0.10 => 10%.
        
    speaker_percentile : float
        Filter to include only speakers above this percentile based on the number
        of subtopics they spoke about (0.0 = include all, 0.9 = top 10% of speakers).
        
    topic_percentile : float
        Filter to include only topics above this percentile based on the number
        of speakers who spoke about them (0.0 = include all, 0.9 = top 10% of topics).

    draws : int
        Number of posterior samples (after burn-in).
    
    tune : int
        Number of tuning (burn-in) iterations for the NUTS sampler.
    
    target_accept : float
        Target acceptance probability for NUTS (helps with convergence).
    """

    # -------------------------------------------------------------------------
    # 1. Read & Preprocess Data (Wide -> Long)
    # -------------------------------------------------------------------------
    df_wide = pd.read_csv(input_csv)

    # Identify subtopic columns (exclude metadata columns like 'speaker' and 'chamber')
    metadata_cols = ["speaker", "chamber"]
    subtopic_cols = [c for c in df_wide.columns if c not in metadata_cols]

    # Make sure we have chamber information, otherwise throw error
    if "chamber" not in df_wide.columns:
        raise ValueError("Chamber information is missing from the input CSV. Please update ideal_points.py.")

    # Reshape to long format:
    #   speaker, chamber, subtopic_1, stance_score
    df_long = df_wide.melt(
        id_vars=["speaker", "chamber"],
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
    # 2. Filter Speakers and Topics based on Percentile Thresholds
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
    # 3. Compute Topic Coverage & Filter Topics Below Coverage Threshold
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

    print(f"Total subtopic_1 columns originally: {len(subtopic_cols)}")
    print(f"Final dataset has {len(df_long['speaker'].unique())} speakers and {len(valid_topics)} topics")

    # -------------------------------------------------------------------------
    # 4. Setup Model Indices
    # -------------------------------------------------------------------------
    # Create integer indices for speakers and subtopic_1 items
    df_long["speaker_id"] = df_long["speaker"].astype("category").cat.codes
    df_long["item_id"] = df_long["subtopic_1"].astype("category").cat.codes

    n_speakers = df_long["speaker_id"].nunique()
    n_items = df_long["item_id"].nunique()

    print(f"Number of speakers with valid data (at least 1 topic): {n_speakers}")
    print(f"Number of retained topics (items): {n_items}")

    # Build arrays for the model
    speaker_idx = df_long["speaker_id"].values
    item_idx = df_long["item_id"].values
    stance_scores = df_long["stance_score"].values

    if n_speakers == 0 or n_items == 0:
        print("No speakers or no items remain after filtering. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 5. Build and Fit Continuous 2D IRT Model in PyMC
    # -------------------------------------------------------------------------
    with pm.Model() as irt_model:

        # Latent ideal points for each speaker (2D)
        # shape = (n_speakers, 2)
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=(n_speakers, 2))

        # Discrimination for each subtopic_1 (2D)
        # shape = (n_items, 2)
        a = pm.Normal("a", mu=0.0, sigma=1.0, shape=(n_items, 2))

        # Difficulty for each subtopic_1 (scalar)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=n_items)

        # Residual standard deviation (common to all items)
        sigma = pm.Exponential("sigma", lam=1.0)

        # The expected stance is the dot product of a_i and theta_j minus b_i
        # dot(a[i], theta[j]) = a[i,0]*theta[j,0] + a[i,1]*theta[j,1]
        mu = (
            a[item_idx, 0] * theta[speaker_idx, 0] +
            a[item_idx, 1] * theta[speaker_idx, 1] -
            b[item_idx]
        )

        # Likelihood
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=stance_scores)

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
    # 6. Extract Posterior Means of Ideal Points (2D)
    # -------------------------------------------------------------------------
    # theta shape => (draws * chains, n_speakers, 2)
    # We'll take the mean across chain+draw dimension => shape = (n_speakers, 2)
    theta_means = trace.posterior["theta"].mean(dim=("chain", "draw")).values

    # Map speaker_id -> speaker name
    speaker_map = (
        df_long[["speaker_id", "speaker"]]
        .drop_duplicates()
        .sort_values("speaker_id")
        .set_index("speaker_id")["speaker"]
    )

    # Add chamber column to results from df_long
    speaker_chambers = df_long[['speaker', 'chamber']].drop_duplicates('speaker')
    speaker_chambers = speaker_chambers.set_index('speaker')['chamber']
    
    # Create results DataFrame
    results = pd.DataFrame({
        "speaker_id": range(n_speakers),
        "speaker": speaker_map.values,
        "chamber": [speaker_chambers.get(spk, "UNK") for spk in speaker_map.values],
        "dim1": theta_means[:, 0],
        "dim2": theta_means[:, 1]
    })

    # -------------------------------------------------------------------------
    # 7. Save Results & Plot
    # -------------------------------------------------------------------------
    # Sort by first dimension for convenience (descending)
    results = results.sort_values("dim1", ascending=False).reset_index(drop=True)

    results.to_csv(output_csv, index=False)
    print(f"Saved speaker 2D ideal points to {output_csv}")

    # Load party data for plot
    party_info = load_party_data(party_csv=party_csv)

    # -------------------------------------------------------------------------
    # 8. Generate 2D scatter plot
    # -------------------------------------------------------------------------
    results_sorted = results.sort_values("speaker_id").reset_index(drop=True)

    plot_2d_ideal_points(
        speaker_coords=results_sorted[["speaker", "chamber", "dim1", "dim2"]],
        party_info=party_info,
        output_png=plot_png
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    print(jax.default_backend())
    print(jax.devices())
    # Example usage:
    run_continuous_irt_subtopic1(
        input_csv="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/speaker_subtopic1_matrix.csv",
        output_csv="ideal_points_2d.csv",
        party_csv="congress_117_party.csv",
        plot_png="stance_2d_plot.png",
        coverage_cutoff=0,
        speaker_percentile=0.5,
        topic_percentile=0.5,
        draws=500,
        tune=2000,
        target_accept=0.9
    )
