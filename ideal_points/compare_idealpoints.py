#!/usr/bin/env python3
"""
StrideStance Ideal Points: Comparison with DW-NOMINATE

This script implements comparative analysis between stance-based ideal points from the StrideStance
system and DW-NOMINATE scores from roll-call votes. It provides statistical validation of the
stance-based approach by measuring correlations with the established gold standard in
political science.

Key features:
1. Loading and matching stance-based ideal points with DW-NOMINATE scores
2. Calculating correlation statistics (Pearson and Spearman)
3. Identifying legislators with the greatest discrepancies between measures
4. Visualizing the relationship between speech-based and vote-based ideological positioning
5. Separate analysis by chamber (House vs. Senate)

The script generates both statistical outputs and publication-quality visualizations that
highlight the relationship between how legislators position themselves in their speeches
versus how they vote, revealing strategic positioning and rhetorical moderation.

Outputs:
- Scatter plot with regression line showing the relationship between measures
- Correlation statistics (overall and by chamber)
- CSV file with combined data for further analysis
- List of legislators with greatest discrepancies between measures

This analysis serves as validation for the stance-based ideal points methodology while
also revealing important insights about legislators who strategically position themselves
differently in their speeches compared to their voting records.

Usage:
  python compare_idealpoints.py --stance_csv speaker_subtopic1_idealpoints.csv
                               --nominate_csv external/nominate_117.csv
                               --output_csv idealpoint_comparison_data.csv
                               --output_plot idealpoint_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re
import os

# File paths
IRT_IDEALPOINTS_FILE = "/n/home09/michaelzhao/Downloads/thesis/vast/speaker_subtopic1_idealpoints.csv"
PARTY_INFO_FILE = "/n/home09/michaelzhao/Downloads/thesis/vast/congress_117_party.csv"
DW_NOMINATE_FILE = "/n/home09/michaelzhao/Downloads/thesis/vast/data/HSall_members.csv"
OUTPUT_PLOT_FILE = "/n/home09/michaelzhao/Downloads/thesis/vast/idealpoint_comparison.png"

def match_speaker_to_party(speaker_name, chamber, party_df):
    """
    Match speaker names from stance predictions to the party dataframe.
    This is the same matching logic from irt.py.
    
    Args:
        speaker_name (str): Name in format "Mr. SCHUMER" or "Mr. JOHNSON of OHIO"
        chamber (str): Chamber ('H' or 'S')
        party_df (pd.DataFrame): DataFrame with party information
    
    Returns:
        tuple: (chamber, party, lastname) or (chamber, 'UNK', lastname) if not found
    """
    # Handle empty or None values
    if not speaker_name or pd.isna(speaker_name):
        return (chamber[0].upper() if chamber else 'H', 'UNK', '')
    
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
        return (row['chamber'], row['party'], lastname)
    elif len(match_by_name) > 1 and state is not None:
        # If multiple matches and we have a state, use state to disambiguate
        match_by_state = match_by_name[match_by_name['state'] == state]
        if len(match_by_state) >= 1:
            # If we have at least one match by state, return the first one
            row = match_by_state.iloc[0]
            return (row['chamber'], row['party'], lastname)
    
    # If no match is found, print the speaker information
    print(f"No match found for speaker: '{speaker_name}', extracted lastname: '{lastname}', chamber: '{chamber_code}', state: '{state if state else 'N/A'}'")
    
    # Default to unknown party if no match is found
    return (chamber_code, 'UNK', lastname)

def load_irt_idealpoints():
    """Load IRT-based ideal points."""
    print("Loading IRT-based ideal points...")
    df = pd.read_csv(IRT_IDEALPOINTS_FILE)
    print(f"Loaded {len(df)} speakers with IRT ideal points")
    return df

def load_party_info():
    """Load party information."""
    print("Loading party information...")
    try:
        party_df = pd.read_csv(PARTY_INFO_FILE)
        print(f"Loaded party information for {len(party_df)} speakers")
        return party_df
    except Exception as e:
        print(f"Error loading party data: {e}")
        return pd.DataFrame(columns=["speaker", "chamber", "party"])

def load_dw_nominate():
    """Load DW-NOMINATE scores and filter for 117th Congress."""
    print("Loading DW-NOMINATE scores...")
    df = pd.read_csv(DW_NOMINATE_FILE)
    
    # Filter for 117th Congress
    df_117 = df[df['congress'] == 117]
    
    # Filter out non-legislators (e.g., President)
    df_117 = df_117[df_117['chamber'].isin(['House', 'Senate'])]
    
    # Convert chamber to our format
    df_117['chamber_code'] = df_117['chamber'].apply(lambda x: 'H' if x == 'House' else 'S')
    
    # Format bioname to get lastname
    df_117['lastname'] = df_117['bioname'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else '')
    
    # Map party_code to our format (100=D, 200=R, 328=I)
    party_map = {100: 'D', 200: 'R', 328: 'I'}
    df_117['party'] = df_117['party_code'].apply(lambda x: party_map.get(x, 'UNK'))
    
    print(f"Loaded DW-NOMINATE scores for {len(df_117)} members of the 117th Congress")
    return df_117

def match_legislators(irt_df, party_df, dw_df):
    """Match legislators between the datasets using the irt.py matching logic."""
    # Process each IRT speaker and find matches
    irt_speakers_count = len(irt_df)
    print(f"Starting with {irt_speakers_count} speakers from IRT dataset")
    
    match_results = []
    
    for idx, row in irt_df.iterrows():
        speaker_name = row['speaker']
        chamber = row['chamber'] if 'chamber' in row else None
        
        # Special case handling for specific legislators
        special_case_match = None
        
        if speaker_name == "Mrs. RODGERS of Washington":
            matched_chamber, matched_party, lastname = 'H', "R", 'McMORRIS RODGERS, Cathy'
            special_case_match = True
        elif speaker_name == "Mr. RODNEY DAVIS of Illinois":
            matched_chamber, matched_party, lastname = 'H', "R", 'DAVIS, Rodney'
            special_case_match = True
        elif speaker_name == "Ms. VELAZQUEZ":
            matched_chamber, matched_party, lastname = 'H', "D", 'VELÁZQUEZ, Nydia M.'
            special_case_match = True
        elif speaker_name == "Mr. DANNY K. DAVIS of Illinois":
            matched_chamber, matched_party, lastname = 'H', "D", 'DAVIS, Danny K.'
            special_case_match = True
        elif speaker_name == "Mrs. CAROLYN B. MALONEY of New York":
            matched_chamber, matched_party, lastname = 'H', "D", 'MALONEY, Carolyn Bosher'
            special_case_match = True
        elif speaker_name == "Mr. RYAN":
            matched_chamber, matched_party, lastname = 'H', 'D', 'RYAN, Timothy J.'
            special_case_match = True
        else:
            # Match using the function from irt.py
            matched_chamber, matched_party, lastname = match_speaker_to_party(speaker_name, chamber, party_df)
            special_case_match = False
        
        # Find the corresponding DW-NOMINATE entry by lastname and chamber
        if special_case_match:
            # For special cases, use the bioname field to match
            dw_matches = dw_df[dw_df['bioname'] == lastname]
        else:
            # Standard matching using lastname, chamber, and party
            dw_matches = dw_df[(dw_df['lastname'] == lastname) & 
                              (dw_df['chamber_code'] == matched_chamber) & 
                              (dw_df['party'] == matched_party)]
        
        if len(dw_matches) > 0:
            # Use the first match if multiple
            dw_match = dw_matches.iloc[0]
            
            match_results.append({
                'speaker_id': row['speaker_id'],
                'speaker': speaker_name,
                'lastname': lastname,
                'chamber': matched_chamber,
                'party': matched_party,
                'ideal_point_mean': row['ideal_point_mean'],
                'nominate_dim1': dw_match['nominate_dim1'],
                'nominate_dim2': dw_match['nominate_dim2'],
                'bioname': dw_match['bioname'],
                'bioguide_id': dw_match['bioguide_id']
            })
        else:
            # Print the name of unmatched legislator
            print(f"Unmatched legislator: {speaker_name} ({matched_chamber}, {matched_party}, {lastname})")
    
    # Convert to DataFrame
    matched_df = pd.DataFrame(match_results)
    print(f"Final matched dataset: {len(matched_df)} speakers out of {irt_speakers_count} original IRT speakers ({len(matched_df)/irt_speakers_count*100:.1f}%)")
    
    # Print distribution of matches by party and chamber
    print("\nMatches by party and chamber:")
    party_chamber_counts = matched_df.groupby(['chamber', 'party']).size().reset_index(name='count')
    print(party_chamber_counts)
    
    return matched_df

def analyze_correlation(matched_df):
    """Analyze correlation between IRT ideal points and DW-NOMINATE scores."""
    # Print information about expected correlation direction
    print("\nNote on correlation interpretation:")
    print("  IRT scores: Lower values = more liberal/left-leaning")
    print("  DW-NOMINATE scores: Lower (more negative) values = more liberal/left-leaning")
    print("  Therefore, a positive correlation is expected between these measures")
    
    # Show examples of the most liberal and conservative legislators by each measure
    print("\nExamples for validation:")
    most_liberal_irt = matched_df.loc[matched_df['ideal_point_mean'].idxmin()]
    most_conservative_irt = matched_df.loc[matched_df['ideal_point_mean'].idxmax()]
    most_liberal_dw = matched_df.loc[matched_df['nominate_dim1'].idxmin()]
    most_conservative_dw = matched_df.loc[matched_df['nominate_dim1'].idxmax()]
    
    print(f"  Most liberal by IRT:        {most_liberal_irt['bioname']} (IRT={most_liberal_irt['ideal_point_mean']:.3f}, DW={most_liberal_irt['nominate_dim1']:.3f})")
    print(f"  Most conservative by IRT:   {most_conservative_irt['bioname']} (IRT={most_conservative_irt['ideal_point_mean']:.3f}, DW={most_conservative_irt['nominate_dim1']:.3f})")
    print(f"  Most liberal by DW:         {most_liberal_dw['bioname']} (IRT={most_liberal_dw['ideal_point_mean']:.3f}, DW={most_liberal_dw['nominate_dim1']:.3f})")
    print(f"  Most conservative by DW:    {most_conservative_dw['bioname']} (IRT={most_conservative_dw['ideal_point_mean']:.3f}, DW={most_conservative_dw['nominate_dim1']:.3f})")
    
    # Calculate correlations for all data
    pearson_corr, pearson_p = pearsonr(matched_df['ideal_point_mean'], matched_df['nominate_dim1'])
    spearman_corr, spearman_p = spearmanr(matched_df['ideal_point_mean'], matched_df['nominate_dim1'])
    
    print(f"\nCorrelation Analysis (Overall):")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3g})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3g})")
    
    # Calculate correlations by party
    for party in ['D', 'R']:
        party_df = matched_df[matched_df['party'] == party]
        if len(party_df) > 5:  # Only if we have enough data points
            p_corr, p_p = pearsonr(party_df['ideal_point_mean'], party_df['nominate_dim1'])
            s_corr, s_p = spearmanr(party_df['ideal_point_mean'], party_df['nominate_dim1'])
            print(f"\nParty {party}:")
            print(f"  Pearson correlation: {p_corr:.3f} (p-value: {p_p:.3g})")
            print(f"  Spearman correlation: {s_corr:.3f} (p-value: {s_p:.3g})")
    
    # Calculate correlations by chamber
    for chamber in matched_df['chamber'].unique():
        chamber_df = matched_df[matched_df['chamber'] == chamber]
        if len(chamber_df) > 5:  # Only if we have enough data points
            p_corr, p_p = pearsonr(chamber_df['ideal_point_mean'], chamber_df['nominate_dim1'])
            s_corr, s_p = spearmanr(chamber_df['ideal_point_mean'], chamber_df['nominate_dim1'])
            print(f"\n{chamber} Chamber:")
            print(f"  Pearson correlation: {p_corr:.3f} (p-value: {p_p:.3g})")
            print(f"  Spearman correlation: {s_corr:.3f} (p-value: {s_p:.3g})")

def analyze_ordinal_position_shifts(matched_df):
    """
    Analyze the shift in ordinal position between IRT ideal points and DW-NOMINATE scores.
    Identifies speakers with the greatest change in ranking between the two systems.
    """
    print("\nAnalyzing ordinal position shifts...")
    
    # Define filters similar to those in create_visualizations
    filters = [
        {"name": "Overall", "filter_func": lambda df: df, "title_suffix": "All Legislators"},
        {"name": "Democrats", "filter_func": lambda df: df[df['party'] == 'D'], "title_suffix": "Democrats Only"},
        {"name": "Republicans", "filter_func": lambda df: df[df['party'] == 'R'], "title_suffix": "Republicans Only"},
        {"name": "House", "filter_func": lambda df: df[df['chamber'] == 'H'], "title_suffix": "House Members Only"},
        {"name": "Senate", "filter_func": lambda df: df[df['chamber'] == 'S'], "title_suffix": "Senate Members Only"}
    ]
    
    # First save the full data with rankings to a CSV for further analysis
    calculate_and_save_rank_data(matched_df, "Overall")
    
    # Then analyze each filter
    for filter_info in filters:
        filtered_df = filter_info["filter_func"](matched_df)
        
        # Skip if filtered dataset is too small
        if len(filtered_df) < 10:  # Need reasonable number for meaningful rankings
            print(f"Not enough data points for {filter_info['name']} rank shift analysis. Skipping.")
            continue
        
        print(f"\n=========== RANK SHIFT ANALYSIS: {filter_info['title_suffix']} ===========")
        analyze_single_filter_rank_shifts(filtered_df, filter_info["name"])

def analyze_single_filter_rank_shifts(data_df, filter_name):
    """
    Analyze rank shifts for a specific filtered dataset.
    """
    # Calculate rank for both measures
    # For IRT: lower scores are more liberal (left-leaning)
    # For DW-NOMINATE: more negative scores are more liberal (left-leaning)
    # So we need to rank them in similar directions to align the ideological spectrum
    data_df = data_df.copy()  # Create a copy to avoid modifying the original
    data_df['irt_rank'] = data_df['ideal_point_mean'].rank(ascending=True)  # Lower IRT = more liberal = lower rank number
    data_df['dw_rank'] = data_df['nominate_dim1'].rank(ascending=True)  # Lower DW = more liberal = lower rank number
    
    # Print the ranges of scores and ranks for validation
    print(f"\nValidation of score ranges ({filter_name}):")
    print(f"  IRT scores:  min={data_df['ideal_point_mean'].min():.3f}, max={data_df['ideal_point_mean'].max():.3f}")
    print(f"  DW scores:   min={data_df['nominate_dim1'].min():.3f}, max={data_df['nominate_dim1'].max():.3f}")
    print(f"  IRT ranks:   min={data_df['irt_rank'].min():.0f}, max={data_df['irt_rank'].max():.0f}")
    print(f"  DW ranks:    min={data_df['dw_rank'].min():.0f}, max={data_df['dw_rank'].max():.0f}")
    
    # Examples for validation: show most liberal and most conservative from each measure
    print("\nExamples for validation:")
    most_liberal_irt = data_df.loc[data_df['ideal_point_mean'].idxmin()]
    most_conservative_irt = data_df.loc[data_df['ideal_point_mean'].idxmax()]
    most_liberal_dw = data_df.loc[data_df['nominate_dim1'].idxmin()]
    most_conservative_dw = data_df.loc[data_df['nominate_dim1'].idxmax()]
    
    print(f"  Most liberal by IRT:        {most_liberal_irt['bioname']} (IRT={most_liberal_irt['ideal_point_mean']:.3f}, DW={most_liberal_irt['nominate_dim1']:.3f})")
    print(f"  Most conservative by IRT:   {most_conservative_irt['bioname']} (IRT={most_conservative_irt['ideal_point_mean']:.3f}, DW={most_conservative_irt['nominate_dim1']:.3f})")
    print(f"  Most liberal by DW:         {most_liberal_dw['bioname']} (IRT={most_liberal_dw['ideal_point_mean']:.3f}, DW={most_liberal_dw['nominate_dim1']:.3f})")
    print(f"  Most conservative by DW:    {most_conservative_dw['bioname']} (IRT={most_conservative_dw['ideal_point_mean']:.3f}, DW={most_conservative_dw['nominate_dim1']:.3f})")
    
    # Calculate the absolute rank difference
    data_df['rank_diff'] = (data_df['irt_rank'] - data_df['dw_rank']).abs()
    
    # Calculate the relative rank shift as a percentage of total legislators
    total_legislators = len(data_df)
    data_df['rank_shift_pct'] = 100 * data_df['rank_diff'] / total_legislators
    
    # Display the legislators with the greatest rank shift
    print(f"\nTop 10 speakers with greatest ordinal position shifts ({filter_name}):")
    print("(Lower ranks = more liberal for both measures)")
    top_shifts = data_df.sort_values('rank_diff', ascending=False).head(10)
    
    # Create a nice formatted table
    shift_table = pd.DataFrame({
        'Name': top_shifts['bioname'],
        'Party': top_shifts['party'],
        'Chamber': top_shifts['chamber'],
        'IRT Rank': top_shifts['irt_rank'].astype(int),
        'DW Rank': top_shifts['dw_rank'].astype(int),
        'Rank Diff': top_shifts['rank_diff'].astype(int),
        'Shift %': top_shifts['rank_shift_pct'].round(1),
        'IRT Score': top_shifts['ideal_point_mean'].round(3),
        'DW Score': top_shifts['nominate_dim1'].round(3)
    })
    
    # Clean up the bioname for display
    shift_table['Name'] = shift_table['Name'].apply(lambda x: x.replace(',', ', ').title() if isinstance(x, str) else x)
    
    print(shift_table.to_string(index=False))
    
    # Also calculate and return summary statistics on the rank shifts
    mean_abs_shift = data_df['rank_diff'].mean()
    median_abs_shift = data_df['rank_diff'].median()
    
    print(f"\nSummary of rank shifts ({filter_name}):")
    print(f"  Mean absolute rank shift: {mean_abs_shift:.1f} positions ({(100*mean_abs_shift/total_legislators):.1f}% of legislators)")
    print(f"  Median absolute rank shift: {median_abs_shift:.1f} positions ({(100*median_abs_shift/total_legislators):.1f}% of legislators)")
    
    # If not the overall analysis, save this specific filtered data too
    if filter_name != "Overall":
        calculate_and_save_rank_data(data_df, filter_name)

def calculate_and_save_rank_data(data_df, filter_name):
    """
    Calculate rankings and save to CSV.
    """
    # Calculate ranks if not already done
    if 'irt_rank' not in data_df.columns:
        data_df = data_df.copy()
        data_df['irt_rank'] = data_df['ideal_point_mean'].rank(ascending=True)
        data_df['dw_rank'] = data_df['nominate_dim1'].rank(ascending=True)
        data_df['rank_diff'] = (data_df['irt_rank'] - data_df['dw_rank']).abs()
        total_legislators = len(data_df)
        data_df['rank_shift_pct'] = 100 * data_df['rank_diff'] / total_legislators
    
    # Create output filename based on filter name
    output_path = "/n/home09/michaelzhao/Downloads/thesis/vast"
    if filter_name == "Overall":
        output_file = f"{output_path}/idealpoint_rank_comparison.csv"
    else:
        output_file = f"{output_path}/idealpoint_rank_comparison_{filter_name}.csv"
    
    # Save the full data with rankings to a CSV for further analysis
    data_df.to_csv(output_file, index=False)
    print(f"Full ranking data for {filter_name} saved to {os.path.basename(output_file)}")

def create_visualizations(matched_df):
    """Create visualizations comparing IRT ideal points with DW-NOMINATE scores."""
    # Create five different plots: overall, D, R, H, S
    filters = [
        {"name": "Overall", "filter_func": lambda df: df, "title_suffix": "All Legislators"},
        {"name": "Democrats", "filter_func": lambda df: df[df['party'] == 'D'], "title_suffix": "Democrats Only"},
        {"name": "Republicans", "filter_func": lambda df: df[df['party'] == 'R'], "title_suffix": "Republicans Only"},
        {"name": "House", "filter_func": lambda df: df[df['chamber'] == 'H'], "title_suffix": "House Members Only"},
        {"name": "Senate", "filter_func": lambda df: df[df['chamber'] == 'S'], "title_suffix": "Senate Members Only"}
    ]
    
    for filter_info in filters:
        filtered_df = filter_info["filter_func"](matched_df)
        
        # Skip if filtered dataset is too small
        if len(filtered_df) < 5:
            print(f"Not enough data points for {filter_info['name']} plot. Skipping.")
            continue
            
        create_single_plot(filtered_df, filter_info["name"], filter_info["title_suffix"])

def create_single_plot(data_df, filter_name, title_suffix):
    """Create a single visualization comparing IRT ideal points with DW-NOMINATE scores."""
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {'D': 'blue', 'R': 'red', 'I': 'green', 'UNK': 'gray'}
    markers = {'H': 'o', 'S': '^'}
    
    # Create scatter plot
    for party in data_df['party'].unique():
        for chamber in data_df['chamber'].unique():
            subset = data_df[(data_df['party'] == party) & (data_df['chamber'] == chamber)]
            if len(subset) > 0:
                plt.scatter(
                    subset['ideal_point_mean'],
                    subset['nominate_dim1'],
                    c=colors.get(party, 'gray'),
                    marker=markers.get(chamber, 'x'),
                    alpha=0.7,
                    s=80,
                    label=f"{chamber}-{party}"
                )
    
    # Calculate and plot regression line
    x = data_df['ideal_point_mean']
    y = data_df['nominate_dim1']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='black', linestyle='--', alpha=0.8)
    
    # Add correlation information
    pearson_corr, _ = pearsonr(data_df['nominate_dim1'], data_df['ideal_point_mean'])
    spearman_corr, _ = spearmanr(data_df['nominate_dim1'], data_df['ideal_point_mean'])
    plt.figtext(
        0.05, 0.05,
        f"Pearson r: {pearson_corr:.3f}\nSpearman ρ: {spearman_corr:.3f}\nN = {len(data_df)}",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add labels and title
    plt.xlabel("IRT Ideal Point", fontsize=14)
    plt.ylabel("DW-NOMINATE (1st Dimension)", fontsize=14)
    plt.title(f"Comparison of IRT-based Ideal Points vs. DW-NOMINATE Scores\n for 134 Legislators in the 117th Congress", fontsize=16)
    
    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create output filename based on filter name
    output_path = os.path.dirname(OUTPUT_PLOT_FILE)
    base_name = os.path.basename(OUTPUT_PLOT_FILE).split('.')[0]
    extension = os.path.basename(OUTPUT_PLOT_FILE).split('.')[1]
    
    if filter_name == "Overall":
        output_file = OUTPUT_PLOT_FILE
    else:
        output_file = f"{output_path}/{base_name}_{filter_name}.{extension}"
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    """Main function to run the analysis."""
    # Load data
    irt_df = load_irt_idealpoints()
    party_df = load_party_info()
    dw_df = load_dw_nominate()
    
    # Match legislators
    matched_df = match_legislators(irt_df, party_df, dw_df)
    
    # If we have matches, analyze and visualize
    if len(matched_df) > 0:
        analyze_correlation(matched_df)
        analyze_ordinal_position_shifts(matched_df)
        create_visualizations(matched_df)
        
        # Save matched data for reference
        matched_df.to_csv("/n/home09/michaelzhao/Downloads/thesis/vast/idealpoint_comparison_data.csv", index=False)
        print("Matched data saved to idealpoint_comparison_data.csv")
    else:
        print("No matches found between datasets. Check speaker name formats.")

if __name__ == "__main__":
    main()
