#!/usr/bin/env python
"""
Congress Member Data Collection Module

This script fetches information about members of Congress using the Congress.gov API,
extracts relevant information such as party affiliation and chamber, and saves it to a CSV file.

Usage:
    python get_congress_members.py

Output:
    Creates a CSV file containing the following columns:
    - speaker: Member name in the format used in congressional record (LASTNAME)
    - chamber: Chamber code (S for Senate, H for House)
    - party: Party code (D for Democratic, R for Republican, I for Independent)
    - state: Member's state
"""

import requests
import pandas as pd
import api_keys
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('get_congress_members.log'),
        logging.StreamHandler()
    ]
)

API_KEY = api_keys.CONGRESS_API_KEY
BASE_URL = "https://api.congress.gov/v3"
CONGRESS_NUMBER = 117
OUTPUT_PATH = "congress_members.csv"  # Default output path, can be modified in main()

def get_congress_members(congress_number):
    """
    Fetch all members of a specific congress using the Congress API.
    
    Args:
        congress_number (int): The congress number (e.g., 117)
        
    Returns:
        list: List of member dictionaries with their information
    """
    ret = []
    url = f"{BASE_URL}/member/congress/{congress_number}"
    
    # Congress API has pagination, so we need to make multiple requests
    for offset in [0, 250, 500]:
        params = {
            "currentMember": False,  # Include all members of the specified Congress
            "api_key": API_KEY,
            "offset": offset,
            "limit": 250,
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            ret.extend(data.get("members", []))
        except Exception as e:
            logging.error(f"Error fetching congress members: {e}")
            return []
    
    return ret

def format_name(name):
    """
    Convert name to the format used in our data (LASTNAME).
    
    Args:
        name (str): Original name from the API
        
    Returns:
        str: Formatted name in uppercase with suffixes removed
    """
    # Remove suffixes and split the name
    name = name.split(',')[0].upper()  # Remove anything after a comma, make uppercase
    return name

def create_member_dataframe(members):
    """
    Create a DataFrame with member information in the required format.
    
    Args:
        members (list): List of member dictionaries from the API
        
    Returns:
        pd.DataFrame: DataFrame with speaker, chamber, and party information
    """
    records = []
    
    for member in tqdm(members, desc="Processing members"):
        try:
            # Get the most recent term
            terms = member.get("terms", {}).get("item", [])
            latest_term = terms[-1]
            chamber = latest_term.get("chamber", "")
            
            # Convert chamber format
            if chamber == "Senate":
                chamber_code = "S"
            elif chamber == "House of Representatives":
                chamber_code = "H"
            else:
                continue
            
            # Convert party format
            party_name = member.get("partyName", "")
            if party_name == "Democratic":
                party_code = "D"
            elif party_name == "Republican":
                party_code = "R"
            elif party_name == "Independent":
                party_code = "I"
            else:
                continue

            # Get state
            state = member.get("state", "")
            
            # Format the name
            formatted_name = format_name(member.get("name", ""))
            
            records.append({
                "speaker": formatted_name,
                "chamber": chamber_code,
                "party": party_code,
                "state": state
            })
            
        except Exception as e:
            logging.error(f"Error processing member {member.get('name', 'Unknown')}: {e}")
            continue
    
    return pd.DataFrame(records)

def main():
    """
    Main function to execute the data collection process.
    """
    # Fetch members
    logging.info(f"Fetching members of the {CONGRESS_NUMBER}th Congress...")
    members = get_congress_members(CONGRESS_NUMBER)
    
    if not members:
        logging.error("No members found. Exiting.")
        return
    
    logging.info(f"Found {len(members)} members")
    
    # Create DataFrame
    df = create_member_dataframe(members).sort_values(by=['speaker'])

    # Check for duplicates in speaker names
    duplicate_check = df.duplicated(subset=['speaker'], keep=False)
    if duplicate_check.any():
        logging.warning("Found duplicate speaker entries:")
        for speaker in df[duplicate_check]['speaker'].unique():
            logging.warning(f"\n{df[df['speaker'] == speaker]}")
    
    # Save to CSV
    output_path = OUTPUT_PATH
    df.to_csv(output_path, index=False)
    logging.info(f"Saved member data to {output_path}")
    
    # Print some statistics
    logging.info("\nMember Statistics:")
    logging.info(df.groupby(['chamber', 'party']).size())

if __name__ == "__main__":
    main()
