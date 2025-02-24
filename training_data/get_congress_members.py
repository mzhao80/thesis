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

def get_congress_members(congress_number):
    """
    Fetch all members of a specific congress using the Congress API.
    
    Args:
        congress_number (int): The congress number (e.g., 117)
        
    Returns:
        list: List of member dictionaries with their information
    """
    url = f"{BASE_URL}/member/congress/{congress_number}"
    params = {
        "currentMember": False,  # Include all members of the 117th Congress
        "api_key": API_KEY,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("members", [])
    except Exception as e:
        logging.error(f"Error fetching congress members: {e}")
        return []

def format_name(name):
    """Convert name to the format used in our data (Mr./Ms. LastName)"""
    # Remove suffixes and split the name
    name = name.split(',')[0]  # Remove anything after a comma
    parts = name.split()
    last_name = parts[-1]
    first_name = parts[0]
    
    return f"{first_name} {last_name}"

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
            if not terms:
                continue
            
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
            
            # Format the name
            formatted_name = format_name(member.get("name", ""))
            
            records.append({
                "speaker": formatted_name,
                "chamber": chamber_code,
                "party": party_code
            })
            
        except Exception as e:
            logging.error(f"Error processing member {member.get('name', 'Unknown')}: {e}")
            continue
    
    return pd.DataFrame(records)

def main():
    # Fetch members
    logging.info(f"Fetching members of the {CONGRESS_NUMBER}th Congress...")
    members = get_congress_members(CONGRESS_NUMBER)
    
    if not members:
        logging.error("No members found. Exiting.")
        return
    
    logging.info(f"Found {len(members)} members")
    
    # Create DataFrame
    df = create_member_dataframe(members)
    
    # Save to CSV
    output_path = "/n/home09/michaelzhao/Downloads/thesis/vast/congress_117_party.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved member data to {output_path}")
    
    # Print some statistics
    logging.info("\nMember Statistics:")
    logging.info(df.groupby(['chamber', 'party']).size())

if __name__ == "__main__":
    main()
