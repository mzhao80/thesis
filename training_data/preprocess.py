#!/usr/bin/env python
"""
Congressional Record and Bill Data Preprocessing Module

This script performs the following key tasks:
1. Fetches Congressional Record speech data from local files
2. Processes and filters speech text to identify those related to bills
3. Retrieves detailed bill information from the Congress.gov API
4. Matches speeches to their corresponding bills
5. Enriches the dataset with bill metadata (policy areas, summaries, committees)
6. Outputs a processed dataset for further analysis

Usage:
    python preprocess.py [--data-dir DATA_DIR] [--output-file OUTPUT_FILE]

For this to work, you'll need:
1. Valid Congress.gov API keys in api_keys.py
2. Input Congressional Record files in CSV format
"""

import pandas as pd
import requests
from datetime import datetime
from dateutil import parser
from tqdm.auto import tqdm
import json
import re
import api_keys
import os
import time
import logging
import argparse

# Configuration
API_KEY = api_keys.CONGRESS_API_KEY
ALT_API_KEY = api_keys.ALT_CONGRESS_API_KEY  # Backup API key
BASE_URL = "https://api.congress.gov/v3"
CONGRESS_NUMBER = 117

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Clean and standardize speech text from the Congressional Record.
    
    This function performs several text normalization steps:
    1. Standardizes whitespace
    2. Normalizes forms of address (e.g., 'Madam Speaker' -> 'Mr. Speaker')
    3. Removes common procedural text and formalities
    
    Args:
        text (str): Raw speech text from the Congressional Record
        
    Returns:
        str: Preprocessed text with normalized formatting
    """
    # Replace all consecutive whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Replace various forms of address with standardized "Mr. Speaker"
    address_patterns = [
        (r'Mr\. President', 'Mr. Speaker'),
        (r'Mr\. Clerk', 'Mr. Speaker'),
        (r'Mr\. Chair', 'Mr. Speaker'),
        (r'Mr\. Chairman', 'Mr. Speaker'),
        (r'Mr\. Speakerman', 'Mr. Speaker'),
        (r'Madam President', 'Mr. Speaker'),
        (r'Madam Speaker', 'Mr. Speaker'),
        (r'Madam Clerk', 'Mr. Speaker'),
        (r'Madam Chair', 'Mr. Speaker'),
        (r'Madam Chairman', 'Mr. Speaker'),
        (r'Madam Chairwoman', 'Mr. Speaker')
    ]
    
    for pattern, replacement in address_patterns:
        text = re.sub(pattern, replacement, text)

    # Remove common speech opening formalities
    text = re.sub(r'^Mr\. Speaker, ', '', text)
    text = re.sub(r'^I yield.*?\. *', '', text)
    text = re.sub(r'^Mr\. Speaker, ', '', text)  # Check again for sequential patterns

    return text

def standardize_bill_title(title):
    """
    Standardize a bill title for consistent matching.
    
    This function normalizes bill titles by:
    1. Converting to lowercase
    2. Standardizing whitespace
    3. Removing procedural prefixes and suffixes
    4. Removing special characters
    
    Args:
        title (str): Original bill title
        
    Returns:
        str: Standardized bill title, or None if input is None/empty
    """
    def strip_special_chars(s):
        """Remove non-alphanumeric characters except spaces."""
        return re.sub(r'[^A-Za-z0-9\s]+', '', s)
    
    if not title:
        return None
        
    # Convert to lowercase
    title = title.lower()

    # Combine multiple spaces into one
    title = ' '.join(title.split())

    # Remove various prefixes up to the next comma
    prefixes_with_comma = [
        "providing for consideration of",
        "providing for further consideration of",
        "report on",
        "motion to",
        "concerns about",
        "conference report on",
        "request to consider"
    ]

    for prefix in prefixes_with_comma:
        if title.startswith(prefix):
            comma_pos = title.find(", ")
            if comma_pos != -1:
                title = title[comma_pos + 2:]  # +2 to skip the comma and space
                break
    
    # Skip titles that begin with "authorizing the clerk"
    if title.startswith("authorizing the clerk"):
        return None
    
    # Remove simple prefixes
    simple_prefixes = [
        "supporting the ",
        "introduction of the ",
        "reintroduction of the ",
        "re-introduction of the ",
        "introduction of ",
        "reintroduction of ",
        "cosponsorship of ",
    ]
    
    for prefix in simple_prefixes:
        if title.startswith(prefix):
            title = title[len(prefix):]
            
    # Remove everything after double dash
    if "--" in title:
        title = title.split("--")[0].strip()

    # Remove trailing punctuation
    if title.endswith(".") or title.endswith(",") or title.endswith(";"):
        title = title[:-1]

    # Remove common trailing phrases
    trailing_phrases = [
        " (executive session)",
        " (executive calendar)",
        "; and for other purposes",
        ", and for other purposes"
    ]
    
    for phrase in trailing_phrases:
        if title.endswith(phrase):
            title = title[:-len(phrase)]

    # Remove year patterns
    title = re.sub(r', \d{4}$', '', title)
    title = re.sub(r'act of \d{4}$', 'act', title)
    title = re.sub(r'act of$', 'act', title)  # Remove "of" from end of act

    # Remove special characters
    title = strip_special_chars(title)
    
    return title if title else None

def get_bill_data(title, all_bills_dict):
    """
    Get the bill data for a given title.
    
    Retrieves metadata for a bill including policy area, legislative subjects,
    summary, and committee information from the previously built dictionary.
    
    Args:
        title (str): Title of the bill
        all_bills_dict (dict): Dictionary containing bill data
        
    Returns:
        dict: Dictionary containing relevant bill data, or None if not found
    """
    # Standardize the title
    title_std = standardize_bill_title(title)
    if not title_std:
        return None
    
    # Try to find the bill in the dictionary
    if title_std not in all_bills_dict:
        return None
    
    bill_data = all_bills_dict[title_std]
    
    result = {
        'bill_type': bill_data.get('bill_type', ""),
        'bill_number': bill_data.get('bill_number', ""),
        'matched_title': bill_data.get('title', ""),
        'update_date': bill_data.get('updateDate', ""),
        'latest_summary': bill_data.get('latest_summary', ""),
        'latest_summary_date': bill_data.get('latest_summary_date', ""),
        'policy_area': None,
        'policy_area_update': None,
        'legislative_subjects': [],
        'committees': [],
        'url': bill_data.get('url', "")
    }
    
    # Get policy area
    policy_area_data = bill_data.get('subjects_policy', {}).get('policy_area', {})
    if policy_area_data:
        result['policy_area'] = policy_area_data.get('name') or None
        if policy_area_data.get('updateDate'):
            result['policy_area_update'] = parser.parse(policy_area_data['updateDate']).date().isoformat()
    
    # Get legislative subjects
    leg_subjects = bill_data.get('subjects_policy', {}).get('legislative_subjects', [])
    for subject in leg_subjects:
        subject_data = {
            'name': subject.get('name') or None,
            'updateDate': parser.parse(subject['updateDate']).date().isoformat() if subject.get('updateDate') else None
        }
        result['legislative_subjects'].append(subject_data)

    # Get committees with their subcommittees
    committees = bill_data.get('committees', [])
    
    for committee in committees:
        committee_data = {
            'name': committee.get('name') or None,
            'chamber': committee.get('chamber') or None,
            'type': committee.get('type') or None,
            'latest_activity': None,
            'subcommittees': []
        }
        
        # Get latest committee activity
        activities = committee.get('activities', [])
        if activities:
            latest_activity = max(activities, key=lambda x: parser.parse(x['date']).date() if x.get('date') else parser.parse('1900-01-01').date())
            if latest_activity.get('date'):
                committee_data['latest_activity'] = parser.parse(latest_activity['date']).date().isoformat()
        
        # Get subcommittees
        for subcommittee in committee.get('subcommittees', []):
            sub_activities = subcommittee.get('activities', [])
            if sub_activities:
                latest_sub_activity = max(sub_activities, key=lambda x: parser.parse(x['date']).date() if x.get('date') else parser.parse('1900-01-01').date())
                if latest_sub_activity.get('date'):
                    subcommittee_data = {
                        'name': subcommittee.get('name') or None,
                        'latest_activity': parser.parse(latest_sub_activity['date']).date().isoformat()
                    }
                    committee_data['subcommittees'].append(subcommittee_data)
        
        result['committees'].append(committee_data)
    
    return result

def get_bill_subjects_and_policy_area(api_key, congress, bill_type, bill_number):
    """
    Fetch policy area and legislative subjects for a bill from the Congress API.
    
    Args:
        api_key (str): Congress API key
        congress (int): Congress number (e.g., 117)
        bill_type (str): Type of bill (e.g., 'hr', 's')
        bill_number (str): Bill number
        
    Returns:
        dict: Dictionary containing policy area and legislative subjects, or empty dict if request fails
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/subjects"
    params = {"api_key": api_key, "format": "json"}
    
    resp = requests.get(url, params=params)
    # Try with alternate API key if first request fails
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.info(f"Failed to get bill subjects for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return {}
    
    data = resp.json()
    subjects_data = data.get("subjects", {})
    
    # Extract policy area
    policy_area = subjects_data.get("policyArea", {})
    
    # Extract legislative subjects
    legislative_subjects = subjects_data.get("legislativeSubjects", [])
    
    return {
        "policy_area": {
            "name": policy_area.get("name", ""),
            "updateDate": policy_area.get("updateDate")
        },
        "legislative_subjects": [
            {
                "name": s.get("name", ""),
                "updateDate": s.get("updateDate")
            }
            for s in legislative_subjects
        ]
    }

def get_bill_summaries_all(api_key, congress, bill_type, bill_number):
    """
    Fetch all summaries for a bill from the Congress API.
    
    Args:
        api_key (str): Congress API key
        congress (int): Congress number (e.g., 117)
        bill_type (str): Type of bill (e.g., 'hr', 's')
        bill_number (str): Bill number
        
    Returns:
        list: List of summary dictionaries, or empty list if request fails
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/summaries"
    params = {"api_key": api_key, "format": "json"}
    
    resp = requests.get(url, params=params)
    # Try with alternate API key if first request fails
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.info(f"Failed to get bill summaries for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    
    data = resp.json()
    return data.get("summaries", [])

def get_bill_committees_all(api_key, congress, bill_type, bill_number):
    """
    Fetch all committees for a bill from the Congress API.
    
    Args:
        api_key (str): Congress API key
        congress (int): Congress number (e.g., 117)
        bill_type (str): Type of bill (e.g., 'hr', 's')
        bill_number (str): Bill number
        
    Returns:
        list: List of committee dictionaries, or empty list if request fails
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/committees"
    params = {"api_key": api_key, "format": "json"}
    
    resp = requests.get(url, params=params)
    # Try with alternate API key if first request fails
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.info(f"Failed to get bill committees for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    
    data = resp.json()
    return data.get("committees", [])

def fetch_and_cache_bill_data(api_key, congress, bill_type, bill_number):
    """
    Fetch and combine all relevant data for a single bill.
    
    This is a convenience function that fetches subjects, policy area, summaries,
    and committees for a bill and returns them in a single dictionary.
    
    Args:
        api_key (str): Congress API key
        congress (int): Congress number (e.g., 117)
        bill_type (str): Type of bill (e.g., 'hr', 's')
        bill_number (str): Bill number
        
    Returns:
        dict: Combined bill data
    """
    result = {}
    # Fetch subjects & policy area
    result["subjects_policy"] = get_bill_subjects_and_policy_area(api_key, congress, bill_type, bill_number)
    # Fetch committees
    result["committees"] = get_bill_committees_all(api_key, congress, bill_type, bill_number)
    
    return result

def build_bill_data_dict(api_key, all_bills, congress=117, limit_per_page=250, offset=0):
    """
    Build a dictionary of bill data from the Congress API.
    
    This function fetches bills in pages and adds them to the provided dictionary.
    For each bill, it also fetches the latest summary and other metadata.
    
    Args:
        api_key (str): Congress API key
        all_bills (dict): Dictionary to store bill data (modified in-place)
        congress (int): Congress number (e.g., 117)
        limit_per_page (int): Number of bills to fetch per page
        offset (int): Starting offset for pagination
        
    Returns:
        None: The input dictionary is modified in-place
    """
    start_time = time.time()
    while True:
        logger.info(f"Fetching bills at offset {offset}, {offset/19315 * 100:.2f}% of the way done!")
        logger.info(f"Time elapsed in minutes: {(time.time() - start_time) / 60:.2f}")
        
        params = {
            "api_key": api_key,
            "format": "json",
            "offset": offset,
            "limit": limit_per_page
        }
        url = f"{BASE_URL}/bill/{congress}"
        
        resp = requests.get(url, params=params)
        # Try with alternate API key if first request fails
        if resp.status_code != 200:
            params["api_key"] = ALT_API_KEY
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                logger.info(f"Failed to get bills at offset {offset}: {resp.status_code}")
                break
        
        data = resp.json()
        bills_page = data.get("bills", [])
        
        if not bills_page:
            break
        
        # Process each bill in this page
        for b in bills_page:
            offset += 1
            b_title_lower = standardize_bill_title(b.get("title", ""))
            bill_type = b.get("type", "")
            bill_number = b.get("number", "")

            # Get summaries to find the latest one
            summaries = get_bill_summaries_all(api_key, congress, bill_type, bill_number)
            latest_summary = ""
            latest_summary_date = ""
            updateDate = b.get("updateDateIncludingText", "")
            
            if summaries:
                # Get the most recent summary
                latest_summary_item = max(summaries, key=lambda x: parser.parse(x['actionDate']).date() if x.get('actionDate') else parser.parse('1900-01-01').date())
                latest_summary = latest_summary_item.get('text', "")
                if latest_summary_item.get('actionDate', ""):
                    latest_summary_date = parser.parse(latest_summary_item['actionDate']).date().isoformat()

            bill_info = {
                "title": b.get("title", "").strip("."),
                "bill_type": b.get("type", ""),
                "bill_number": b.get("number", ""),
                "url": b.get("url", ""),
                "updateDate": updateDate,
                "latest_summary": latest_summary,
                "latest_summary_date": latest_summary_date
            }

            # Get all alias titles for this bill
            all_titles = get_alias_titles(api_key, congress, bill_type, bill_number)
            all_titles = set(all_titles + [b_title_lower])

            # Add bill to dictionary under each of its titles
            for title in all_titles:
                # Determine if we should add/update this bill in our dictionary
                should_update = title not in all_bills
                
                if not should_update:
                    # We have seen this bill before, decide if we should update it
                    if latest_summary_date:
                        if all_bills[title]['latest_summary_date']:
                            # Compare dates if both have summary dates
                            should_update = (datetime.strptime(all_bills[title]['latest_summary_date'], "%Y-%m-%d").date() < 
                                          datetime.strptime(latest_summary_date, "%Y-%m-%d").date())
                        else:
                            # Prefer entries with summary dates
                            should_update = True
                    else:
                        # Current bill has no summary date
                        if all_bills[title]['latest_summary_date']:
                            # Keep existing entry if it has a summary date
                            should_update = False
                        else:
                            # Compare update dates if neither has summary date
                            should_update = (datetime.strptime(all_bills[title]['updateDate'], "%Y-%m-%d").date() < 
                                          datetime.strptime(updateDate, "%Y-%m-%d").date())

                if should_update:
                    all_bills[title] = bill_info

def get_alias_titles(api_key, congress, bill_type, bill_number):
    """
    Get alternative titles for a bill from the Congress API.
    
    Args:
        api_key (str): Congress API key
        congress (int): Congress number (e.g., 117)
        bill_type (str): Type of bill (e.g., 'hr', 's')
        bill_number (str): Bill number
        
    Returns:
        list: List of standardized alias titles
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/titles"
    params = {"api_key": api_key, "format": "json"}
    
    resp = requests.get(url, params=params)
    # Try with alternate API key if first request fails
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.info(f"Failed to get bill titles for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    
    data = resp.json().get("titles", [])
    ret = []
    
    for title in data:
        alias = standardize_bill_title(title.get("title", ""))
        if alias:
            ret.append(alias)
    
    return ret

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Preprocess Congressional Record data and match with bill information")
    parser.add_argument('--data-dir', type=str, default="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao",
                        help="Directory containing input files and where to save output files")
    parser.add_argument('--output-file', type=str, default="df_bills.csv",
                        help="Name of output CSV file")
    
    return parser.parse_args()

def main():
    """
    Main function to execute the preprocessing pipeline.
    """
    args = parse_arguments()
    data_dir = args.data_dir
    output_file = args.output_file
    
    logger.info('Starting preprocessing pipeline')
    
    # -----------------------------------
    # 1. Read and preprocess the Congressional Record data
    # -----------------------------------
    crec_path_2021 = os.path.join(data_dir, "crec", "crec2021.csv")
    crec_path_2022 = os.path.join(data_dir, "crec", "crec2022.csv")
    
    logger.info(f'Loading Congressional Record data from {crec_path_2021} and {crec_path_2022}')
    df_bills_2021 = pd.read_csv(crec_path_2021)
    df_bills_2022 = pd.read_csv(crec_path_2022)
    df_bills = pd.concat([df_bills_2021, df_bills_2022], ignore_index=True)

    # Remove procedural speakers and filter for substantial speeches
    logger.info('Filtering out procedural speeches and short contributions')
    procedural_speakers = ["the clerk", "the speaker", "the president", "the presiding officer", 
                          "the acting chair", "the chair", "the acting president"]
    df_bills = df_bills[~df_bills["speaker"].str.lower().str.startswith(tuple(procedural_speakers), na=False)]
    df_bills = df_bills[df_bills['speech'].str.len() >= 250]
    
    # -----------------------------------
    # 2. Identify potential bills
    # -----------------------------------
    logger.info('Identifying speeches related to bills')
    df_bills = df_bills.dropna(subset=["doc_title"])
    
    # Filter for documents likely to be about bills
    bill_indicators = [' ACT', ' RESOLUTION']
    df_bills = df_bills[df_bills['doc_title'].str.contains('|'.join(bill_indicators), na=False, case=False)]
    df_bills = df_bills[~df_bills["doc_title"].str.contains("acting chair", na=False, case=False)]
    
    # Preprocess speech text
    logger.info('Preprocessing speech text')
    df_bills['speech'] = df_bills.apply(lambda x: preprocess_text(x['speech']), axis=1)
    
    # -----------------------------------
    # 3. Fetch bill data from Congress API
    # -----------------------------------
    bills_dict_path = os.path.join(data_dir, "all_bills_dict.json")
    cache_path = os.path.join(data_dir, "cache.json")
    
    # Load or build the bills dictionary
    if not os.path.exists(bills_dict_path):
        logger.info('Building bills dictionary from Congress API')
        all_bills_dict = {}
        build_bill_data_dict(API_KEY, all_bills_dict, CONGRESS_NUMBER, offset=0)
        
        with open(bills_dict_path, "w") as f:
            json.dump(all_bills_dict, f, indent=2)
        logger.info(f'Saved bills dictionary to {bills_dict_path}')
    else:
        logger.info(f'Loading bills dictionary from {bills_dict_path}')
        with open(bills_dict_path, "r") as f:
            all_bills_dict = json.load(f)
    
    # Load or build the cache for bill metadata
    if os.path.exists(cache_path):
        logger.info(f'Loading bill metadata cache from {cache_path}')
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        logger.info('Creating new bill metadata cache')
        cache = {}
    
    # Fetch detailed bill data
    logger.info('Fetching detailed bill data for each bill')
    for title_lower, info in tqdm(all_bills_dict.items()):
        bt = info["bill_type"]
        bn = info["bill_number"]
        key = f"{bt},{bn}"
        
        # Fetch data if not in cache
        if key in cache:
            fetched = cache[key]
        else:
            fetched = fetch_and_cache_bill_data(API_KEY, CONGRESS_NUMBER, bt, bn)
            cache[key] = fetched
        
        # Update bill info with detailed data
        info.update(fetched)

    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f'Saved updated bill metadata cache to {cache_path}')
    
    # -----------------------------------
    # 4. Match speeches with bill data
    # -----------------------------------
    logger.info("Matching speeches with bill data")
    bill_data_list = []
    
    for _, row in tqdm(df_bills.iterrows(), total=len(df_bills)):
        bill_data = get_bill_data(row['doc_title'], all_bills_dict)
        bill_data_list.append(bill_data if bill_data else {})

    # Add bill data to the dataframe
    logger.info("Adding bill data columns to dataframe")
    df_bills['standardized_title'] = df_bills['doc_title'].apply(standardize_bill_title)
    df_bills['matched_title'] = [data.get('matched_title', None) for data in bill_data_list]
    df_bills['bill_type'] = [data.get('bill_type', None) for data in bill_data_list]
    df_bills['bill_number'] = [data.get('bill_number', None) for data in bill_data_list]
    df_bills['update_date'] = [data.get('update_date', None) for data in bill_data_list]
    df_bills['policy_area'] = [data.get('policy_area', None) for data in bill_data_list]
    df_bills['legislative_subjects'] = [data.get('legislative_subjects', []) for data in bill_data_list]
    df_bills['latest_summary'] = [data.get('latest_summary', None) for data in bill_data_list]
    df_bills['latest_summary_date'] = [data.get('latest_summary_date', None) for data in bill_data_list]
    df_bills['committees'] = [data.get('committees', []) for data in bill_data_list]

    # Save the final processed dataframe
    output_path = os.path.join(data_dir, output_file)
    df_bills.to_csv(output_path, index=False)
    logger.info(f"Done! Processed data saved to {output_path}")

if __name__ == "__main__":
    main()