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

API_KEY = api_keys.CONGRESS_API_KEY
ALT_API_KEY = api_keys.ALT_CONGRESS_API_KEY
BASE_URL = "https://api.congress.gov/v3"
CONGRESS_NUMBER = 117

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler()  # This will maintain console output
    ]
)

def preprocess_text(text):
    # Replace all consecutive whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    # Replace Madam Speaker, Mr. President, Madam President with Mr. Speaker
    text = re.sub(r'Mr\. President', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chair', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Speakerman', 'Mr. Speaker', text)
    text = re.sub(r'Madam President', 'Mr. Speaker', text)
    text = re.sub(r'Madam Speaker', 'Mr. Speaker', text)
    text = re.sub(r'Madam Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chair', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairwoman', 'Mr. Speaker', text)

    # "Mr. Speaker, " 
    text = re.sub(r'^Mr\. Speaker, ', '', text)
    # Strike starting sentence that starts with "I yield"
    text = re.sub(r'^I yield.*?\. *', '', text)
    # Remove second "Mr. Speaker, " that sometimes follows
    text = re.sub(r'^Mr\. Speaker, ', '', text)

    return text

def standardize_bill_title(title):
    """
    Standardize a bill title by:
    1. Converting to lowercase
    2. Combining multiple spaces into single space
    3. Removing specific prefixes and suffixes
    4. Removing specific procedural phrases
    
    Args:
        title (str): Original bill title
        
    Returns:
        str: Standardized bill title, or None if input is None/empty
    """
    def strip_special_chars(s):
        # [^A-Za-z0-9\s] = any character that is not:
        #   A–Z, a–z, 0–9, or whitespace
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
            
    # If title ends contains "--", remove everything including and after it
    if "--" in title:
        title = title.split("--")[0].strip()

    # Remove trailing period
    if title.endswith(".") or title.endswith(",") or title.endswith(";"):
        title = title[:-1]

    # if title ends with " (executive session)" remove it
    if title.endswith(" (executive session)"):
        title = title[:-20]

    # if title ends with " (executive calendar)" remove it
    if title.endswith(" (executive calendar)"):
        title = title[:-21]

    # if title ends with "; and for other purposes" remove it
    if title.endswith("; and for other purposes") or title.endswith(", and for other purposes"):
        title = title[:-24]

    # Remove ", 20xx" from end of act if it exists
    title = re.sub(r', \d{4}$', '', title)
    
    title = re.sub(r'act of \d{4}$', 'act', title)

    # Remove "of" from end of act if it exists
    title = re.sub(r'act of$', 'act', title)

    title = strip_special_chars(title)
    
    return title if title else None

def get_bill_data(title, all_bills_dict):
    """
    Get the bill data (policy area, legislative subjects, summary, committees).
    Date checks removed - will return most recent data regardless of speech date.
    
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
    Returns:
        {
           "policy_area": {
               "name": str,
               "updateDate": str or None
           },
           "legislative_subjects": [
               {
                 "name": str,
                 "updateDate": str or None
               },
               ...
           ]
        }
    or an empty dict if no data is found or request fails.
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/subjects"
    params = {"api_key": api_key, "format": "json"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logging.info(f"Failed to get bill subjects for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return {}
    data = resp.json()
    
    subjects_data = data.get("subjects", {})
    
    # Policy area
    policy_area = subjects_data.get("policyArea", {})
    
    # Legislative subjects (list)
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
    Returns a list of all summaries, each item:
        {
          "actionDate": str (YYYY-MM-DD),
          "actionDesc": str,
          "text": str,
          "updateDate": str,
          "versionCode": str
        }
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/summaries"
    params = {"api_key": api_key, "format": "json"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logging.info(f"Failed to get bill summaries for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    data = resp.json()
    return data.get("summaries", [])

def get_bill_committees_all(api_key, congress, bill_type, bill_number):
    """
    Returns a list of committees. Each committee looks like:
        {
            "name": str,
            "chamber": str,
            "type": str,
            "activities": [
               {"date": "2023-05-29T18:00:30Z", "name": "Referred To"}, ...
            ],
            "subcommittees": [
               {
                  "name": "...",
                  "activities": [...],
                  ...
               },
               ...
            ]
        }
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/committees"
    params = {"api_key": api_key, "format": "json"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logging.info(f"Failed to get bill committees for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    data = resp.json()
    return data.get("committees", [])

def fetch_and_cache_bill_data(api_key, congress, bill_type, bill_number):
    """
    Fetches all the relevant data for a single bill
    (subjects, policy area, all summaries, committees, etc.)
    Returns a dict with keys:
        "bill_type", "bill_number", "subjects_policy", "summaries", "committees"
    """
    result = {}
    # subjects & policy area
    result["subjects_policy"] = get_bill_subjects_and_policy_area(api_key, congress, bill_type, bill_number)
    # committees
    result["committees"] = get_bill_committees_all(api_key, congress, bill_type, bill_number)
    return result

def build_bill_data_dict(api_key, all_bills, congress=117, limit_per_page=250, offset=0):
    """
    Returns a dictionary: 
      { 
          <lowercase bill title> : {
               "title": str,
               "bill_type": str,
               "bill_number": str,
               "url": str,
               "updateDate": str,
               # plus the deeper fetched data:
               "subjects_policy": {...},
               "summaries": [...],
               "committees": [...]
          },
          ...
      }
    """
    start_time = time.time()
    while True:
        logging.info(f"Fetching bills at offset {offset}, {offset/19315 * 100:.2f}% of the way done!")
        logging.info(f"Time elapsed in minutes: {(time.time() - start_time) / 60:.2f}")
        params = {
            "api_key": api_key,
            "format": "json",
            "offset": offset,
            "limit": limit_per_page
        }
        url = f"{BASE_URL}/bill/{congress}"
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            params = {
                "api_key": ALT_API_KEY,
                "format": "json",
                "offset": offset,
                "limit": limit_per_page
            }
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                logging.info(f"Failed to get bills at offset {offset}: {resp.status_code}")
                break
        data = resp.json()
        
        bills_page = data.get("bills", [])
        if not bills_page:
            break
        
        # For each bill in this page, store minimal metadata
        for b in bills_page:
            offset += 1
            b_title_lower = standardize_bill_title(b.get("title", ""))
            bill_type = b.get("type", "")
            bill_number = b.get("number", "")

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

            toAdd = {
                    "title": b.get("title", "").strip("."),
                    "bill_type": b.get("type", ""),
                    "bill_number": b.get("number", ""),
                    "url": b.get("url", ""),
                    "updateDate": updateDate,
                    "latest_summary": latest_summary,
                    "latest_summary_date": latest_summary_date
                }

            all_titles = get_alias_titles(api_key, congress, bill_type, bill_number)
            all_titles = set(all_titles + [b_title_lower])

            for title in all_titles:
                # we have not seen this bill before
                predicate = title not in all_bills
                if not predicate:
                    # we have seen this bill before, but the summary is outdated
                    if latest_summary_date:
                        if all_bills[title]['latest_summary_date']:
                            # we have seen both summary dates, take the more recent one
                            predicate = (datetime.strptime(all_bills[title]['latest_summary_date'], "%Y-%m-%d").date() < datetime.strptime(latest_summary_date, "%Y-%m-%d").date())
                        else:
                            # default to taking the new one with the summary date
                            predicate = True
                    else:
                        # default to taking the old one with the summary date
                        if all_bills[title]['latest_summary_date']:
                            predicate = False
                        else:
                            # take the one with the more recent update date
                            predicate = datetime.strptime(all_bills[title]['updateDate'], "%Y-%m-%d").date() < datetime.strptime(updateDate, "%Y-%m-%d").date()

                if predicate:
                    all_bills[title] = toAdd

def get_alias_titles(api_key, congress, bill_type, bill_number):
    """
    Returns a list of aliases for the bill
    """
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/titles"
    params = {"api_key": api_key, "format": "json"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        params = {"api_key": ALT_API_KEY, "format": "json"}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logging.info(f"Failed to get bill titles for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    data = resp.json().get("titles", [])
    ret = []
    for title in data:
        alias = standardize_bill_title(title.get("title", ""))
        if alias:
            ret.append(alias)
    return ret

def main():

    # -----------------------------------
    # 1. Read the CSV data
    # -----------------------------------
    logging.info('Starting')
    df_bills_2021 = pd.read_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/crec/crec2021.csv")
    df_bills_2022 = pd.read_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/crec/crec2022.csv")
    df_bills = pd.concat([df_bills_2021, df_bills_2022], ignore_index=True)

    # Remove rows where speaker starts with any of The CLERK, The SPEAKER, The PRESIDENT, The PRESIDING OFFICER, The Acting CHAIR, The CHAIR, The ACTING PRESIDENT case-insensitive
    df_bills = df_bills[~df_bills["speaker"].str.lower().str.startswith(("the clerk", "the speaker", "the president", "the presiding officer", "the acting chair", "the chair", "the acting president"), na=False)]

    # Filter out rows where speech is less than 250 characters, to filter out procedural speecges
    df_bills = df_bills[df_bills['speech'].str.len() >= 250]
    # -----------------------------------
    # 2. Identify potential bills
    #    Simple criterion: title contains the word 'ACT'or 'RESOLUTION'
    # -----------------------------------

    # Drop or fill missing values in the 'title' column
    df_bills = df_bills.dropna(subset=["doc_title"])

    # Filter rows where 'title' contains ' ACT' or 'RESOLUTION'
    df_bills = df_bills[df_bills['doc_title'].str.contains(' ACT', na=False, case=False) | df_bills['doc_title'].str.contains(' RESOLUTION', na=False, case=False)]
    # does not contain "acting chair"
    df_bills = df_bills[~df_bills["doc_title"].str.contains("acting chair", na=False, case=False)]
    df_bills['speech'] = df_bills.apply(lambda x: preprocess_text(x['speech']), axis=1)
    if not os.path.exists("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json"):
        all_bills_dict = {}
        build_bill_data_dict(API_KEY, all_bills_dict, CONGRESS_NUMBER, offset=0)
        with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "w") as f:
            json.dump(all_bills_dict, f, indent=2)
    else:
        print("all bills json loaded.")
        with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "r") as f:
            all_bills_dict = json.load(f)
    # Now for each stored bill, fetch deeper data (subjects, committees, etc.)
    cache = {}
    if os.path.exists("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/cache.json"):
        print("Cache loaded.")
        with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/cache.json", "r") as f:
            cache = json.load(f)
    for title_lower, info in tqdm(all_bills_dict.items()):
        bt = info["bill_type"]
        bn = info["bill_number"]
        key = f"{bt},{bn}"
        # fetch the deeper data
        if key in cache:
            fetched = cache[key]
        else:
            fetched = fetch_and_cache_bill_data(API_KEY, CONGRESS_NUMBER, bt, bn)
            cache[key] = fetched
        info.update(fetched)

    with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/cache.json", "w") as f:
        json.dump(cache, f, indent=2)

    # Apply the function to each row in df_bills
    logging.info("Adding bill data columns...")
    bill_data_list = []
    for _, row in tqdm(df_bills.iterrows(), total=len(df_bills)):
        bill_data = get_bill_data(row['doc_title'], all_bills_dict)
        bill_data_list.append(bill_data if bill_data else {})

    # Add new columns to df_bills
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

    logging.info("Done!")

    df_bills.to_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/df_bills.csv")

if __name__ == "__main__":
    main()