import pandas as pd
import requests
from datetime import datetime
from dateutil import parser
from tqdm.auto import tqdm
import json
import re
import api_keys
import os

API_KEY = api_keys.CONGRESS_API_KEY
ALT_API_KEY = api_keys.ALT_CONGRESS_API_KEY
BASE_URL = "https://api.congress.gov/v3"
CONGRESS_NUMBER = 118

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

    title = strip_special_chars(title)
        
    # Convert to lowercase
    title = title.lower()
    
    # Combine multiple spaces into one
    title = ' '.join(title.split())
    
    # Skip titles that begin with "authorizing the clerk"
    if title.startswith("authorizing the clerk"):
        return None
        
    # Remove various prefixes up to the next comma
    prefixes_with_comma = [
        "providing for consideration of",
        "providing for further consideration of",
        "report on",
        "motion to",
        "concerns about",
        "conference report on"
    ]
    
    for prefix in prefixes_with_comma:
        if title.startswith(prefix):
            comma_pos = title.find(", ")
            if comma_pos != -1:
                title = title[comma_pos + 2:]  # +2 to skip the comma and space
                break
    
    # Remove simple prefixes
    simple_prefixes = [
        "supporting the ",
        "introduction of the ",
        "reintroduction of the ",
        "re-introduction of the "
        "introduction of ",
        "reintroduction of "
    ]
    
    for prefix in simple_prefixes:
        if title.startswith(prefix):
            title = title[len(prefix):]
            
    # Remove trailing period or "--"
    if title.endswith(".") or title.endswith(","):
        title = title[:-1]
    if title.endswith("--"):
        title = title[:-2]
        
    # Trim any resulting whitespace
    title = title.strip()

    # If title ends with "continued", remove it and 2 characters before it
    if title.endswith("continued"):
        title = title[:-11]

    # If title ends with "motion to proceed", remove it and 2 characters before it
    if title.endswith("motion to proceed"):
        title = title[:-19]

    # if title ends with " (Executive Session)" remove it
    if title.endswith(" (Executive Session)"):
        title = title[:-20]

    # Remove ", 20xx" from end of act if it exists
    title = re.sub(r', \d{4}$', '', title)

    # Remove " of 20xx" from end of act if it exists
    title = re.sub(r' of \d{4}$', '', title)

    # Remove "of" from end of act if it exists
    title = re.sub(r' of$', '', title)

    title = title.strip(".")
    
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
        'bill_type': bill_data.get('bill_type') or None,
        'bill_number': bill_data.get('bill_number') or None,
        'policy_area': None,
        'policy_area_update': None,
        'legislative_subjects': [],
        'latest_summary': None,
        'latest_summary_date': None,
        'committees': []
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
    
    # Get most recent summary
    summaries = bill_data.get('summaries', [])
    if summaries:
        # Get the most recent summary
        latest_summary = max(summaries, key=lambda x: parser.parse(x['actionDate']).date() if x.get('actionDate') else parser.parse('1900-01-01').date())
        result['latest_summary'] = latest_summary.get('text')
        if latest_summary.get('actionDate'):
            result['latest_summary_date'] = parser.parse(latest_summary['actionDate']).date().isoformat()
    
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
            print(f"Failed to get bill subjects for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
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
            print(f"Failed to get bill summaries for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
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
            print(f"Failed to get bill committees for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
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

def build_bill_data_dict(api_key, all_bills, congress=118, limit_per_page=250, offset=0):
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
    report = 1000
    while True:
        params = {
            "api_key": api_key,
            "format": "json",
            "offset": offset,
            "limit": limit_per_page
        }
        url = f"{BASE_URL}/bill/{congress}"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
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
            if summaries:
                # Get the most recent summary
                latest_summary = max(summaries, key=lambda x: parser.parse(x['actionDate']).date() if x.get('actionDate') else parser.parse('1900-01-01').date())
                latest_summary = latest_summary.get('text', "")
                if latest_summary.get('actionDate', ""):
                    latest_summary_date = parser.parse(latest_summary['actionDate']).date().isoformat()

            toAdd = {
                    "title": b.get("title", "").strip("."),
                    "bill_type": b.get("type", ""),
                    "bill_number": b.get("number", ""),
                    "url": b.get("url", ""),
                    "updateDate": b.get("updateDateIncludingText", ""),
                    "latest_summary": latest_summary,
                    "latest_summary_date": latest_summary_date
                }

            all_titles = get_alias_titles(api_key, congress, bill_type, bill_number)
            all_titles = set(all_titles + [b_title_lower])

            for title in all_titles:
                if title not in all_bills or (datetime.strptime(all_bills[title]['latest_summary_date'], "%Y-%m-%d").date() < datetime.strptime(latest_summary_date, "%Y-%m-%d").date()):
                    all_bills[title] = toAdd

        if offset >= report:
            print(offset)
            report += 1000

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
            print(f"Failed to get bill titles for {congress}, {bill_type}, {bill_number}: {resp.status_code}")
            return []
    data = resp.json().get("titles", [])
    ret = []
    for title in data:
        title = standardize_bill_title(title)
        ret.append(title)
    return ret

def main():

    # -----------------------------------
    # 1. Read the CSV data
    # -----------------------------------
    df_bills = pd.read_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/crec2023.csv")

    # Remove rows where speaker starts with any of The CLERK, The SPEAKER, The PRESIDENT, The PRESIDING OFFICER, The Acting CHAIR, The CHAIR, The ACTING PRESIDENT case-insensitive
    df_bills = df_bills[~df_bills["speaker"].str.lower().str.startswith(("the clerk", "the speaker", "the president", "the presiding officer", "the acting chair", "the chair", "the acting president"), na=False)]

    # Filter out rows where speech is less than 250 characters, to filter out procedural speehces
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
        # Now for each stored bill, fetch deeper data (subjects, committees, etc.)
        for title_lower, info in tqdm(all_bills_dict.items()):
            # if we already have the deeper data, skip
            bt = info["bill_type"]
            bn = info["bill_number"]
            # fetch the deeper data
            fetched = fetch_and_cache_bill_data(API_KEY, CONGRESS_NUMBER, bt, bn)
            info.update(fetched)

        with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "w") as f:
            json.dump(all_bills_dict, f, indent=2)
    else:
        with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "r") as f:
            all_bills_dict = json.load(f)

    # Apply the function to each row in df_bills
    print("Adding bill data columns...")
    bill_data_list = []
    for _, row in tqdm(df_bills.iterrows(), total=len(df_bills)):
        bill_data = get_bill_data(row['doc_title'], all_bills_dict)
        bill_data_list.append(bill_data if bill_data else {})

    # Add new columns to df_bills
    df_bills['standardized_title'] = df_bills['doc_title'].apply(standardize_bill_title)
    df_bills['bill_type'] = [data.get('bill_type') for data in bill_data_list]
    df_bills['bill_number'] = [data.get('bill_number') for data in bill_data_list]
    df_bills['policy_area'] = [data.get('policy_area') for data in bill_data_list]
    df_bills['legislative_subjects'] = [data.get('legislative_subjects', []) for data in bill_data_list]
    df_bills['latest_summary'] = [data.get('latest_summary') for data in bill_data_list]
    df_bills['committees'] = [data.get('committees', []) for data in bill_data_list]

    print("Done!")

    df_bills.to_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/df_bills.csv")

if __name__ == "__main__":
    main()