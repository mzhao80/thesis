import pandas as pd
import requests
import datetime
from dateutil import parser
from tqdm.auto import tqdm
import json

API_KEY = "api_key"
BASE_URL = "https://api.congress.gov/v3"
CONGRESS_NUMBER = 118

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
    # summaries
    result["summaries"] = get_bill_summaries_all(api_key, congress, bill_type, bill_number)
    # committees
    result["committees"] = get_bill_committees_all(api_key, congress, bill_type, bill_number)
    return result


with open("all_bills_dict.json", "r") as f:
    all_bills_dict = json.load(f)

# Now for each stored bill, fetch deeper data (subjects, committees, etc.)
i = 0
changes = False
print("Starting to fetch bill metadata...")
for title_lower, info in tqdm(all_bills_dict.items()):
    i += 1
    # if we already have the deeper data, skip
    if 'subjects_policy' not in info or not info['subjects_policy']:
        bt = info["bill_type"]
        bn = info["bill_number"]
        # fetch the deeper data
        fetched = fetch_and_cache_bill_data(API_KEY, CONGRESS_NUMBER, bt, bn)
        info.update(fetched)
        changes = True
    
    # Save progress every 100 iterations
    if i % 100 == 0 and changes:
        changes = False
        print(f"\nSaving progress at iteration {i}...")
        with open("all_bills_dict.json", "w") as f:
            json.dump(all_bills_dict, f, indent=2)
        print("Progress saved.")

# Final save
print("\nSaving final results...")
with open("all_bills_dict.json", "w") as f:
    json.dump(all_bills_dict, f, indent=2)
print("Done!")