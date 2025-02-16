import pandas as pd
import requests
from datetime import datetime
from dateutil import parser
from tqdm.auto import tqdm
import json
import re
import api_keys
import os

api_key = api_keys.CONGRESS_API_KEY
BASE_URL = "https://api.congress.gov/v3"
congress = 118

# with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "r") as f:
#     all_bills_dict = json.load(f)
# test = all_bills_dict.get("recovering america's wildlife act")
# print(test['title'])

bill_type = "s"
bill_number = 1172
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
url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/titles"
params = {"api_key": api_key, "format": "json"}
resp = requests.get(url, params=params)
if resp.status_code != 200:
    print("error")
data = resp.json().get("titles", [])
print(data)