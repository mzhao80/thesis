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
congress = 117

# with open("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/all_bills_dict.json", "r") as f:
#     all_bills_dict = json.load(f)
# test = all_bills_dict.get("recovering america's wildlife act")
# print(test['title'])

bill_type = "hr"
bill_number = 8326
url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/subjects"
params = {"api_key": api_key, "format": "json"}
resp = requests.get(url, params=params)
print(resp.status_code)
print(resp.json())