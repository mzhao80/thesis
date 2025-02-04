import openai
import pandas as pd
import json


df = pd.read_csv("df_bills.csv")
# Initialize OpenAI client
print("Initializing OpenAI client...")
client = openai.OpenAI()

system_message = {
    "role": "system", 
    "content": """You are a helpful assistant that analyzes congressional speeches and extracts topics and key phrases from them.
    For each speech, provide:
    1. A short topic phrase (1-5 words) describing the main political issue discussed in the speech.
    2. A key phrase or list of key phrases (2-10 words each) exactly quoted from the speech that best represent this topic you chose, presented without quotation marks.
    3. A key phrase or list of key phrases (2-10 words each) exactly quoted from the speech that best represent the broad policy area provided by the Congressional Research Service, presented without quotation marks.
    
    Format your response exactly as follows on three lines:
    <topic>
    <yourphrase1> | <yourphrase2> | <yourphrase3>
    <providedphrase1> | <providedphrase2> | <providedphrase3>"""
}

with open("to_label.jsonl", "w") as f:
    for i, doc in df.iterrows():
        prompt = f"""Analyze this congressional speech and extract its main topic and key topical phrases:

        Speech: {doc['speech']}

        Congressional Research Service provided broad policy area: {doc['policy_area']}

        Response:"""
        
        toWrite = {"custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini", 
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            }
        }
    
        json.dump(toWrite, f)
        f.write('\n')

print("Done writing jsonl file")

# Upload jsonl
upload_response = client.files.create(
    file=open("to_label.jsonl", "rb"),
    purpose="batch"
)

print(upload_response.id)

batch_response = client.batches.create(
    input_file_id=upload_response.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(batch_response)