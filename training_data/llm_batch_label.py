import openai
import pandas as pd
import json


df = pd.read_csv("/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/df_bills.csv")
# Initialize OpenAI client
print("Initializing OpenAI client...")
client = openai.OpenAI()

system_message = (
    "You are a helpful assistant that extracts the main topic from congressional speeches. "
    "For the following speech, please output only a short, general topic (1–5 words) that describes the main political issue discussed in the speech. It should be general and unstanced, for example Budget Cuts instead of Opposition to Budget Cuts."
)

with open("to_label.jsonl", "w") as f:
    for i, doc in df.iterrows():
        speech = doc['speech']
        policy = doc['policy_area']
        prompt = (
            "Here we have the text of a congressional speech and a broad policy area assigned by the Congressional Research Service. For the following speech, please output only a short, general topic (1–5 words) that describes the main political issue discussed in the speech. It should be general and unstanced, for example Budget Cuts instead of Opposition to Budget Cuts.\n"
            f"Speech: {speech}\n\n"
            f"Broad Policy Area: {policy}\n\n"
            "Response:\n"
        )
        
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