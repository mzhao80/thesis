from openai import OpenAI
client = OpenAI()

response = client.batches.retrieve("batch_67995c14d934819089db703d0b174e83")

print(response)