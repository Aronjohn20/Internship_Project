import json
import pandas as pd

MAX_REVIEWS = 200000
data = []

print("Streaming raw JSON lines...")
with open('Sports_and_Outdoors.json', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i >= MAX_REVIEWS:
            break
        data.append(json.loads(line))

print("Writing directly to CSV...")
df = pd.DataFrame(data)
df.to_csv('sampled_reviews.csv', index=False)
print("Done! CSV created.")