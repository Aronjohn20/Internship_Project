import pandas as pd
df = pd.read_csv(
    "data/processed_reviews.csv"
)

df = df.sample(
    n=750,
    random_state=42
)
df.to_csv("data/test_sample.csv", index=False)
