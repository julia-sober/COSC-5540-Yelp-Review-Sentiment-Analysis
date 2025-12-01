import pandas as pd

# Load the full 50,000-row CSV
df = pd.read_csv("/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_10pct.csv")

# Randomly sample 1,000 rows (set random_state for reproducibility)
df_sample = df.sample(n=1000, random_state=42)

# Save to a new CSV
df_sample.to_csv("yelp_sample_1000.csv", index=False)

print("Saved sample dataset with", len(df_sample), "rows.")
