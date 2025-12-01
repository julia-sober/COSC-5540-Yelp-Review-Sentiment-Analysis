import pandas as pd

# Original labeled dataset
original_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_10pct.csv"
df_orig = pd.read_csv(original_path)

# Predictions-only output (from async job)
pred_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_predictions_final.csv"
df_pred = pd.read_csv(pred_path)

# Sanity check: lengths must match
assert len(df_orig) == len(df_pred), (
    f"Row mismatch: original={len(df_orig)}, predictions={len(df_pred)}"
)

# Attach predictions into the original dataframe
df_orig["predicted_rating"] = df_pred["predicted_rating"]

# Save merged output (10% version)
merged_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/merged_yelp_predictions_10pct.csv"
df_orig.to_csv(merged_path, index=False)

print("Merged file saved to:", merged_path)
