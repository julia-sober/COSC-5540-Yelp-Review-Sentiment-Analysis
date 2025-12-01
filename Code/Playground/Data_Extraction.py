import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

folder = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/cleaned-data"
csv_files = glob.glob(os.path.join(folder, "*.csv"))

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['state'] = os.path.splitext(os.path.basename(file))[0] 
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.dropna()
print(f"Loaded {len(data)} reviews from {len(dfs)} states.")

sample = data.sample(n=500000) # only using 500,000 data points for the sake of computation time (instead of 5,000,000) (arbitrary)

X = sample['text']
y = sample['stars'] # 1-5 RRP

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Combine X_test and y_test into a single DataFrame
test_df = pd.DataFrame({
    "text": X_test,
    "stars": y_test
})

# Choose the folder to save into
output_folder = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data"

# Make sure the folder exists (this will not overwrite anything)
os.makedirs(output_folder, exist_ok=True)

# Full path for the new CSV
output_path = os.path.join(output_folder, "yelp_test_10pct.csv")

# Export the CSV file
test_df.to_csv(output_path, index=False)

print("Test set exported to:", output_path)