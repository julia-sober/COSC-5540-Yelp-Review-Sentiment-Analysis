import pandas as pd
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the file with true and predicted ratings
df = pd.read_csv("/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/merged_yelp_predictions_10pct.csv")

# Compute accuracy (exact match)
accuracy = (df["stars"] == df["predicted_rating"]).mean()

print("Accuracy:", accuracy)

y_true = df["stars"].astype(int)
y_pred = df["predicted_rating"].astype(int)


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[1,2,3,4,5],
            yticklabels=[1,2,3,4,5])

plt.xlabel("Predicted Rating")
plt.ylabel("True Rating")
plt.title("Confusion Matrix for Yelp Star Prediction")
plt.tight_layout()
plt.show()



# Load merged dataset (has 'stars' and 'predicted_rating')
df = pd.read_csv("/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/merged_yelp_predictions_10pct.csv")

interval = 1000
n = len(df)

accuracies = []

for start in range(0, n, interval):
    end = min(start + interval, n)
    chunk = df.iloc[start:end]

    acc = (chunk["stars"] == chunk["predicted_rating"]).mean()
    accuracies.append((start, end, acc))

    print(f"Accuracy for rows {start} to {end}: {acc:.4f}")