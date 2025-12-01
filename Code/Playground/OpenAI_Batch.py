import pandas as pd
from openai import OpenAI
import math, time, os
import json

# --- 1. Load dataset ---
test_csv_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_sample_1000.csv"
df = pd.read_csv(test_csv_path)
print("Loaded", len(df), "reviews\n")

# --- 2. Initialize client ---
client = OpenAI()
MODEL = "gpt-5-nano"

# --- 3. Safe request wrapper (handles rate limits & retries) ---
def safe_request(func, *args, **kwargs):
    delay = 1.0  # start at 1 second
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("\n Error:", type(e).__name__, "-", e)
            print(f" Retrying in {delay} seconds...\n")
            time.sleep(delay)
            delay = min(delay * 2, 60)  # exponential backoff

# --- 4. Batch classifier ---
def classify_batch(text_list):
    # Build a numbered list of reviews for the prompt
    numbered_reviews = "\n".join(
        f'{i+1}. "{review}"' for i, review in enumerate(text_list)
    )

    # Prompt enforcing structured JSON output
    prompt = f"""
You are a Yelp review star-rating classifier.

For EACH review below, output a star rating from 1 to 5 (integers only).

Return EXACTLY AND ONLY valid JSON in this format:

{{
  "1": 3,
  "2": 5,
  "3": 1
}}

Where the keys are the review numbers (as strings) and the values are integers 1–5.

Reviews:
{numbered_reviews}
"""

    # Single Responses API request
    response = safe_request(
        client.responses.create,
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        max_output_tokens=200,
        reasoning={"effort": "minimal"},
    )

    # Use the combined text output — NOT response.output[…]
    raw_text = response.output_text.strip()

    # Try to parse JSON
    try:
        parsed = json.loads(raw_text)
    except Exception:
        print("\n--- JSON PARSE ERROR ---")
        print(raw_text)
        print("------------------------\n")
        return [None for _ in text_list]

    # Map JSON output back to original order
    preds = []
    for i in range(len(text_list)):
        key = str(i + 1)
        val = parsed.get(key, None)
        try:
            rating = int(val)
            preds.append(max(1, min(5, rating)))
        except:
            preds.append(None)

    return preds

# --- 5. Process batches ---
BATCH_SIZE = 20   # <-- lowered to avoid TPM errors

predictions = []
num_reviews = len(df)
num_batches = math.ceil(num_reviews / BATCH_SIZE)

for i in range(num_batches):
    start = i * BATCH_SIZE
    end = min((i + 1) * BATCH_SIZE, num_reviews)

    batch_texts = df["text"].iloc[start:end].tolist()
    print(f"=== Processing batch {i+1}/{num_batches} ({len(batch_texts)} reviews) ===")

    batch_preds = classify_batch(batch_texts)
    predictions.extend(batch_preds)

    time.sleep(0.5)  # <-- smooth out token-per-minute usage

# --- 6. Save output ---
df["predicted_rating"] = predictions
out_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_predictions_batch.csv"
df.to_csv(out_path, index=False)

print("\nSaved predictions to:", out_path)
print("Done!")
