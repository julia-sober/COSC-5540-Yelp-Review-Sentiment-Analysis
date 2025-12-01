import pandas as pd
from openai import OpenAI
import time
import os

# --- 1. Load dataset ---
test_csv_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_sample_1000.csv"
df = pd.read_csv(test_csv_path)
print("Loaded test dataset with:", len(df), "reviews")

# --- 2. Initialize OpenAI client ---
client = OpenAI()
MODEL = "gpt-5-nano"   # YES — safe to use now!


# --- 3. Safe API wrapper (handles rate limits & errors) ---
def safe_call(func, *args, **kwargs):
    delay = 1  # starting backoff
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n Error (likely rate limit or network): {e}")
            print(f" Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, 45)  # exponential backoff

# --- 4. Use RESPONSES API for gpt-5-nano ---
def classify_review(text: str) -> int:

    prompt = (
        "Rate this Yelp review from 1 to 5 stars.\n"
        "Return ONLY the number 1, 2, 3, 4, or 5.\n\n"
        + text
    )

    # gpt-5-nano ONLY works with responses.create (not chat.completions!)
    response = safe_call(
        client.responses.create,
        model=MODEL,
        reasoning = {"effort": "minimal"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        max_output_tokens=64,
    )

    # --- Extract output correctly ---
    rating_text = None
    for item in response.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    rating_text = c.text.strip()
                    break
        if rating_text:
            break

    if rating_text is None:
        print(" No output_text found. Full response:", response)
        return None

    try:
        return max(1, min(5, int(rating_text)))
    except:
        return None


# --- 5. Run predictions ---
predictions = []

for i, text in enumerate(df["text"]):
    print(f"Processing review {i+1}/{len(df)}...")
    pred = classify_review(text)
    predictions.append(pred)

    # Small pause to reduce rate-limit hits
    time.sleep(0.15)

df["predicted_rating"] = predictions

# --- 6. Save results ---
output_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_predictions.csv"
df.to_csv(output_path, index=False)

print("\nSaved predictions to:", output_path)
