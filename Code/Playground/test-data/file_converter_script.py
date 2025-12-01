import pandas as pd
import json

# === Load your CSV file ===
df = pd.read_csv("test_set.csv")   # adjust path if needed

# === Create JSONL batch file ===
with open("batch_requests.jsonl", "w") as f:
    for i, row in df.iterrows():
        
        # Each line must be a complete API request
        request = {
            "custom_id": f"review-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-5-nano",   # ← your model
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Rate this Yelp review from 1–5. "
                                    "Return ONLY the digit.\n\n"
                                    + row["text"]
                                )
                            }
                        ]
                    }
                ],
                "max_output_tokens": 8
            },
            # Optional: include original label for later evaluation
            "metadata": {
                "true_stars": int(row["stars"])
            }
        }

        f.write(json.dumps(request) + "\n")

print("Created batch_requests.jsonl")
