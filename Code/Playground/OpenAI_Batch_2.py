import pandas as pd
from openai import AsyncOpenAI
import asyncio
import time
import math

# --- 1. Load dataset ---
test_csv_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_sample_1000.csv"
df = pd.read_csv(test_csv_path)
print("Loaded", len(df), "reviews\n")

# --- 2. Initialize client ---
aclient = AsyncOpenAI()
MODEL = "gpt-5-nano"

# --- 3. Async safe request wrapper with exponential backoff ---
async def safe_request_async(func, *args, **kwargs):
    delay = 1.0
    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"\nError: {type(e).__name__} - {e}")
            print(f"Retrying in {delay} seconds...\n")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

# --- 4. Async classify-one-review ---
async def classify_one(review_text):
    response = await safe_request_async(
        aclient.responses.create,
        model=MODEL,
        reasoning={"effort": "minimal"},
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Rate this Yelp review from 1 to 5 stars.\n"
                            "Return ONLY the number 1, 2, 3, 4, or 5.\n\n"
                            + review_text
                        )
                    }
                ]
            }
        ],
        max_output_tokens=64
    )

    raw = response.output_text.strip()

    try:
        rating = int(raw)
        return max(1, min(5, rating))  # clamp to [1–5]
    except:
        return None


# GLOBAL RATE LIMITER STATE
rate_limiter_lock = asyncio.Lock()
next_allowed_time = 0.0

async def wait_for_rate_limit(min_interval=0.15):
    global next_allowed_time
    async with rate_limiter_lock:
        now = time.perf_counter()
        if now < next_allowed_time:
            await asyncio.sleep(next_allowed_time - now)
        next_allowed_time = time.perf_counter() + min_interval
    

# --- 5. Async parallel processing of all reviews ---
async def process_all_async(reviews, max_concurrency=8):
    sem = asyncio.Semaphore(max_concurrency)

    async def worker(review):
        # GLOBAL RATE LIMIT (no bursts)
        await wait_for_rate_limit(0.25)   # 150ms = ~400 RPM safe buffer

        async with sem:                   # concurrency guard
            return await classify_one(review)

    tasks = [asyncio.create_task(worker(r)) for r in reviews]
    return await asyncio.gather(*tasks)

# --- 6. Run async pipeline ---
start_time = time.time()

all_reviews = df["text"].tolist()

print("Starting async processing...\n")
predictions = asyncio.run(process_all_async(all_reviews, max_concurrency=50))

elapsed = time.time() - start_time
print(f"\nDone! Processed {len(all_reviews)} reviews in {elapsed:.2f} seconds.")

# --- 7. Save output ---
df["predicted_rating"] = predictions
out_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_predictions_async.csv"
df.to_csv(out_path, index=False)

print("\nSaved predictions to:", out_path)
print("Done!")
