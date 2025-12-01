import pandas as pd
from openai import AsyncOpenAI
import asyncio
import time
import os

# -----------------------------------
# 1. Load CSV
# -----------------------------------
input_path = "/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_test_10pct.csv"
df = pd.read_csv(input_path)
all_reviews = df["text"].tolist()
TOTAL = len(all_reviews)

print(f"Loaded {TOTAL} reviews.\n")

# -----------------------------------
# 2. Async client
# -----------------------------------
aclient = AsyncOpenAI()
MODEL = "gpt-5-nano"

# -----------------------------------
# 3. Global rate limiter
# -----------------------------------
rate_limiter_lock = asyncio.Lock()
next_allowed_time = 0.0

async def wait_for_rate_limit(min_interval=0.20):  # 200ms = ~300 RPM safe buffer
    global next_allowed_time
    async with rate_limiter_lock:
        now = time.perf_counter()
        if now < next_allowed_time:
            await asyncio.sleep(next_allowed_time - now)
        next_allowed_time = time.perf_counter() + min_interval

# -----------------------------------
# 4. Safe async request wrapper
# -----------------------------------
async def safe_request_async(func, *args, **kwargs):
    delay = 1.0
    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {type(e).__name__} - {e}")
            print(f"Retrying in {delay} sec...\n")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)  # exponential backoff

# -----------------------------------
# 5. WORKING classify_one() function
# -----------------------------------
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
        return max(1, min(5, rating))
    except:
        return None

# -----------------------------------
# 6. Progress + checkpoint state
# -----------------------------------
progress = 0
progress_lock = asyncio.Lock()

checkpoint_file = "checkpoint_predictions.csv"
predictions = []

# -----------------------------------
# 7. If checkpoint exists, resume
# -----------------------------------
if os.path.exists(checkpoint_file):
    cp = pd.read_csv(checkpoint_file)
    done = len(cp)
    predictions = cp["predicted_rating"].tolist()
    progress = done
    print(f"Resuming from checkpoint: {done}/{TOTAL} completed.\n")
else:
    print("No existing checkpoint. Starting fresh.\n")

# -----------------------------------
# 8. Async worker (rate limited + checkpointing)
# -----------------------------------
async def worker(review):
    global progress, predictions

    await wait_for_rate_limit(0.20)

    rating = await classify_one(review)

    async with progress_lock:
        predictions.append(rating)
        progress += 1

        # progress print
        if progress % 100 == 0:
            pct = (progress / TOTAL) * 100
            print(f"{progress}/{TOTAL} ({pct:.1f}%)")

        # checkpoint every 500
        if progress % 500 == 0:
            cp_df = pd.DataFrame({
                "text": all_reviews[:progress],
                "predicted_rating": predictions
            })
            cp_df.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved at {progress} reviews.\n")

    return rating

# -----------------------------------
# 9. Main async processor
# -----------------------------------
async def process_all_async(reviews, max_concurrency=6):
    sem = asyncio.Semaphore(max_concurrency)

    # Skip any we already processed (resume mode)
    remaining = reviews[progress:]

    tasks = []
    for r in remaining:
        async def run(review=r):
            async with sem:
                return await worker(review)
        tasks.append(asyncio.create_task(run()))

    return await asyncio.gather(*tasks)

# -----------------------------------
# 10. Run job (with crash-safe fallback)
# -----------------------------------
start = time.time()

try:
    asyncio.run(process_all_async(all_reviews, max_concurrency=6))
except Exception as e:
    print("FATAL ERROR:", e)
    partial = pd.DataFrame({
        "text": all_reviews[:progress],
        "predicted_rating": predictions
    })
    partial.to_csv("partial_results.csv", index=False)
    print("Partial results saved.")

elapsed = time.time() - start
print(f"\nFinished {progress}/{TOTAL} reviews in {elapsed/60:.1f} minutes.")

# -----------------------------------
# 11. Final save
# -----------------------------------

out_path ="/Users/cole/Desktop/Georgetown/Fall 2025/Text Mining & Analysis 5540/Playground/test-data/yelp_predictions_final.csv"
df_out = pd.DataFrame({
    "text": all_reviews,
    "predicted_rating": predictions
})
df_out.to_csv(out_path, index=False)

print("\nSaved final predictions to:", out_path)
print("Done!")
