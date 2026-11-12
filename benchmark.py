"""Compare local models on 10 prompts: speed, cost (tokens) and quality.

Step 1 - run:     python benchmark.py --models llama3.2 qwen2.5:3b
Step 2 - score:   open results.csv and fill the quality_1to5 column
Step 3 - summary: python benchmark.py --summarize results.csv
"""
import argparse
import asyncio
import csv
import time

from providers import SYSTEM_PROMPT, Usage, get_provider

PROMPTS = [
    "Explain what an API is to a complete beginner.",
    "What is the difference between a list and a tuple in Python?",
    "Give me 3 tips to prepare for a job interview.",
    "Summarise in one sentence: Machine learning lets computers learn patterns "
    "from data instead of being told exact rules for every case.",
    "If a train travels 120 km in 1.5 hours, what is its average speed?",
    "Write a Python function that checks if a word is a palindrome.",
    "Why is the sky blue? Answer in exactly two sentences.",
    "My laptop is slow. What should I check first?",
    "What is overfitting? Use a real-life analogy.",
    "Rewrite politely: 'Send me the report now, you are late.'",
]

FIELDS = ["model", "prompt_id", "prompt", "answer", "ttft_s", "total_s",
          "input_tokens", "output_tokens", "tokens_per_s", "quality_1to5"]


async def run_one(llm, prompt: str) -> dict:
    start = time.perf_counter()
    first_token = None
    parts, usage = [], Usage(0, 0)
    async for piece in llm.stream(SYSTEM_PROMPT, [{"role": "user", "content": prompt}]):
        if isinstance(piece, Usage):
            usage = piece
        else:
            if first_token is None:
                first_token = time.perf_counter() - start  # time to FIRST token
            parts.append(piece)
    total = time.perf_counter() - start
    return {
        "answer": "".join(parts).strip(),
        "ttft_s": round(first_token or total, 3),
        "total_s": round(total, 3),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "tokens_per_s": round(usage.output_tokens / total, 1) if total else 0,
    }


async def run_all(models: list[str]) -> list[dict]:
    rows = []
    for model in models:
        llm = get_provider(model)
        print(f"\n== {model} ==")
        try:
            # Warm-up: the model loads into memory on the first call, so we don't time it
            await run_one(llm, "Say hi.")
        except Exception as e:
            print(f"  Could not use '{model}': {e}")
            print("  Is Ollama running? Did you run: ollama pull " + model)
            continue
        for i, prompt in enumerate(PROMPTS, 1):
            r = await run_one(llm, prompt)
            rows.append({"model": model, "prompt_id": i, "prompt": prompt,
                         "quality_1to5": "", **r})
            print(f"  prompt {i:>2}: first token {r['ttft_s']}s, total {r['total_s']}s")
    return rows


def summarize(rows: list[dict]):
    print("\nmodel          | first token (s) | total (s) | out tokens | quality (1-5)")
    print("-" * 70)
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        n = len(rs)
        scores = [float(r["quality_1to5"]) for r in rs if r["quality_1to5"] not in ("", None)]
        quality = f"{sum(scores)/len(scores):.2f}" if scores else "not scored yet"
        print(f"{model:14} | {sum(float(r['ttft_s']) for r in rs)/n:15.2f} | "
              f"{sum(float(r['total_s']) for r in rs)/n:9.2f} | "
              f"{sum(float(r['output_tokens']) for r in rs)/n:10.0f} | {quality}")
    print("\nCost: every local model is $0 per question. 'out tokens' is your cost proxy:")
    print("on a paid API you pay per token, so fewer tokens would mean a smaller bill.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2"])
    ap.add_argument("--summarize", metavar="CSV", help="print the summary from a scored CSV")
    args = ap.parse_args()

    if args.summarize:
        with open(args.summarize, newline="", encoding="utf-8") as f:
            summarize(list(csv.DictReader(f)))
        return

    rows = asyncio.run(run_all(args.models))
    if not rows:
        return
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("\nSaved results.csv. Now score each answer 1-5 in the quality_1to5 column.")
    summarize(rows)


if __name__ == "__main__":
    main()
