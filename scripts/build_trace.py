#!/usr/bin/env python3
"""Build an Alpaca request trace for benchmark.py.

Samples prompts from tatsu-lab/alpaca and records the model's greedy output length for
each one (capped at --max-tokens) by querying a running SGLang server.
Writes one {"prompt": ..., "out_tokens": N} JSON object per line.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def make_prompt(row):
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return f"{instruction}\n\n{inp}".strip() if inp else instruction


async def profile(session, sem, url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    async with sem:
        async with session.post(url + "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return {"prompt": prompt, "out_tokens": int(data["usage"]["completion_tokens"])}


async def main_async(args):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.num_prompts, len(ds))))
    prompts = [p for p in (make_prompt(r) for r in ds) if p]
    print(f"Profiling {len(prompts)} Alpaca prompts on {args.url} ...", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *(profile(session, sem, args.url, args.model, p, args.max_tokens) for p in prompts),
            return_exceptions=True,
        )
    ok = [r for r in results if isinstance(r, dict)]
    failed = len(results) - len(ok)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")

    lens = sorted(r["out_tokens"] for r in ok)
    capped = sum(1 for n in lens if n >= args.max_tokens)
    print(
        f"Wrote {len(ok)} prompts to {args.out} in {time.time() - t0:.0f}s "
        f"({failed} failed). Output tokens: mean {statistics.mean(lens):.0f}, "
        f"median {statistics.median(lens):.0f}, {capped} hit the {args.max_tokens}-token cap",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("-n", "--num-prompts", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "alpaca_trace.jsonl"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
