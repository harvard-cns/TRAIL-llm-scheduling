#!/usr/bin/env python3
"""Open-loop latency benchmark for one SGLang server.

Sends requests from an Alpaca trace (see build_trace.py) with Poisson arrivals at each
given rate and records per-request latency and time to first token. Every request is
sent with ignore_eos=True and max_tokens equal to its profiled output length, so all
policies serve exactly the same work (and TRAIL+ knows the true lengths).

Appends one JSON row per rate to --out. Use compare.py to print a table.
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent


def load_workload(path, n, seed):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if "out_tokens" in r]
    rng = random.Random(seed)
    if n <= len(rows):
        return rng.sample(rows, n)
    return [rng.choice(rows) for _ in range(n)]


def poisson_arrivals(n, rate, seed):
    rng = random.Random(seed + 1)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


async def one_request(session, url, model, row, arrival, start, results):
    loop = asyncio.get_running_loop()
    delay = start + arrival - loop.time()
    if delay > 0:
        await asyncio.sleep(delay)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 0.0,
        "max_tokens": max(int(row["out_tokens"]), 1),
        "ignore_eos": True,
        "stream": True,
    }
    t_start = loop.time()
    ttft = None
    try:
        async with session.post(url + "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.strip()
                if ttft is None and line and line != b"data: [DONE]":
                    ttft = loop.time() - t_start
        latency = loop.time() - t_start
        results.append(
            {"latency": latency, "ttft": ttft or latency, "out_tokens": row["out_tokens"]}
        )
    except Exception as e:  # keep going; failures are counted
        results.append({"error": str(e)[:200]})


async def run_rate(url, model, rows, rate, seed):
    arrivals = poisson_arrivals(len(rows), rate, seed)
    results = []
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start = asyncio.get_running_loop().time() + 1.0
        t0 = time.time()
        await asyncio.gather(
            *(
                one_request(session, url, model, row, arrival, start, results)
                for row, arrival in zip(rows, arrivals)
            )
        )
        duration = time.time() - t0
    return results, duration


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float("nan")


async def main_async(args):
    rows = load_workload(args.trace, args.num_requests, args.seed)
    avg_len = statistics.mean(r["out_tokens"] for r in rows)
    print(
        f"[{args.label}] {len(rows)} requests from {args.trace} "
        f"(mean output {avg_len:.0f} tokens), seed {args.seed}",
        flush=True,
    )
    for rate in [float(x) for x in args.rates.split(",") if x.strip()]:
        results, duration = await run_rate(args.url, args.model, rows, rate, args.seed)
        ok = [r for r in results if "latency" in r]
        lat = [r["latency"] for r in ok]
        ttft = [r["ttft"] for r in ok]
        rec = {
            "label": args.label,
            "rate": rate,
            "num_requests": len(results),
            "completed": len(ok),
            "mean_latency": round(statistics.mean(lat), 3) if lat else None,
            "p50_latency": round(pct(lat, 0.50), 3),
            "p90_latency": round(pct(lat, 0.90), 3),
            "p99_latency": round(pct(lat, 0.99), 3),
            "mean_ttft": round(statistics.mean(ttft), 3) if ttft else None,
            "p50_ttft": round(pct(ttft, 0.50), 3),
            "duration": round(duration, 1),
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[{args.label}] rate={rate:g} req/s  done={len(ok)}/{len(results)}  "
            f"mean latency={rec['mean_latency']}s  p50={rec['p50_latency']}s  "
            f"mean TTFT={rec['mean_ttft']}s  ({rec['duration']}s)",
            flush=True,
        )
        if len(ok) < len(results):
            errors = [r["error"] for r in results if "error" in r]
            print(f"[{args.label}] {len(errors)} failed, e.g. {errors[0]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--label", required=True, help="name for this run, e.g. fcfs or trail")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--trace", default=str(ROOT / "data" / "alpaca_trace.jsonl"))
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("-n", "--num-requests", type=int, default=1000)
    ap.add_argument("--rates", default="10,20,30", help="comma-separated req/s")
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
