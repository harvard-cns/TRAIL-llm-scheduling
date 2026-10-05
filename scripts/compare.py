#!/usr/bin/env python3
"""Print mean latency per policy and rate from benchmark.py results, with the
speedup of each policy over fcfs.

Usage: python scripts/compare.py results.jsonl
"""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    table = defaultdict(dict)  # rate -> label -> row (last one wins)
    labels = []
    for r in rows:
        table[r["rate"]][r["label"]] = r
        if r["label"] not in labels:
            labels.append(r["label"])
    if "fcfs" in labels:
        labels.remove("fcfs")
        labels.insert(0, "fcfs")

    header = ["rate"] + [f"{label} (s)" for label in labels]
    header += [f"{label} speedup" for label in labels if label != "fcfs"]
    print("Mean latency")
    print(" | ".join(f"{h:>16}" for h in header))
    for rate in sorted(table):
        cells = [f"{rate:g}"]
        for label in labels:
            r = table[rate].get(label)
            cells.append(f"{r['mean_latency']:.2f}" if r else "-")
        base = table[rate].get("fcfs")
        for label in labels:
            if label == "fcfs":
                continue
            r = table[rate].get(label)
            if base and r:
                cells.append(f"{base['mean_latency'] / r['mean_latency']:.2f}x")
            else:
                cells.append("-")
        print(" | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
