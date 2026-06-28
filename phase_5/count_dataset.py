import json
paths = ["data/dataset_v3_expert.jsonl", "data/dataset_v3_beam.jsonl"]
total = 0
for p in paths:
    c = 0
    with open(p) as f:
        for _ in f: c += 1
    total += c
    print(f"{p}: {c} lines")
print(f"Total lines: {total}")
