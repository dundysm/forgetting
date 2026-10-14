"""Pack replayed runs into compact binaries for the web piece.

Usage: python tools/pack.py build/replay web/data
Writes <run>.bin (struct of arrays: t float32 ms | write float32 ms | block uint16 | index int16 | kind uint8 | session int8)
and index.json with each run's metadata, its per-request records and the measured headline numbers.
"""
import json, statistics as st, sys
from pathlib import Path

import numpy as np

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KIND = {"born": 0, "recall": 1, "release": 2, "evict": 3, "evict_scratch": 4}
RUNS = [  # (file, label) one run per budget; rep 1 of the main cohort where it exists
    ("pilot-2.5gib-rep_1", "2.5"), ("pilot-3gib-rep_1", "3"), ("main-4.5gib-rep_1", "4.5"),
    ("pilot-6gib-rep_1", "6"), ("pilot-8gib-rep_1", "8"), ("main-10gib-rep_1", "10"),
]
OUT.mkdir(parents=True, exist_ok=True)
summary = {s["cohort"] + "-" + f"{s['gib']:g}gib-" + s["rep"]: s for s in json.load(open(SRC / "summary.json"))}
index = {"block_tokens": 16, "working_set_tokens": 54173, "kinds": list(KIND), "runs": []}
for name, gib in RUNS:
    d = json.load(open(SRC / f"{name}.json"))
    ev = d["events"]
    t = np.array([e[0] for e in ev], np.float32)
    kind = np.array([KIND[e[1]] for e in ev], np.uint8)
    block = np.array([e[2] for e in ev], np.uint16)
    sess = np.array([e[3] for e in ev], np.int8)
    idx = np.array([max(-1, min(32767, e[4])) for e in ev], np.int16)
    write = np.array([e[5] if len(e) > 5 else 0.0 for e in ev], np.float32)   # born: ms to write (TTFT for prefill)
    blob = t.tobytes() + write.tobytes() + block.tobytes() + idx.tobytes() + kind.tobytes() + sess.tobytes()
    (OUT / f"{name}.bin").write_bytes(blob)
    reqs = d["requests"]
    res = [r for r in reqs if r["turn"] > 0]
    s = summary[name]
    index["runs"].append({
        "file": f"{name}.bin", "gib": gib, "cohort": s["cohort"], "rep": s["rep"], "kv_tokens": d["kv_tokens"],
        "n_blocks": d["n_blocks"], "n_events": len(ev), "end_ms": float(t.max()),
        "resumed_reuse_median": st.median(r["measured"] / r["prompt"] for r in res),
        "measured_cached_tokens": s["measured_cached"], "replay_exact": s["exact"],
        "inconsistent": s["measured_hits_not_in_replay_cache"],
        "counts": {k: int((kind == v).sum()) for k, v in KIND.items()},
    })
    print(name, len(ev), "events", len(blob) // 1024, "KiB")
json.dump(index, open(OUT / "index.json", "w"), indent=1)
