"""Replay vLLM v1's KV block manager over a measured run and emit every block's lifecycle.

Usage: python tools/replay.py <runs dir: results/2026-10-12-runpod-a40> <out dir>

Model (vLLM v1 prefix caching, block size 16):
- A request's prompt is its session's history so far (append-only, scripted). Block 0 is a header
  shared by every session; block i>0 of session s has the same content in every later turn of s,
  so its hash is (s, i). The block holding the end of the prompt also gets generated tokens, which
  the next scripted prompt never repeats, so it and all decode blocks are unique to their request.
- At request start: longest chain of cached full blocks is a hit (capped so at least one prompt
  token is computed); the rest is allocated from the head of the free queue, evicting whatever
  cached block was there. Decode blocks are allocated when the first token appears.
- At request end: all of the request's blocks go to the tail of the free queue in reverse order,
  so a request's tail is evicted before its head.
- Pool: kv_tokens / 16 blocks, minus vLLM's reserved null block. Starts empty.

The measured inputs are the request timestamps, sizes and order; the replay's predicted cached
tokens are checked against each request's measured `cached_tokens`.
"""
import glob, json, re, sys
from collections import OrderedDict
from pathlib import Path

B = 16
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def replay(rows, n_blocks):
    rows = sorted(rows, key=lambda r: r["start_unix"])
    t0 = min(r["start_unix"] for r in rows)
    free = OrderedDict((b, None) for b in range(n_blocks))  # block id -> None; head = next evicted
    key_of = {}       # block id -> hash key (cached) or None
    by_key = {}       # hash key -> block id
    ref = [0] * n_blocks
    events = []       # (t_ms, kind, block, session, index)
    per_req = []
    prev_prompt = {}  # session -> previous prompt length
    uniq = [0]
    missing = [0]
    owner_of = {}

    def evict_alloc(t, s, idx, k):
        b, _ = free.popitem(last=False)
        old = key_of.get(b)
        if old is not None:
            if by_key.get(old) == b:
                del by_key[old]
            # who wrote it, and could it ever have been reused (prompt blocks yes; request-unique scratch no)
            reusable = not (isinstance(old, tuple) and old[0] == "u")
            events.append((t, "evict" if reusable else "evict_scratch", b, owner_of[b], old[-1] if reusable else -1))
        key_of[b] = k
        owner_of[b] = s
        if k is not None and not (isinstance(k, tuple) and k[0] == "u"):
            by_key[k] = b
        ref[b] = 1
        events.append((t, "born", b, s, idx))
        return b

    timeline = []
    for r in rows:
        timeline.append((r["start_unix"], 0, r))
        timeline.append((r["start_unix"] + r["ttft_ms"] / 1000, 1, r))
        timeline.append((r["end_unix"], 2, r))
    timeline.sort(key=lambda x: (x[0], x[1]))
    held = {}
    for ts, kind, r in timeline:
        t = round((ts - t0) * 1000, 1)
        s, P = r["session"], r["prompt_tokens"]
        rid = (s, r["turn"])
        if kind == 0:
            full = P // B
            def key(i):
                if i == 0:
                    return ("h", 0)
                if i == full and P % B:          # prompt tail + generated tokens
                    return ("x", s, i)
                return (s, i)
            pp = prev_prompt.get(s)
            def hit_key(i):
                # the block straddling the previous prompt's end held generated tokens
                if pp is not None and i == pp // B and i > 0:
                    return ("x", s, i)
                return ("h", 0) if i == 0 else (s, i)
            # model prediction (generated tokens never match the next scripted prompt)
            pred = 0
            for i in range((P - 1) // B):
                k = hit_key(i)
                if k in by_key and k[0] != "x":
                    pred += 1
                else:
                    break
            # measured reuse drives the replay; every measured hit must still be cached here
            hit = r["cached_tokens"] // B
            blocks = []
            for i in range(hit):
                k = hit_key(i)
                b = by_key.get(k)
                if b is None:
                    missing[0] += 1
                    b = evict_alloc(t, s, i, key(i))
                else:
                    if ref[b] == 0:
                        free.pop(b)
                    ref[b] += 1
                    events.append((t, "recall", b, s, i))
                    if k != key(i):          # straddle block becomes a plain prompt block
                        del by_key[k]; key_of[b] = key(i); by_key[key(i)] = b
                blocks.append(b)
            for i in range(hit, -(-P // B)):
                blocks.append(evict_alloc(t, s, i, key(i)))
            held[rid] = blocks
            prev_prompt[s] = P
            per_req.append({"session": s, "turn": r["turn"], "prompt": P, "measured": r["cached_tokens"], "replay": pred * B})
        elif kind == 1:
            total = -(-(P + 32) // B)
            blocks = held[rid]
            for i in range(len(blocks), total):
                if i == P // B and P % B == 0:   # first decode block starts right at the prompt end
                    k = ("x", s, i)
                else:
                    uniq[0] += 1; k = ("u", uniq[0])
                blocks.append(evict_alloc(t, s, i, k))
        else:
            for b in reversed(held.pop(rid)):
                ref[b] -= 1
                if ref[b] == 0:
                    free[b] = None
                    events.append((t, "release", b, s, -1))
    return events, per_req, missing[0]


summary = []
OUT.mkdir(parents=True, exist_ok=True)
dirs = sorted(set(glob.glob(str(ROOT / "main/*/rep_*/C")) + glob.glob(str(ROOT / "pilot-scan/pilot*/*/rep_1/C"))))
for d in dirs:
    d = Path(d)
    rows = [json.loads(l) for l in open(d / "requests.jsonl")]
    kv = int(re.search(r"GPU KV cache size: ([\d,]+)", (d / "kv_allocation.log").read_text()).group(1).replace(",", ""))
    n_blocks = kv // B - 1
    events, per_req, missing = replay(rows, n_blocks)
    exact = sum(p["measured"] == p["replay"] for p in per_req)
    tot_m = sum(p["measured"] for p in per_req); tot_r = sum(p["replay"] for p in per_req)
    worst = max(per_req, key=lambda p: abs(p["measured"] - p["replay"]))
    cohort = "main" if "/main/" in d.as_posix() else "pilot"
    budget, rep = int(d.parts[-3]), d.parts[-2]
    summary.append({"cohort": cohort, "gib": budget / 2**30, "rep": rep, "kv_tokens": kv, "requests": len(per_req),
                    "exact": exact, "measured_cached": tot_m, "replay_cached": tot_r,
                    "worst": worst, "events": len(events), "measured_hits_not_in_replay_cache": missing})
    name = f"{cohort}-{budget / 2**30:g}gib-{rep}"
    json.dump({"kv_tokens": kv, "n_blocks": n_blocks, "events": events, "requests": per_req}, open(OUT / f"{name}.json", "w"))
for s in summary:
    print(f"{s['cohort']:5} {s['gib']:4} {s['rep']:6} exact {s['exact']}/{s['requests']}  cached measured {s['measured_cached']:>7} replay {s['replay_cached']:>7}  events {s['events']}  INCONSISTENT {s['measured_hits_not_in_replay_cache']}  worst {s['worst']['session']}/{s['worst']['turn']} m={s['worst']['measured']} r={s['worst']['replay']}")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
