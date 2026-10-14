# forgetting

**An AI agent's memory, replayed inside a real human brain.**

Every point of light is one token slot of vLLM's KV cache (the GPU memory an LLM server keeps of each conversation), replaying a real benchmark run event by event. The brain is a real person's cortex, from MRI.

- **4.5 GiB of memory:** the brain keeps forgetting each agent's history just before that agent comes back for it. 36,481 blocks of reusable memory are lost in one 80-second run, and every turn recomputes its history from scratch.
- **10 GiB:** it remembers. 98.9% of each returning prompt is recalled from cache, time to first token drops from 581 ms to 67 ms, and nothing is forgotten.

Interactive piece: `web/index.html`. How it was made, with the encoding table and validation: `web/about.html`.

## Where the data comes from

The benchmark is [dundysm/inference-under-pressure](https://github.com/dundysm/inference-under-pressure):
- vLLM 0.31.0, Qwen3-4B-Instruct, one A40.
- 16 scripted agent sessions × 12 turns, with 0.5 s tool pauses and 4 requests in flight.
- Prefix caching on; only the KV budget varies (2.5–10 GiB).
- Pre-registered 5-repetition cohort at 4.5 and 10 GiB.

```bash
# 1. replay vLLM v1's block manager over the measured runs (needs the benchmark checkout)
python tools/replay.py ../inference-under-pressure/results/2026-10-12-runpod-a40 build/replay
# 2. pack one run per budget for the browser
python tools/pack.py build/replay web/data
# 3. build the brain (TemplateFlow MNIColin27 T1w + brain mask in build/colin)
python tools/brain.py build/colin web/assets 72816 450000
# 4. look at it
python -m http.server 8780 --directory web
```

### The replay, and how it was checked

vLLM reports how many prompt tokens each request reused from cache. It doesn't log which block was evicted when. `tools/replay.py` replays the block manager over the measured timeline:
- Each request's real start, first-token and end times are used.
- Prompt blocks are allocated at start, and decode blocks at the first token.
- At the end, a request's blocks are freed in reverse order to the tail of an LRU queue; new allocations evict from its head.
- Which blocks were reused comes from the measured cached-token counts; the replay decides when everything else is evicted.

| Check | Result |
|---|---|
| Measured reuses still present in the replayed cache | all of them; **0 inconsistencies** in 15 runs |
| Pure replay prediction = measured cached tokens, 2.5–6 GiB | **192/192** requests in every run |
| Same, 8 and 10 GiB | 168–171/192 |

At 8 and 10 GiB, each miss is one 16-token block: the one holding the end of the previous prompt plus the model's reply. In those requests, the next scripted prompt happened to start with the same tokens the model had generated.

| KV budget | Pool (tokens) | Resumed reuse | Forgotten blocks / run |
|---|---|---|---|
| 2.5 GiB | 18,192 | 0.5% | 37,384 |
| 3 GiB | 21,840 | 0.5% | 37,158 |
| 4.5 GiB | 32,768 | 0.5% | 36,481 |
| 6 GiB | 43,680 | 0.5% | 35,805 |
| 8 GiB | 58,240 | 98.9% | 77 |
| 10 GiB | 72,816 | 98.9% | 0 |

A forgotten block is lost a median **2.35 s** before its session returns for it (n = 34,958, at 4.5 GiB).

### The brain

`tools/brain.py` builds the brain from Colin27:
1. Segment the T1 by multi-Otsu.
2. Upsample to 0.5 mm.
3. Extract the pial surface with marching cubes: 1.2M vertices, 2,029 cm² of cortex.
4. Compute sulcal depth (distance from the closed brain envelope) for shading.

From that surface it samples:
- 450,000 anatomy points within 7 mm of the envelope.
- One point per KV token slot on the visible cortex, grouped 16 to a block grain. The file stores the grains shuffled, so a smaller pool takes a uniform subset of the cortex.

In the browser (`web/src/brain.js`), the pool's grains are chained nearest-neighbour across the surface, so block *b* sits next to block *b*+1. vLLM's free queue tends to hand out runs of neighbouring blocks, so a session's memory lands as one territory and visibly fragments as the LRU recycles slots.

Where a slot sits on the brain is artistic. When it lights, how fast it settles (the request's measured time to first token), when it goes gold, how warm it stays (how often it was recalled) and when it turns to dust is data.

## Credits

- **Brain:** Colin27 (Holmes et al., 1998), MNI / McConnell Brain Imaging Centre, via [TemplateFlow](https://www.templateflow.org). © 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University; permissive licence in `web/assets/MNIColin27_LICENSE.txt`.
- **Rendering:** three.js.
- **Engine measured:** vLLM.
