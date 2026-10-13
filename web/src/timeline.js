// Plays a packed run back in time and keeps the state of every KV block in flat arrays the
// renderer can upload straight to the GPU.
//
// Per block:
//   state:   0 empty (never written) · 1 written (just computed by a running request) ·
//            2 idle (cached, evictable) · 3 forgotten (evicted) · 4 recalled (prefix hit, in use)
//   session: who wrote it (-1 shared header, 0..15 sessions)
//   index:   block index within that session's prompt (position in its history)
//   tChange: when the state last changed (ms of run time), for transition animations
//   recalls: how many times it was recalled (prefix hit) since being written
export const KIND = { born: 0, recall: 1, release: 2, evict: 3, evict_scratch: 4 };
export const STATE = { empty: 0, written: 1, idle: 2, forgotten: 3, recalled: 4 };

export async function loadIndex(base) {
  return (await fetch(base + 'index.json')).json();
}

export async function loadRun(base, meta) {
  const buf = await (await fetch(base + meta.file)).arrayBuffer();
  const n = meta.n_events;
  let o = 0;
  const t = new Float32Array(buf, o, n); o += 4 * n;
  const block = new Uint16Array(buf, o, n); o += 2 * n;
  const index = new Int16Array(buf, o, n); o += 2 * n;
  const kind = new Uint8Array(buf, o, n); o += n;
  const session = new Int8Array(buf, o, n);
  return { meta, n, t, block, index, kind, session };
}

export class Timeline {
  constructor(run) {
    this.run = run;
    const nb = run.meta.n_blocks;
    this.state = new Uint8Array(nb);
    this.session = new Int8Array(nb).fill(-2);
    this.index = new Int16Array(nb);
    this.tChange = new Float32Array(nb).fill(-1e9);
    this.recalls = new Uint16Array(nb);
    this.dirty = true;
    this.reset();
  }
  reset() {
    this.state.fill(0); this.session.fill(-2); this.tChange.fill(-1e9); this.recalls.fill(0);
    this.cursor = 0; this.now = 0;
    this.stats = { born: 0, recall: 0, evict: 0, scratch: 0 };
    this.flashes = []; // recent notable events for effects: {t, kind, block, session}
    this.dirty = true;
  }
  get duration() { return this.run.meta.end_ms; }
  // Advance to run time `ms`. Seeking backwards replays from the start.
  seek(ms) {
    if (ms < this.now) this.reset();
    const R = this.run, S = this.state;
    while (this.cursor < R.n && R.t[this.cursor] <= ms) {
      const i = this.cursor++, b = R.block[i], k = R.kind[i], tt = R.t[i];
      if (k === KIND.born) {
        S[b] = STATE.written; this.session[b] = R.session[i]; this.index[b] = R.index[i];
        this.recalls[b] = 0; this.stats.born++;
      } else if (k === KIND.recall) {
        S[b] = STATE.recalled; this.recalls[b]++; this.stats.recall++;
      } else if (k === KIND.release) {
        S[b] = STATE.idle;
      } else { // evicted: the memory is gone; the slot is about to be reborn as something else
        S[b] = STATE.forgotten;
        if (k === KIND.evict) this.stats.evict++; else this.stats.scratch++;
      }
      this.tChange[b] = tt;
      // an eviction is recorded before the slot is reborn, so session/index here are the memory being lost
      if (k !== KIND.release) this.flashes.push({ t: tt, kind: k, block: b, session: this.session[b], index: this.index[b] });
    }
    this.now = ms;
    const cutoff = ms - 1500;
    if (this.flashes.length > 4000 || (this.flashes.length && this.flashes[0].t < cutoff)) {
      let j = 0; while (j < this.flashes.length && this.flashes[j].t < cutoff) j++;
      this.flashes.splice(0, j);
    }
    this.dirty = true;
  }
}
