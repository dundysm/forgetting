// The brain: token-slot points on the Colin27 cortical surface, built by tools/brain.py.
// The file holds 72,816 points (the 10 GiB pool); a smaller pool takes the first n, which is a
// uniform random subset because the 16-point grains were shuffled.
export async function loadBrain(base) {
  const meta = await (await fetch(base + 'brain.json')).json();
  const buf = await (await fetch(base + 'brain.bin')).arrayBuffer();
  const n = meta.points;
  const f = new Float32Array(buf);
  return { n, meta, pos: f.subarray(0, 3 * n), nor: f.subarray(3 * n, 6 * n), depth: f.subarray(6 * n, 7 * n), hemi: f.subarray(7 * n, 8 * n) };
}
// A pool of n token slots takes the first n points (a uniform random subset of the cortex: the
// 16-point grains were shuffled at build time), then chains the grains nearest-neighbour by
// nearest-neighbour across the surface, so block id b and b+1 sit side by side on the cortex.
// vLLM hands out neighbouring block ids together, so a session's memory forms one organic territory;
// when LRU recycles slots, that territory visibly fragments. (A Z-order curve did this too, but
// cut axis-aligned rectangles into the brain.)
export function brainFace(brain, n) {
  if (n > brain.n) throw new Error(`pool has ${n} token slots; the brain file has ${brain.n}`);
  const TOK = 16, nb = n / TOK;
  const cx = new Float32Array(nb), cy = new Float32Array(nb), cz = new Float32Array(nb);
  for (let g = 0; g < nb; g++) {
    for (let j = 0; j < TOK; j++) { const o = (g * TOK + j) * 3; cx[g] += brain.pos[o]; cy[g] += brain.pos[o + 1]; cz[g] += brain.pos[o + 2]; }
    cx[g] /= TOK; cy[g] /= TOK; cz[g] /= TOK;
  }
  // greedy chain: start at the front-most grain, always step to the nearest unvisited one
  const used = new Uint8Array(nb), ord = new Uint32Array(nb);
  let cur = 0;
  for (let g = 1; g < nb; g++) if (cz[g] > cz[cur]) cur = g;
  for (let k = 0; k < nb; k++) {
    ord[k] = cur; used[cur] = 1;
    let best = -1, bd = Infinity;
    for (let g = 0; g < nb; g++) {
      if (used[g]) continue;
      const dx = cx[g] - cx[cur], dy = cy[g] - cy[cur], dz = cz[g] - cz[cur], d = dx * dx + dy * dy + dz * dz;
      if (d < bd) { bd = d; best = g; }
    }
    cur = best;
  }
  const pos = new Float32Array(3 * n), nor = new Float32Array(3 * n), depth = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const g = ord[b];
    pos.set(brain.pos.subarray(g * TOK * 3, (g + 1) * TOK * 3), b * TOK * 3);
    nor.set(brain.nor.subarray(g * TOK * 3, (g + 1) * TOK * 3), b * TOK * 3);
    depth.set(brain.depth.subarray(g * TOK, (g + 1) * TOK), b * TOK);
  }
  return { pos, nor, depth };
}
