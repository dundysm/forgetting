// The brain: token-slot points on the MNI152 cortical surface, built by tools/brain.py.
// The file holds 72,816 points (the 10 GiB pool); a smaller pool takes the first n, which is a
// uniform random subset because the 16-point grains were shuffled.
export async function loadBrain(base) {
  const meta = await (await fetch(base + 'brain.json')).json();
  const buf = await (await fetch(base + 'brain.bin')).arrayBuffer();
  const n = meta.points;
  const f = new Float32Array(buf);
  return { n, meta, pos: f.subarray(0, 3 * n), nor: f.subarray(3 * n, 6 * n), depth: f.subarray(6 * n, 7 * n), hemi: f.subarray(7 * n, 8 * n) };
}
export function brainFace(brain, n) {
  if (n > brain.n) throw new Error(`pool has ${n} token slots; the brain file has ${brain.n}`);
  return { pos: brain.pos.subarray(0, 3 * n), nor: brain.nor.subarray(0, 3 * n), depth: brain.depth.subarray(0, n) };
}
