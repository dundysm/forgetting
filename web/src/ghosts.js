// Dust: when a block of memory is forgotten, its 16 token-stars tear off the face and drift away
// while the slot is rewritten. A ring buffer of particles the CPU fills on each eviction.
import * as THREE from 'three';

export function createGhosts(capacityBlocks, layoutGLSL, shadeGLSL) {
  const TOK = 16, n = capacityBlocks * TOK;
  const base = new Float32Array(n * 3), nrm = new Float32Array(n * 3);
  const born = new Float32Array(n).fill(-1e9), sess = new Float32Array(n), seed = new Float32Array(n);
  const g = new THREE.BufferGeometry();
  const A = {
    position: new THREE.BufferAttribute(base, 3), aNormal: new THREE.BufferAttribute(nrm, 3),
    aBorn: new THREE.BufferAttribute(born, 1), aSess: new THREE.BufferAttribute(sess, 1), aSeed: new THREE.BufferAttribute(seed, 1),
  };
  for (const [k, v] of Object.entries(A)) { v.setUsage(THREE.DynamicDrawUsage); g.setAttribute(k, v); }
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  const uniforms = { uNow: { value: 0 }, uTime: { value: 0 }, uFocus: { value: -9 }, uPixel: { value: 1 }, uSize: { value: 1 }, uLife: { value: 4000 } };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      uniform float uNow, uPixel, uSize, uLife;
      attribute float aBorn, aSess, aSeed; attribute vec3 aNormal;
      varying vec3 vColor; varying float vAlpha;
      ${layoutGLSL}
      void main() {
        float age = (uNow - aBorn) / uLife;
        vec3 p; float size; vec3 col; float a;
        ghost(position, aNormal, aSess, aSeed, clamp(age, 0.0, 1.0), p, size, col, a);
        if (age < 0.0 || age > 1.0) a = 0.0;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = size * uSize * uPixel * (7.0 / -mv.z);
        vColor = col; vAlpha = a;
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vColor; varying float vAlpha;
      ${shadeGLSL}
      void main() {
        float r = length(gl_PointCoord - 0.5) * 2.0;
        if (r > 1.0 || vAlpha <= 0.0) discard;
        gl_FragColor = vec4(vColor * glow(r) * vAlpha, 1.0);
      }`,
  });
  const points = new THREE.Points(g, mat);
  points.frustumCulled = false;
  let head = 0, lo = Infinity, hi = -1;
  // a forgotten block: its 16 stars (positions from the face) become dust at time t
  function spawn(t, session, face, slot) {
    const b = head * TOK, s = slot * TOK;
    base.set(face.pos.subarray(s * 3, (s + TOK) * 3), b * 3);
    nrm.set(face.nor.subarray(s * 3, (s + TOK) * 3), b * 3);
    for (let j = 0; j < TOK; j++) { born[b + j] = t; sess[b + j] = session; seed[b + j] = Math.random(); }
    lo = Math.min(lo, b); hi = Math.max(hi, b + TOK);
    head = (head + 1) % capacityBlocks;
  }
  function flush() {
    if (hi < 0) return;
    for (const [k, v] of Object.entries(A)) {
      const w = k === 'position' || k === 'aNormal' ? 3 : 1;
      v.clearUpdateRanges(); v.addUpdateRange(lo * w, (hi - lo) * w); v.needsUpdate = true;
    }
    lo = Infinity; hi = -1;
  }
  function clear() { born.fill(-1e9); A.aBorn.clearUpdateRanges(); A.aBorn.needsUpdate = true; head = 0; }
  return { points, uniforms, spawn, flush, clear };
}
