// The anatomy: a dense, dim point cloud of the MNI152 cortex that draws the folds. Static; it
// carries no KV data. The memory (stars.js) glows on top of it.
import * as THREE from 'three';

export async function createCortex(base, meta, layoutGLSL, shadeGLSL) {
  const n = meta.cortex_points, lo = meta.cortex_lo, hi = meta.cortex_hi;
  const buf = await (await fetch(base + 'cortex.bin')).arrayBuffer();
  const qp = new Uint16Array(buf, 0, 3 * n), qn = new Int8Array(buf, 6 * n, 3 * n), qd = new Uint8Array(buf, 9 * n, n);
  const pos = new Float32Array(3 * n), nor = new Float32Array(3 * n), dep = new Float32Array(n), seed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      pos[i * 3 + k] = lo[k] + (qp[i * 3 + k] / 65535) * (hi[k] - lo[k]);
      nor[i * 3 + k] = qn[i * 3 + k] / 127;
    }
    dep[i] = qd[i]; seed[i] = Math.abs(Math.sin(i * 78.233) * 43758.5453) % 1;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aNormal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('aDepth', new THREE.BufferAttribute(dep, 1));
  g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
  const uniforms = { uNow: { value: 0 }, uTime: { value: 0 }, uFocus: { value: -9 }, uPixel: { value: 1 }, uSize: { value: 1 }, uGain: { value: 2.3 } };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      uniform float uPixel, uSize, uGain;
      attribute vec3 aNormal; attribute float aDepth, aSeed;
      varying vec3 vColor; varying float vAlpha;
      ${layoutGLSL}
      void main() {
        float fold = 1.0 - 0.92 * smoothstep(0.3, 4.5, aDepth);   // crowns bright, sulci dark
        float L = lit(position, aNormal) * fold;
        vColor = vec3(0.36, 0.46, 0.72) * L * uGain;
        vAlpha = 0.16 + 0.04 * sin(uTime * 0.7 + aSeed * 6.2831);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = 0.85 * uSize * uPixel * (7.0 / -mv.z);
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vColor; varying float vAlpha;
      ${shadeGLSL}
      void main() {
        float r = length(gl_PointCoord - 0.5) * 2.0;
        if (r > 1.0) discard;
        gl_FragColor = vec4(vColor * glow(r) * vAlpha, 1.0);
      }`,
  });
  const points = new THREE.Points(g, mat);
  points.frustumCulled = false;
  return { points, uniforms };
}
