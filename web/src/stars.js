// One point of light per KV-cache token slot, fixed to the face. Block state lives in a float
// texture (state, session, index, tChange) the vertex shader samples, so the CPU uploads one
// small texture per frame however many events happen.
import * as THREE from 'three';

export function createStars(nBlocks, face, layoutGLSL, shadeGLSL) {
  const TOK = 16, n = nBlocks * TOK;
  const W = 256, H = Math.ceil(nBlocks / W);
  const tex = new THREE.DataTexture(new Float32Array(W * H * 4), W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  const slot = new Float32Array(n), seed = new Float32Array(n);
  for (let k = 0; k < n; k++) { slot[k] = Math.floor(k / TOK); seed[k] = (Math.sin(k * 12.9898) * 43758.5453) % 1; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(face.pos, 3));
  g.setAttribute('aNormal', new THREE.BufferAttribute(face.nor, 3));
  g.setAttribute('aDepth', new THREE.BufferAttribute(face.depth || new Float32Array(n), 1));
  g.setAttribute('aSlot', new THREE.BufferAttribute(slot, 1));
  g.setAttribute('aSeed', new THREE.BufferAttribute(seed.map(Math.abs), 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  const uniforms = {
    uBlocks: { value: tex }, uTexW: { value: W }, uTexH: { value: H },
    uNow: { value: 0 }, uTime: { value: 0 }, uFocus: { value: -9 }, uPixel: { value: 1 }, uSize: { value: 1.0 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      uniform sampler2D uBlocks; uniform float uTexW, uTexH, uNow, uPixel, uSize;
      attribute float aSlot, aSeed, aDepth; attribute vec3 aNormal;
      varying vec3 vColor; varying float vAlpha;
      ${layoutGLSL}
      void main() {
        vec2 uv = vec2((mod(aSlot, uTexW) + 0.5) / uTexW, (floor(aSlot / uTexW) + 0.5) / uTexH);
        vec4 B = texture2D(uBlocks, uv);            // state, session, index, tChange
        vec3 p; float size; vec3 col; float a;
        place(B.x, B.y, B.z, position, aNormal, aSeed, aDepth, (uNow - B.w) / 1000.0, p, size, col, a);
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

  function upload(tl) {
    const d = tex.image.data;
    for (let b = 0; b < nBlocks; b++) {
      const o = b * 4;
      d[o] = tl.state[b]; d[o + 1] = tl.session[b]; d[o + 2] = tl.index[b]; d[o + 3] = tl.tChange[b];
    }
    tex.needsUpdate = true;
  }
  return { points, uniforms, upload, count: n };
}
