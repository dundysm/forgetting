import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { loadIndex, loadRun, Timeline, KIND } from './timeline.js';
import { createStars } from './stars.js';
import { createGhosts } from './ghosts.js';
import { LAYOUT, SHADE } from './look.js';
import { loadBrain, brainFace } from './brain.js';
import { createCortex } from './cortex.js';

const DATA = 'data/';
const CAPTURE = new URLSearchParams(location.search).has('capture');
const $ = id => document.getElementById(id);

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
camera.position.set(3.9, 1.5, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.35; controls.target.set(0, 0.0, 0);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.4, 0.12, 0.42);   // only flares bloom
composer.addPass(bloom);
// hue-preserving clamp (stacked additive gold stays gold, not white) and edges fading to true black
composer.addPass(new ShaderPass({
  uniforms: { tDiffuse: { value: null } },
  vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `uniform sampler2D tDiffuse; varying vec2 vUv;
    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      c.rgb /= max(1.0, max(c.r, max(c.g, c.b)));
      c.rgb *= smoothstep(0.0, 0.06, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)));
      gl_FragColor = c;
    }`,
}));
composer.addPass(new OutputPass());
// keep the whole brain in frame on any screen: back the camera off on tall/narrow windows
function fitCamera() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  if (CAPTURE) return;
  const want = Math.hypot(3.9, 1.5, 2.6) * Math.max(1, 1.25 / camera.aspect);
  const d = camera.position.clone().sub(controls.target);
  camera.position.copy(controls.target).add(d.setLength(want));
}
addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight); fitCamera();
});

const index = await loadIndex(DATA);
const brain = await loadBrain('assets/');
const cortex = await createCortex('assets/', brain.meta, LAYOUT, SHADE);
scene.add(cortex.points);
let face = null;
let stars = null, ghosts = null, tl = null, runMeta = null;
const SPEED = { v: 1 };
let runTime = 0, playing = true;

async function show(gib) {
  runMeta = index.runs.find(r => r.gib === gib);
  const run = await loadRun(DATA, runMeta);
  if (stars) { scene.remove(stars.points); stars.points.geometry.dispose(); }
  if (!ghosts) { ghosts = createGhosts(1600, LAYOUT, SHADE); scene.add(ghosts.points); }
  ghosts.clear();
  face = brainFace(brain, runMeta.n_blocks * 16);   // one point of cortex per token slot of this pool
  stars = createStars(runMeta.n_blocks, face, LAYOUT, SHADE);
  scene.add(stars.points);
  tl = new Timeline(run);
  runTime = 0;
  document.querySelectorAll('#budgets button').forEach(b => b.classList.toggle('on', b.dataset.gib === gib));
}
$('budgets').innerHTML = index.runs.map(r => `<button data-gib="${r.gib}">${r.gib} GiB</button>`).join('');
$('budgets').onclick = e => { if (e.target.dataset.gib) show(e.target.dataset.gib); };
function setPlaying(v) { playing = v; $('play').textContent = v ? 'pause' : 'play'; }
$('play').onclick = () => setPlaying(!playing);
addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); } });
// follow one agent: cycles 5 → 6 → … → 15 → 0 → … → off
let focus = 5;
function setFocus(f) {
  focus = f;
  $('follow').textContent = f < 0 ? 'follow an agent' : `follow agent ${f}`;
  $('follow').classList.toggle('on', f >= 0);
}
$('follow').onclick = () => setFocus(focus < 0 ? 5 : (focus === 4 ? -9 : (focus + 1) % 16));
let scrubbing = false;
$('time').addEventListener('input', e => { scrubbing = true; advance(+e.target.value / 1000 * tl.duration); });
$('time').addEventListener('change', () => { scrubbing = false; });

await show(new URLSearchParams(location.search).get('gib') || '4.5');
fitCamera();
window.__f = { get tl() { return tl; }, camera, controls, SPEED, bloom, get stars() { return stars; }, get ghosts() { return ghosts; }, cortex };

// advance the replay to run time `ms` (forward only, except a reset at the loop point)
function advance(ms) {
  if (ms < runTime) { tl.reset(); ghosts.clear(); }
  runTime = ms;
  const startCursor = tl.cursor;
  tl.seek(Math.min(runTime, tl.duration));
  // memory forgotten this frame becomes dust: an evict event carries the lost block's owner and slot
  const R = tl.run;
  for (let i = startCursor; i < tl.cursor; i++) if (R.kind[i] === KIND.evict) ghosts.spawn(R.t[i], R.session[i], face, R.block[i]);
  ghosts.flush();
  stars.upload(tl);
}
function draw(clockSec) {
  const now_ms = Math.min(runTime, tl.duration + 3500);
  // dust always rises toward the screen's upper right, whatever the camera does
  camera.updateMatrixWorld();
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const wind = right.multiplyScalar(0.6).add(up).normalize();
  for (const u of [stars.uniforms, ghosts.uniforms, cortex.uniforms]) u.uWind.value.copy(wind);
  for (const u of [stars.uniforms, ghosts.uniforms, cortex.uniforms]) { u.uNow.value = now_ms; u.uTime.value = clockSec; u.uPixel.value = renderer.getPixelRatio(); }
  for (const u of [stars.uniforms, ghosts.uniforms, cortex.uniforms]) if (!CAPTURE) u.uFocus.value = focus;
  const s = tl.stats;
  $('hud').innerHTML = `${(Math.min(runTime, tl.duration) / 1000).toFixed(1)} s · remembered <b>${s.recall.toLocaleString()}</b> · <span class="f">forgotten <b>${s.evict.toLocaleString()}</b></span>`;
  if (!scrubbing) $('time').value = Math.round(Math.min(1, runTime / tl.duration) * 1000);
  composer.render();
}

if (CAPTURE) {
  // deterministic frames for video: the page renders only when asked, on the run's own clock
  for (const id of ['ui', 'title']) document.getElementById(id).style.display = 'none';
  controls.enabled = false; controls.autoRotate = false;
  const tag = document.getElementById('tag');
  tag.style.display = 'block';
  let base = null;   // the forgotten counter counts from the clip's first frame
  window.__f.setBaseline = () => { base = tl.stats.evict; };
  window.__f.renderAt = (ms, orbit = 0, dist = 1, focus = -9, height = 1.5) => {
    advance(ms);
    for (const u of [stars.uniforms, ghosts.uniforms, cortex.uniforms]) u.uFocus.value = focus;
    const r = Math.hypot(3.9, 2.6) * dist, a0 = Math.atan2(2.6, 3.9) + orbit;
    camera.position.set(Math.cos(a0) * r, height * dist, Math.sin(a0) * r); camera.lookAt(0, 0.05, 0);
    draw(ms / 1000);
    const lost = base === null ? 0 : tl.stats.evict - base;
    tag.innerHTML = `kv cache · ${runMeta.gib} GiB<span>forgotten ${lost.toLocaleString()}</span>`;
    return { forgotten: tl.stats.evict, remembered: tl.stats.recall };
  };
} else {
  let prev = performance.now();
  const frame = now => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    if (playing && tl && !scrubbing) {
      let next = runTime + dt * 1000 * SPEED.v;
      if (next > tl.duration + 3000) next = 0;
      advance(next);
    }
    controls.update();
    draw(now / 1000);
  };
  requestAnimationFrame(frame);
}
