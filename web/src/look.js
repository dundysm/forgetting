// How a token-star sits and shines. Shared GLSL for the cortex (stars) and the dust (ghosts).
//
// Every star is one token slot in the GPU's KV cache, fixed to a point on the cortex. What it does
// is the replayed vLLM run:
//   written  → tokens just computed: they drift in and settle over the request's measured TTFT
//   recalled → a prefix-cache hit: the memory is read back and blinks gold, then keeps an ember
//   idle     → remembered, waiting: a soft breathing glow
//   empty    → memory that was never used: dark
// and an eviction sparks the block's 16 stars and lifts some off the cortex as dust (see ghost()).
export const LAYOUT = /* glsl */`
  uniform float uTime;
  uniform float uFocus;                        // one session to follow (-9: none)
  uniform vec3 uWind;                          // screen up-right, in world space (set per frame)
  const vec3 ICE  = vec3(0.70, 0.82, 1.00);
  const vec3 FOCUS = vec3(0.35, 0.95, 1.00);
  const vec3 GOLD = vec3(1.00, 0.62, 0.30);
  vec3 tint(float s) {                         // each session is a faint shade of the same light
    float h = s / 16.0 * 6.2831;
    return ICE + 0.16 * vec3(cos(h), cos(h + 2.1), cos(h + 4.2));
  }
  vec3 swirl(vec3 p, float s) {                // cheap divergence-free-ish flow
    return vec3(sin(p.y * 2.3 + s * 5.1 + uTime * 0.4) + sin(p.z * 3.7 + s),
                sin(p.z * 2.9 + s * 3.3 + uTime * 0.3) + sin(p.x * 3.1 - s),
                sin(p.x * 2.7 + s * 7.7 + uTime * 0.5) + sin(p.y * 3.3 + s)) * 0.5;
  }
  // sculpting light: a soft key from the upper left plus a rim on the silhouette
  float lit(vec3 base, vec3 nrm) {
    vec3 V = normalize(cameraPosition - base);
    vec3 L = normalize(vec3(-0.5, 0.65, 0.6));   // grazing light: the folds need relief
    float key = max(dot(nrm, L), 0.0);
    float nv = dot(nrm, V);
    float rim = pow(1.0 - max(nv, 0.0), 3.0);
    float facing = 0.12 + 0.88 * smoothstep(-0.15, 0.45, nv);   // the far side of the brain recedes
    return (0.12 + 1.25 * key * key + 0.55 * rim) * facing;
  }
  // how much a flash may add here: edge-on and far-side points stack up in screen space, so a burst
  // of a few hundred blocks there would sum to a white blob through the brain
  float crowd(vec3 base, vec3 nrm) {
    return 0.3 + 0.7 * smoothstep(0.05, 0.6, dot(nrm, normalize(cameraPosition - base)));
  }
  void place(float state, float sess, float packed, vec3 base, vec3 nrm, float seed, float depth, float age,
             out vec3 p, out float size, out vec3 col, out float a) {
    p = base; col = ICE; a = 0.0;
    size = 1.35;                                // every KV state rests at the same size: lit area tracks slots
    float recalls = floor(packed / 10000.0);    // times recalled since written (capped at 3)
    float writeS = max(0.12, mod(packed, 10000.0) / 1000.0);   // measured time to write it (TTFT for prefill)
    float fold = 1.0 - 0.8 * smoothstep(0.2, 2.5, depth);
    float L = lit(base, nrm) * fold;
    bool focus = abs(sess - uFocus) < 0.5;
    vec3 held = vec3(0.62, 0.70, 0.78) * (0.5 + 0.85 * L);           // memory being kept
    // memory that has been recalled keeps a warm ember: warmth is earned by real reuse
    vec3 rest = mix(held, GOLD * 0.85 * (0.5 + 0.85 * L), 0.65 * min(recalls, 3.0) / 3.0);
    if (state < 0.5) {                          // empty: never written
      a = 0.0;
    } else if (state < 1.5) {                   // written: recomputed, settling over its real prefill time
      float e = smoothstep(0.0, 1.0, clamp(age / writeS, 0.0, 1.0));
      vec3 from = base + nrm * (0.30 + 0.25 * seed) - uWind * 0.25 + swirl(base * 2.0, seed * 9.0) * 0.12;
      p = mix(from, base, e);
      col = mix(vec3(0.78, 0.84, 1.0) * 0.85, mix(held, vec3(1.0), 0.15), e);
      a = mix(0.30 * crowd(base, nrm), 0.38, e);
      if (focus) { col = mix(FOCUS * 0.9, FOCUS * (0.5 + 0.7 * L), e); a = mix(0.45 * crowd(base, nrm), 0.62, e); size = 1.5; }
    } else if (state < 2.5) {                   // idle: remembered, waiting for its session
      p = base + nrm * 0.01 * sin(uTime * 1.2 + seed * 6.2831);
      col = rest; a = 0.38;
      if (focus) { col = FOCUS * (0.5 + 0.7 * L); a = 0.62; size = 1.5; }
    } else if (state < 3.5) {                   // forgotten (only for an instant; the dust carries it)
      a = 0.0;
    } else {                                    // recalled: read straight back from memory — a gold blink
      float f = exp(-age * 5.0);
      col = mix(mix(held, GOLD * 0.6 * (0.45 + 0.7 * L), 0.6), GOLD, f);
      a = 0.35 + 0.4 * f; size = 1.35 + 0.45 * f;   // a whole territory blinks at once: keep it points, not a blob
      if (focus) { col = mix(FOCUS * (0.5 + 0.7 * L), GOLD, 0.85 * f); a = 0.62 + 0.2 * f; size = 1.5 + 0.4 * f; }
    }
  }
  void ghost(vec3 base, vec3 nrm, float sess, float seed, float age, float lifeMs,
             out vec3 p, out float size, out vec3 col, out float a) {
    // forgetting: the block sparks as one, then a few of its points rise off the cortex as a plume
    float travel = age * 0.30 + age * age * 0.75;
    p = base + nrm * 0.12 * sqrt(age)
          + uWind * travel * (0.75 + 0.5 * seed)
          + swirl(base * 1.3 + uWind * travel, seed * 13.0) * age * 0.12;
    bool focus = abs(sess - uFocus) < 0.5;
    vec3 c0 = focus ? FOCUS * 1.25 : tint(max(sess, 0.0));
    col = mix(c0, vec3(0.38, 0.46, 0.85), smoothstep(0.1, 0.9, age));
    float spark = 1.4 * exp(-age * lifeMs / 140.0);                // ~140 ms, the whole block at once
    bool smoke = focus || seed < 0.25;                             // a quarter of each block becomes smoke
    // ~190 neighbouring blocks go per arriving request: the spark is a flicker, not a flare, and the
    // smoke fades in as it lifts off and spreads (born packed on the cortex, it would sum to a white slab)
    float lift = smoothstep(0.0, 0.15, age);
    a = (smoke ? (focus ? 0.45 : 0.28) * lift * pow(1.0 - age, 2.0) : 0.0) + spark * (focus ? 0.55 : 0.22) * crowd(base, nrm);
    size = (smoke && !focus ? 2.0 : 1.0) * (1.6 + 1.6 * age) + 1.2 * spark;
  }
`;
export const SHADE = /* glsl */`
  float glow(float r) { return exp(-r * r * 3.5); }
`;
