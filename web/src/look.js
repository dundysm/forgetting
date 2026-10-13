// How a token-star sits and shines. Shared GLSL for the face (stars) and the dust (ghosts).
//
// Every star is one token slot in the GPU's KV cache, fixed to a point on the face. What it does
// is the replayed vLLM run:
//   written  → tokens just computed: they stream in from the dark and settle into the skin
//   recalled → a prefix-cache hit: the memory is read back and flares gold
//   idle     → remembered, waiting: a soft breathing glow
//   empty    → memory that was never used: barely-there grey
// and an eviction tears the block's 16 stars off the face as dust (see ghost()).
export const LAYOUT = /* glsl */`
  uniform float uTime;
  uniform float uFocus;                        // one session to follow (-9: none)
  const vec3 ICE  = vec3(0.70, 0.82, 1.00);
  const vec3 FOCUS = vec3(0.35, 0.95, 1.00);
  const vec3 GOLD = vec3(1.00, 0.72, 0.36);
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
  void place(float state, float sess, float idx, vec3 base, vec3 nrm, float seed, float depth, float age,
             out vec3 p, out float size, out vec3 col, out float a) {
    p = base; size = 1.0; col = ICE; a = 0.0;
    // crowns of the gyri catch the light; the bottoms of the sulci fall into shadow
    float fold = 1.0 - 0.8 * smoothstep(0.2, 2.5, depth);
    float L = lit(base, nrm) * fold;
    if (state < 0.5) {                         // empty: never written
      col = vec3(0.32, 0.36, 0.44) * L; a = 0.06; size = 0.8;
    } else if (state < 1.5) {                  // written: stream in and settle
      // new memory condenses out of the air just above the skin
      float e = smoothstep(0.0, 1.0, clamp(age / 0.7, 0.0, 1.0));
      vec3 from = base + nrm * (0.22 + 0.25 * seed) + swirl(base * 2.0, seed * 9.0) * 0.12;
      p = mix(from, base, e);
      col = mix(vec3(0.50, 0.62, 0.95) * 0.6, mix(tint(sess), vec3(1.0), 0.35) * L, e);
      a = 0.08 + 0.6 * e; size = 1.0 + 0.6 * (1.0 - e);
      if (abs(sess - uFocus) < 0.5) { col = mix(FOCUS * 0.5, FOCUS * (0.5 + L), e); a = 0.15 + 0.8 * e; size = 1.2 + 0.8 * (1.0 - e); }
    } else if (state < 2.5) {                  // idle: remembered
      p = base + nrm * 0.012 * sin(uTime * 1.2 + seed * 6.2831);
      col = tint(sess) * L * 0.8; a = 0.3; size = 1.0;
      if (abs(sess - uFocus) < 0.5) { col = FOCUS * (0.35 + 0.8 * L); a = 0.75; size = 1.35; }
    } else if (state < 3.5) {                  // forgotten (only for an instant; the dust carries it)
      a = 0.0;
    } else {                                   // recalled: read back from memory
      float f = exp(-age * 2.2);
      p = base + nrm * 0.035 * f;
      col = mix(tint(sess) * L, GOLD * (0.5 + 0.6 * L), 0.25 + 0.75 * f); a = 0.55 + 0.35 * f; size = 1.1 + 1.4 * f;
      if (abs(sess - uFocus) < 0.5) { col = mix(FOCUS * (0.4 + 0.8 * L), GOLD * 1.2, f); a = 0.8; size = 1.35 + 1.4 * f; }
    }
  }
  void ghost(vec3 base, vec3 nrm, float sess, float seed, float age,
             out vec3 p, out float size, out vec3 col, out float a) {
    // the instant of forgetting is a spark on the cortex; then the memory rises off it as smoke
    vec3 wind = normalize(vec3(1.0, 0.85, 0.1));
    float travel = age * 0.8 + age * age * 2.4;
    p = base + nrm * 0.25 * sqrt(age)
          + wind * travel * (0.7 + 0.6 * seed)
          + swirl(base * 1.3 + wind * travel, seed * 13.0) * age * 0.35;
    bool focus = abs(sess - uFocus) < 0.5;
    vec3 c0 = focus ? FOCUS * 1.3 : vec3(1.0, 0.96, 0.9);
    col = mix(c0, vec3(0.40, 0.50, 0.92), smoothstep(0.05, 0.8, age));
    float spark = exp(-age * 28.0);
    a = (focus ? 0.95 : 0.3) * smoothstep(0.0, 0.03, age) * pow(1.0 - age, 1.2) + (focus ? 1.2 : 0.7) * spark;
    size = 1.2 + 0.6 * age + 2.2 * spark;
  }
`;
export const SHADE = /* glsl */`
  float glow(float r) { return exp(-r * r * 3.5); }
`;
