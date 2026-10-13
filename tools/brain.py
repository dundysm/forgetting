"""Build the brain the KV cache lives in: points on the real cortical surface of the MNI152 template.

Usage: python tools/brain.py <dir with MNI152NLin2009cAsym res-01 probsegs> <out dir> [n_points]

- Surface: grey + white matter probability (the pial surface), lightly smoothed, marching cubes at
  the 0.5 level on the 1 mm template; every gyrus and sulcus is the template's own anatomy.
- Sulcal depth: distance (mm) from each point to the brain's closed envelope (the mask closed with a
  ~9 mm ball); 0 on the crowns of gyri, ~20 mm at the bottom of the deepest sulci. Drives shading.
- Points: area-uniform samples, grouped 16 at a time along a Z-order curve (one KV block = one grain
  of cortex), then grains shuffled so consecutive memory slots land all over the brain. A pool with
  fewer slots takes the first n grains, which is a uniform random subset.
- Frame: three.js y-up; x = right, y = superior, z = anterior. Scaled so the brain is 3 units long.

Template: ICBM 152 Nonlinear Asymmetric 2009c (Fonov et al. 2011), via TemplateFlow.
Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
Institute, McGill University (permissive licence; see assets/MNI152_LICENSE.txt).
"""
import json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 72816
TOK = 16
OUT.mkdir(parents=True, exist_ok=True)

COLIN = (SRC / "T1w.nii.gz").exists()
if COLIN:
    # Colin27: one person scanned 27 times; segment the pial surface from the T1 directly.
    from skimage.filters import threshold_multiotsu
    t1 = nib.load(SRC / "T1w.nii.gz")
    mask = nib.load(SRC / "desc-brain_mask.nii.gz").get_fdata() > 0.5
    aff1 = t1.affine
    vol = t1.get_fdata(dtype=np.float32) * ndimage.binary_dilation(mask, iterations=1)
    level = float(threshold_multiotsu(vol[mask], classes=3)[0])          # CSF | grey matter
    solid = (vol > level) & mask
    up = 2                                                                # extract at 0.5 mm
    fine = ndimage.zoom(vol, up, order=3)
    fine = ndimage.gaussian_filter(fine, 0.6)
    verts, faces, normals, _ = measure.marching_cubes(fine, level=level, step_size=1, allow_degenerate=False)
    verts = verts / up                                                    # back to 1 mm voxel coordinates
    print(f"Colin27 T1: CSF/GM threshold {level:.0f}")
else:
    gm = nib.load(SRC / "label-GM_probseg.nii.gz")
    wm = nib.load(SRC / "label-WM_probseg.nii.gz")
    aff1 = gm.affine
    tissue = gm.get_fdata(dtype=np.float32) + wm.get_fdata(dtype=np.float32)
    tissue = ndimage.gaussian_filter(tissue, 0.55)
    solid = tissue > 0.5
    verts, faces, normals, _ = measure.marching_cubes(tissue, level=0.5, step_size=1, allow_degenerate=False)
aff = aff1
mesh = trimesh.Trimesh(verts, faces, process=True)
parts = mesh.split(only_watertight=False)
mesh = max(parts, key=lambda m: m.area)
mesh.fix_normals()
print(f"surface: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, area {mesh.area / 100:.0f} cm^2")

# sulcal depth: distance inside the closed envelope (1 mm grid)
ball = ndimage.generate_binary_structure(3, 1)
envelope = ndimage.binary_closing(solid, structure=ball, iterations=9)
envelope = ndimage.binary_fill_holes(envelope)
depth_vol = ndimage.distance_transform_edt(envelope)

rng = np.random.default_rng(7)
# Token slots go on the visible cortex (gyral crowns and upper walls); most real cortex is buried
# in the folds, and points there would only blur the surface.
TOKEN_MAX_DEPTH = 2.5
def sample_shallow(n, max_depth, seed):
    got_p, got_f, got_d, k = [], [], [], 0
    while sum(len(x) for x in got_p) < n:
        p_, f_ = trimesh.sample.sample_surface(mesh, n * 4, seed=seed + k)
        d_ = ndimage.map_coordinates(depth_vol, p_.T, order=1)
        keep = d_ <= max_depth
        got_p.append(p_[keep]); got_f.append(f_[keep]); got_d.append(d_[keep]); k += 1
    return np.concatenate(got_p)[:n], np.concatenate(got_f)[:n], np.concatenate(got_d)[:n]
pts, fidx, depth = sample_shallow(N, TOKEN_MAX_DEPTH, 7)
nrm = mesh.face_normals[fidx]
# world mm (RAS) for a sanity check, then to the display frame
ras = nib.affines.apply_affine(aff, pts)
print("RAS extent mm:", np.round(ras.min(0), 1), np.round(ras.max(0), 1))
print("sulcal depth mm: p50 %.1f p90 %.1f max %.1f" % tuple(np.percentile(depth, [50, 90, 100])))
nras = nrm @ aff[:3, :3].T
nras /= np.linalg.norm(nras, axis=1, keepdims=True)
disp = ras[:, [0, 2, 1]]                         # x right, y superior, z anterior
dnrm = nras[:, [0, 2, 1]]
c = (disp.min(0) + disp.max(0)) / 2
s = 3.0 / (disp.max(0) - disp.min(0)).max()
disp = (disp - c) * s

# grains of 16 along a Z-order curve, then shuffle the grains
q = np.clip(((disp + 2) / 4 * 1023).astype(np.int64), 0, 1023)
def spread(v):
    v = v & 0x3ff
    v = (v | v << 16) & 0x30000ff; v = (v | v << 8) & 0x300f00f
    v = (v | v << 4) & 0x30c30c3; v = (v | v << 2) & 0x9249249
    return v
key = spread(q[:, 0]) + spread(q[:, 1]) * 2 + spread(q[:, 2]) * 4
order = np.argsort(key, kind="stable")
nb = N // TOK
grains = rng.permutation(nb)
idx = (order[: nb * TOK].reshape(nb, TOK))[grains].reshape(-1)
disp, dnrm, depth = disp[idx], dnrm[idx], depth[idx]
hemi = np.sign(disp[:, 0])

blob = np.concatenate([disp.astype(np.float32).ravel(), dnrm.astype(np.float32).ravel(),
                       depth.astype(np.float32), hemi.astype(np.float32)])
(OUT / "brain.bin").write_bytes(blob.tobytes())
# The anatomy layer: a dense, quantized point cloud of the cortex to draw the folds, independent
# of the KV data. uint16 xyz over the display box, int8 normals, uint8 sulcal depth (mm, clipped).
NC = int(sys.argv[4]) if len(sys.argv) > 4 else 320000
cp, cf, cd = sample_shallow(NC, 7.0, 11)       # crowns plus enough of the walls to shade the folds
cn = (mesh.face_normals[cf] @ aff[:3, :3].T)
cn /= np.linalg.norm(cn, axis=1, keepdims=True)
cr = nib.affines.apply_affine(aff, cp)[:, [0, 2, 1]]
cr = (cr - c) * s
cnrm = cn[:, [0, 2, 1]]
lo, hi = cr.min(0), cr.max(0)
qpos = np.round((cr - lo) / (hi - lo) * 65535).astype(np.uint16)
qn = np.round(cnrm * 127).astype(np.int8)
qd = np.clip(np.round(cd), 0, 255).astype(np.uint8)
(OUT / "cortex.bin").write_bytes(qpos.tobytes() + qn.tobytes() + qd.tobytes())
print("cortex", NC, "points", (qpos.nbytes + qn.nbytes + qd.nbytes) // 1024, "KiB")
json.dump({"points": int(nb * TOK), "cortex_points": NC, "cortex_lo": lo.tolist(), "cortex_hi": hi.tolist(), "layout": "pos[3n] nrm[3n] depth_mm[n] hemi[n] float32",
           "scale_units_per_mm": float(s), "source": "MNI152NLin2009cAsym res-01 GM+WM probseg, TemplateFlow",
           "depth_p90_mm": float(np.percentile(depth, 90))}, open(OUT / "brain.json", "w"), indent=1)
print("wrote", OUT / "brain.bin", len(blob) * 4 // 1024, "KiB")
