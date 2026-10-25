"""sdf.py - signed-distance modelling + narrow-band marching cubes for the G17 v4 (system python: numpy, scipy, skimage).

All units mm in the G frame (x gun-left, y muzzle, z up).  A part is a function f(X, Y, Z) -> signed distance (negative
inside) on numpy arrays.  mesh(f, lo, hi, h) evaluates it only in blocks near the surface and stitches per-block
marching-cubes results into one welded, outward-wound triangle mesh.  Rounded CSG (iq / hg_sdf formulas) gives real
fillets and edge rounds of a chosen radius, so polymer and steel edges get physical bevels instead of shading tricks."""
import math, time
import numpy as np
from skimage import measure

# ------------------------------------------------------------------ 2-D fields (rasters over (y, z))
class Field2D:
    """bilinear sampler of a raster defined on the (y, z) grid of profiles.py (row 0 = z1, col 0 = y0)"""
    def __init__(self, arr, y0, z1, res, outside=None):
        self.a = np.ascontiguousarray(arr, np.float32); self.y0, self.z1, self.res = y0, z1, res
        self.nz, self.ny = self.a.shape; self.outside = outside
    def __call__(self, Y, Z):
        fy = (Y - self.y0) / self.res - 0.5; fz = (self.z1 - Z) / self.res - 0.5
        iy = np.clip(np.floor(fy).astype(np.int64), 0, self.ny - 2); iz = np.clip(np.floor(fz).astype(np.int64), 0, self.nz - 2)
        ty = np.clip(fy - iy, 0, 1); tz = np.clip(fz - iz, 0, 1); a = self.a
        v = (a[iz, iy] * (1 - ty) * (1 - tz) + a[iz, iy + 1] * ty * (1 - tz) + a[iz + 1, iy] * (1 - ty) * tz + a[iz + 1, iy + 1] * ty * tz)
        if self.outside is not None:
            out = (fy < 0) | (fy > self.ny - 1) | (fz < 0) | (fz > self.nz - 1)
            v = np.where(out, self.outside, v)
        return v

def field_from_fn(fn, y0, y1, z0, z1, res):
    """rasterise a smooth (y, z) function once (cheap to sample later)"""
    ny, nz = int(round((y1 - y0) / res)), int(round((z1 - z0) / res))
    ys = y0 + (np.arange(ny) + 0.5) * res; zs = z1 - (np.arange(nz) + 0.5) * res
    YY, ZZ = np.meshgrid(ys, zs); return Field2D(fn(YY, ZZ), y0, z1, res)

# ------------------------------------------------------------------ helpers
def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0); return t * t * (3 - 2 * t)
def lerp(a, b, t): return a + (b - a) * t

# ------------------------------------------------------------------ CSG
def union(*ds):
    out = ds[0]
    for d in ds[1:]: out = np.minimum(out, d)
    return out
def inter(*ds):
    out = ds[0]
    for d in ds[1:]: out = np.maximum(out, d)
    return out
def diff(a, *bs):
    out = a
    for b in bs: out = np.maximum(out, -b)
    return out
def runion(a, b, r):
    """union with a concave fillet of radius r"""
    u = np.maximum(r - a, 0.0); v = np.maximum(r - b, 0.0)
    return np.maximum(r, np.minimum(a, b)) - np.sqrt(u * u + v * v)
def rinter(a, b, r):
    """intersection with the convex edge rounded to radius r"""
    u = np.maximum(r + a, 0.0); v = np.maximum(r + b, 0.0)
    return np.minimum(-r, np.maximum(a, b)) + np.sqrt(u * u + v * v)
def rdiff(a, b, r): return rinter(a, -b, r)
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return b + (a - b) * h - k * h * (1.0 - h)
def smax(a, b, k): return -smin(-a, -b, k)

# ------------------------------------------------------------------ primitives (exact or bound)
def box(X, Y, Z, lo, hi, r=0.0):
    """axis box [lo, hi] with all edges rounded r"""
    c = [(l + h) * 0.5 for l, h in zip(lo, hi)]; e = [(h - l) * 0.5 - r for l, h in zip(lo, hi)]
    qx = np.abs(X - c[0]) - e[0]; qy = np.abs(Y - c[1]) - e[1]; qz = np.abs(Z - c[2]) - e[2]
    o = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return o + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0) - r
def slab(V, lo, hi): return np.maximum(lo - V, V - hi)
def cyl(X, Y, Z, axis, c, rad, a0, a1, r=0.0):
    """cylinder along axis 'x'|'y'|'z' through c = the two other coordinates, from a0 to a1; edges rounded r"""
    if axis == 'y': u, v, w = X - c[0], Z - c[1], Y
    elif axis == 'x': u, v, w = Y - c[0], Z - c[1], X
    else: u, v, w = X - c[0], Y - c[1], Z
    dr = np.sqrt(u * u + v * v) - (rad - r); da = np.abs(w - (a0 + a1) * 0.5) - ((a1 - a0) * 0.5 - r)
    return np.minimum(np.maximum(dr, da), 0.0) + np.sqrt(np.maximum(dr, 0) ** 2 + np.maximum(da, 0) ** 2) - r
def capsule2(U, V, p0, p1, rad):
    """2-D capsule (stadium) distance in a plane"""
    pa_u, pa_v = U - p0[0], V - p0[1]; ba = (p1[0] - p0[0], p1[1] - p0[1]); bb = ba[0] ** 2 + ba[1] ** 2
    t = np.clip((pa_u * ba[0] + pa_v * ba[1]) / max(bb, 1e-12), 0, 1)
    return np.sqrt((pa_u - ba[0] * t) ** 2 + (pa_v - ba[1] * t) ** 2) - rad
def rect2(U, V, lo, hi, r=0.0):
    cu, cv = (lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5; eu, ev = (hi[0] - lo[0]) * 0.5 - r, (hi[1] - lo[1]) * 0.5 - r
    qu = np.abs(U - cu) - eu; qv = np.abs(V - cv) - ev
    return np.sqrt(np.maximum(qu, 0) ** 2 + np.maximum(qv, 0) ** 2) + np.minimum(np.maximum(qu, qv), 0.0) - r
def extrude(d2, W, w0, w1, r=0.0):
    """2-D distance d2 extruded along W from w0 to w1 with edges rounded r"""
    dw = slab(W, w0, w1)
    return rinter(d2, dw, r) if r > 0 else np.maximum(d2, dw)
def rot2(U, V, ang_deg, c=(0.0, 0.0)):
    a = math.radians(ang_deg); ca, sa = math.cos(a), math.sin(a); u, v = U - c[0], V - c[1]
    return c[0] + u * ca + v * sa, c[1] - u * sa + v * ca

# ------------------------------------------------------------------ narrow-band marching cubes
def mesh(f, lo, hi, h, block=24, lip=1.7, verbose=True, name=''):
    """triangle mesh of {f < 0} inside the box [lo, hi] at voxel size h.  Returns (V float64 (n,3), F int64 (m,3))."""
    t0 = time.time(); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    nb = np.ceil((n - 1) / block).astype(int)
    # coarse test at block centres
    bi = np.stack(np.meshgrid(*[np.arange(k) for k in nb], indexing='ij'), -1).reshape(-1, 3)
    cen = lo + (bi * block + block * 0.5) * h
    dc = f(cen[:, 0], cen[:, 1], cen[:, 2])
    half = 0.5 * math.sqrt(3) * block * h
    keep = bi[np.abs(dc) < half * lip + 2 * h]
    Vs, Fs = [], []; nv = 0
    ar = np.arange(block + 1)
    for b in keep:
        i0 = b * block; idx = [i0[k] + ar for k in range(3)]
        idx = [ix[ix < n[k]] for k, ix in enumerate(idx)]
        if min(len(ix) for ix in idx) < 2: continue
        gx, gy, gz = np.meshgrid(lo[0] + idx[0] * h, lo[1] + idx[1] * h, lo[2] + idx[2] * h, indexing='ij')
        vol = f(gx, gy, gz).astype(np.float32)
        if vol.min() >= 0 or vol.max() <= 0: continue
        try:
            v, fa, _, _ = measure.marching_cubes(vol, 0.0, spacing=(h, h, h), allow_degenerate=False, method='lewiner')
        except (ValueError, RuntimeError):
            continue
        v = v + (lo + i0 * h)
        Vs.append(v); Fs.append(fa + nv); nv += len(v)
    if not Vs: return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V = np.concatenate(Vs); F = np.concatenate(Fs).astype(np.int64)
    # weld the shared block-face vertices (same samples -> same interpolation up to float noise)
    key = np.round(V / (h * 1e-3)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]; F = inv.reshape(-1)[F]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]); F = F[ok]
    # outward winding: skimage gives normals along the gradient descent for level sets; check the signed volume
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
    if vol < 0: F = F[:, ::-1]; vol = -vol
    if verbose:
        print('[sdf] mesh %-14s h %.3f blocks %d/%d  verts %d tris %d  vol %.0f mm3  %.1fs' % (name, h, len(keep), len(bi), len(V), len(F), vol, time.time() - t0), flush=True)
    return V, F

def save(path, parts):
    """parts: {name: (V, F, attrs dict of per-vertex arrays)} -> npz"""
    out = {}
    for k, (V, F, at) in parts.items():
        out[k + '__V'] = V.astype(np.float32); out[k + '__F'] = F.astype(np.int32)
        for a, arr in (at or {}).items(): out[k + '__a_' + a] = np.asarray(arr, np.float32)
    np.savez_compressed(path, **out)
