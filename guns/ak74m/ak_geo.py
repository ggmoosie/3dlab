"""ak_geo.py - shared SDF helpers for the AK-74M modules (B frame, mm). Pure field maths, no IO, no Blender."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import ak_ref as R
import ak_spec as SP

HW = SP.HW
bp = R.bp
MAT = dict(steel=0, poly=1, park=2)


def by(x):
    """photo column -> B y (mm)"""
    return float(R.px2b(x, R.ROW_BORE)[0])


def bz(row):
    """photo row -> B z (mm) at the muzzle column"""
    return float(R.px2b(R.X_MUZ, row)[1])


def xr(name):
    p = R.PARTS[name]
    return min(q[0] for q in p), max(q[0] for q in p)


def lathe_photo(Y, Z, X, name, zc=0.0, rmax=99.0, smooth=1):
    """round part: radius per photo column of the part's silhouette (about axis z = zc)"""
    cr = R.column_radius(*xr(name), name)
    ys = np.array([c[0] for c in cr]); rs = np.array([min(c[1], rmax) for c in cr])
    if smooth > 1:
        rs = np.convolve(np.pad(rs, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    o = np.argsort(ys); ys, rs = ys[o], rs[o]
    r = np.interp(Y, ys, rs, left=-1, right=-1)
    return np.maximum(np.hypot(X, Z - zc) - r, np.maximum(ys[0] - Y, Y - ys[-1]))


def lathe(Y, R_, ys, rs, axis_z=0.0, X=None, Z=None):
    """round part from a (y, r) table"""
    o = np.argsort(ys); ys, rs = np.asarray(ys, float)[o], np.asarray(rs, float)[o]
    r = np.interp(Y, ys, rs)
    return np.maximum(np.hypot(X, Z - axis_z) - r, np.maximum(ys[0] - Y, Y - ys[-1]))


def slab(P2, X, hw, r=0.8):
    return S.rinter(P2, np.abs(X) - hw, r)


def yband(Y, y0, y1):
    """inside where y0 <= Y <= y1 (any order)"""
    a, b = min(y0, y1), max(y0, y1)
    return np.maximum(a - Y, Y - b)


def poly2(Y, Z, pts_px, r=0.0):
    """2-D (y, z) field of a closed photo-px polyline (any shape): exact distance, even-odd sign"""
    P = np.array([bp(*q) for q in pts_px]); n = len(P)
    d = np.full(Y.shape, 1e9); inside = np.zeros(Y.shape, bool)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        e = b - a; w0, w1 = Y - a[0], Z - a[1]
        t = np.clip((w0 * e[0] + w1 * e[1]) / max(e @ e, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(w0 - e[0] * t, w1 - e[1] * t))
        c = ((a[1] > Z) != (b[1] > Z)) & (Y < (b[0] - a[0]) * (Z - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
        inside ^= c
    return np.where(inside, -d, d) - r


def chain2(Y, Z, pts_px, rad):
    """2-D capsule chain along photo-px points (a bent strip / lever centre line)"""
    P = [bp(*q) for q in pts_px]
    d = np.full(Y.shape, 1e3)
    for a, b in zip(P, P[1:]):
        d = np.minimum(d, S.capsule2(Y, Z, a, b, rad))
    return d


def helix(X, Y, Z, zc, R_, wire, pitch, y0, y1):
    """coil spring along y about (x=0, z=zc)"""
    rho = np.hypot(X, Z - zc); th = np.arctan2(Z - zc, X) / (2 * np.pi)
    ph = (Y / pitch - th) % 1.0
    dy = (np.minimum(ph, 1 - ph)) * pitch
    d = np.hypot(rho - R_, dy) - wire
    return np.maximum(d, yband(Y, y0, y1))
