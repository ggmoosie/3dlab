"""softgoods/panel.py - panel outline (u, v mm) -> low-poly shell (outer face puffed, flat inner face, side walls) swept
along a 3-D path: u runs along the path, v along the panel's `across` vector, w outward (away from `center`).
Pure numpy + scipy. Units mm, frame: Blender axes (wearer faces -Y, +X wearer's left, +Z up)."""
import numpy as np
from scipy.spatial import Delaunay
from .core import seg_field


def frames(path, across, center):
    """resampled path -> s, P, T, B (across), N (outward) per sample"""
    P = np.asarray(path, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    T = np.gradient(P, s, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.broadcast_to(np.asarray(across, float), P.shape).copy()
    B -= T * np.sum(B * T, 1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
    N = np.cross(T, B)
    flip = np.sum(N * (P - np.asarray(center, float)), 1) < 0
    if flip.mean() > 0.5:
        N = -N
    return s, P, T, B, N


def place(u, v, w, fr):
    s, P, T, B, N = fr
    def at(A):
        return np.stack([np.interp(u, s, A[:, i]) for i in range(3)], -1)
    n = at(N); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    b = at(B); b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return at(P) + b * v[..., None] + n * w[..., None]


def shell(panel, grid=28.0):
    """-> V (n,3) mm, F (m,3), UVs per face corner in panel mm (m,3,2), face side (0 outer, 1 inner, 2 wall)"""
    O = np.asarray(panel["outline"], float)
    t, puff, pw = panel.get("t", 6.0), panel.get("puff", 2.0), panel.get("puff_w", 18.0)
    # boundary resampled + interior grid
    ring = []
    for a, b in zip(O, np.roll(O, -1, 0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / (grid / 2))))
        ring += [a + (b - a) * k / n for k in range(n)]
    ring = np.array(ring)
    lo, hi = O.min(0), O.max(0)
    gu, gv = np.meshgrid(np.arange(lo[0] + grid / 2, hi[0], grid), np.arange(lo[1] + grid / 2, hi[1], grid))
    G = np.c_[gu.ravel(), gv.ravel()]
    dG, _ = seg_field(G[:, 0], G[:, 1], O)
    G = G[dG < -grid * 0.35]
    pts = np.vstack([ring, G])
    tri = Delaunay(pts).simplices
    c = pts[tri].mean(1); dc, _ = seg_field(c[:, 0], c[:, 1], O); tri = tri[dc < 0]
    d, _ = seg_field(pts[:, 0], pts[:, 1], O)
    w_out = t + puff * np.sin(np.clip(-d / pw, 0, 1) * np.pi / 2)
    fr = frames(panel["path"], panel["across"], panel["center"])
    nb, n = len(ring), len(pts)
    Vo = place(pts[:, 0], pts[:, 1], w_out, fr); Vi = place(pts[:, 0], pts[:, 1], np.zeros(n), fr)
    V = np.vstack([Vo, Vi])
    # orient outer faces outward (winding test against the outward normal)
    e = np.cross(Vo[tri[:, 1]] - Vo[tri[:, 0]], Vo[tri[:, 2]] - Vo[tri[:, 0]])
    nrm = place(c[dc < 0][:, 0], c[dc < 0][:, 1], np.full(len(tri), 1.0), fr) - place(c[dc < 0][:, 0], c[dc < 0][:, 1], np.zeros(len(tri)), fr)
    tri = np.where((np.sum(e * nrm, 1) < 0)[:, None], tri[:, ::-1], tri)
    F_out = tri; F_in = tri[:, ::-1] + n
    ring_i = np.arange(nb); nxt = np.roll(ring_i, -1)
    walls = np.r_[np.c_[ring_i, nxt, nxt + n], np.c_[ring_i, nxt + n, ring_i + n]]
    ew = np.cross(V[walls[:, 1]] - V[walls[:, 0]], V[walls[:, 2]] - V[walls[:, 0]])
    mid = V[walls].mean(1); cen = Vo.mean(0)
    walls = np.where((np.sum(ew * (mid - cen), 1) < 0)[:, None], walls[:, ::-1], walls)
    F = np.vstack([F_out, F_in, walls]); side = np.r_[np.zeros(len(F_out)), np.ones(len(F_in)), np.full(len(walls), 2)].astype(int)
    P2 = np.vstack([pts, pts])
    UV = P2[F]                                                    # panel mm per corner (walls reuse the outline coords)
    return V, F, UV, side
