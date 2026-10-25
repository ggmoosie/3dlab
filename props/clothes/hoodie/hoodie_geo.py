"""Hoodie geometry in numpy (cm, Blender frame: front = -Y, left = +X).

Pipeline (same functions build the game mesh at level 2 and the bake source at level 4, so they agree):
  cage (torso rings + set-in sleeves)  -> Catmull-Clark (own, so seams / regions / loops / params propagate)
  -> project onto hoodie_body.garment_sdf -> tangential relax -> hood grid off the neckline loop -> folds
  -> pocket panel, drawstrings, grommets, knots, aglets -> shell solidify (own: rims, inner lining shell)
"""
import math
import numpy as np
import hoodie_body as B

# face regions
R_TORSO, R_BAND, R_SLEEVE, R_CUFF, R_HOOD, R_CHAN, R_POCKET, R_INNER, R_LINING, R_POCK_IN, R_RIM, R_STRING, R_GROM, R_KNOT, R_AGLET = range(15)
REGION_NAMES = ["torso", "band", "sleeve", "cuff", "hood", "channel", "pocket", "inner", "lining", "pocket_in", "rim", "string",
                "grommet", "knot", "aglet"]
T_BODY, T_RIB, T_HOOD, T_CHAN, T_POCKET = 0.45, 0.7, 0.4, 0.8, 0.35


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------- Catmull-Clark (quads)
def edges_of(F):
    e = np.stack([F, np.roll(F, -1, 1)], 2).reshape(-1, 2)
    es = np.sort(e, 1)
    E, inv = np.unique(es, axis=0, return_inverse=True)
    return E, inv.reshape(len(F), F.shape[1])


def catmull_clark(V, F, attrs=None, loops=(), seams=(), freg=None):
    nv, nf = len(V), len(F)
    E, FE = edges_of(F)
    ne = len(E)
    efc = np.bincount(FE.ravel(), minlength=ne)                 # faces per edge
    FP = V[F].mean(1)
    esum = np.zeros((ne, 3)); np.add.at(esum, FE.ravel(), np.repeat(FP, 4, 0))
    mid = V[E].mean(1)
    EP = np.where((efc == 2)[:, None], (V[E[:, 0]] + V[E[:, 1]] + esum) / 4, mid)
    # vertex points
    val = np.bincount(F.ravel(), minlength=nv).astype(float)    # faces per vertex
    Q = np.zeros((nv, 3)); np.add.at(Q, F.ravel(), np.repeat(FP, 4, 0)); Q /= np.maximum(val, 1)[:, None]
    deg = np.bincount(E.ravel(), minlength=nv).astype(float)
    Rm = np.zeros((nv, 3)); np.add.at(Rm, E[:, 0], mid); np.add.at(Rm, E[:, 1], mid); Rm /= np.maximum(deg, 1)[:, None]
    n = deg
    VP = (Q + 2 * Rm + (n - 3)[:, None] * V) / np.maximum(n, 1)[:, None]
    bnd = efc == 1
    bnb = np.zeros((nv, 3)); bcount = np.zeros(nv)
    be = E[bnd]
    np.add.at(bnb, be[:, 0], V[be[:, 1]]); np.add.at(bnb, be[:, 1], V[be[:, 0]])
    np.add.at(bcount, be[:, 0], 1); np.add.at(bcount, be[:, 1], 1)
    isb = bcount > 0
    VP[isb] = 0.75 * V[isb] + 0.125 * bnb[isb]
    NV = np.concatenate([VP, EP, FP])
    fi = nv + ne + np.arange(nf)
    e0, e1, e2, e3 = (nv + FE[:, k] for k in range(4))          # FE[:,k] = edge (F[k], F[k+1])
    a, b, c, d = F.T
    NF = np.concatenate([np.stack([a, e0, fi, e3], 1), np.stack([b, e1, fi, e0], 1),
                         np.stack([c, e2, fi, e1], 1), np.stack([d, e3, fi, e2], 1)])
    out_attrs = {}
    for k, A in (attrs or {}).items():
        A = np.asarray(A, float)
        out_attrs[k] = np.concatenate([A, A[E].mean(1), A[F].mean(1)])
    emap = {(int(x), int(y)): nv + i for i, (x, y) in enumerate(E)}

    def ep(x, y):
        return emap[(min(x, y), max(x, y))]
    new_loops = []
    for lp, closed in loops:
        out = []
        m = len(lp)
        for i in range(m if closed else m - 1):
            out += [lp[i], ep(lp[i], lp[(i + 1) % m])]
        if not closed:
            out.append(lp[-1])
        new_loops.append((out, closed))
    new_seams = {}
    for (x, y), typ in (seams.items() if isinstance(seams, dict) else ((e, "cut") for e in seams)):
        p = ep(x, y)
        new_seams[(min(x, p), max(x, p))] = typ; new_seams[(min(y, p), max(y, p))] = typ
    nreg = None if freg is None else np.tile(np.asarray(freg), 4)
    return NV, NF, out_attrs, new_loops, new_seams, nreg


def vertex_normals(V, F):
    n = np.zeros_like(V)
    for k in range(1, F.shape[1] - 1):
        fn = np.cross(V[F[:, k]] - V[F[:, 0]], V[F[:, k + 1]] - V[F[:, 0]])
        for j in (0, k, k + 1):
            np.add.at(n, F[:, j], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def neighbours(F, nv):
    E, _ = edges_of(F)
    return E


def relax(V, F, fixed, f, target=0.0, iters=6, lam=0.45):
    """Laplacian relax (umbrella) with re-projection onto f = target; fixed verts stay put."""
    E = neighbours(F, len(V))
    deg = np.bincount(E.ravel(), minlength=len(V)).astype(float)
    for _ in range(iters):
        s = np.zeros_like(V); np.add.at(s, E[:, 0], V[E[:, 1]]); np.add.at(s, E[:, 1], V[E[:, 0]])
        L = s / np.maximum(deg, 1)[:, None] - V
        Vn = V + lam * L
        Vn = B.project(f, Vn, target, iters=4)
        V = np.where(fixed[:, None], V, Vn)
    return V


# ------------------------------------------------------------------------------------------------------- body cage
TH = np.radians(np.arange(12) * 30.0)                       # around the torso, 0 = front centre, +90 = left (+X)
ROWS_Z = [B.Z_HEM, B.Z_BAND, 103.0, 116.0, 128.0]


def ring_pts(z, scale=1.0):
    P = []
    for t in TH:
        a = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 1]) + np.interp(z, B.EASE_T[:, 0], B.EASE_T[:, 1])
        b = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 2]) + np.interp(z, B.EASE_T[:, 0], B.EASE_T[:, 1])
        c = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 3])
        P.append((a * math.sin(t) * scale, c - b * math.cos(t) * scale, z))
    return np.array(P)


def neckline_xyz(theta):
    """Hood neckline: an ellipse around the neck axis (crew drop at the front), dropped straight down onto the garment."""
    th = np.asarray(theta, float)
    fr = ((1 + np.cos(th)) / 2) ** 1.5
    ry = 7.1 + 3.4 * fr
    x, y = 8.5 * np.sin(th), 0.9 - ry * np.cos(th)
    zs = np.arange(164.0, 140.0, -0.02)
    out = []
    for xi, yi in zip(x, y):
        d = B.garment_sdf(np.stack([np.full(len(zs), xi), np.full(len(zs), yi), zs], 1))
        k = int(np.argmax(d < 0))
        out.append((xi, yi, zs[k]))
    return np.array(out)


def project_in_plane(f, P, axis, iters=8):
    """Projection onto f = 0 that never moves along `axis` (hem stays level, cuff stays square to the forearm)."""
    P = np.array(P, float); a = np.asarray(axis, float) / np.linalg.norm(axis)
    for _ in range(iters):
        g = B.grad(f, P); g -= (g @ a)[:, None] * a
        P -= (f(P) / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
    return P


def arm_axis(s):
    sh, el, wr = B.J["upperarm_" + s], B.J["lowerarm_" + s], B.J["hand_" + s]
    return sh, el, wr


def body_cage():
    V, F, reg, seams = [], [], [], {}
    sl = []

    def add(p, slv=0.0):
        V.append(np.asarray(p, float)); sl.append(slv)
        return len(V) - 1
    rows = [[add(p) for p in ring_pts(z)] for z in ROWS_Z]
    # R4 side points sit under the arm
    for k, x in ((3, 1), (9, -1)):
        V[rows[4][k]] = np.array([x * 18.6, 1.2, 129.0])
    # R5 shoulder line
    r5 = []
    for k, t in enumerate(TH):
        if k in (3, 9):
            p = (math.sin(t) * 20.4, 2.4, 146.4)
        else:
            q = ring_pts(141.5)[k]
            z = 141.5 + (2.0 if abs(math.cos(t)) > 0.8 else 0.0) + (0.8 if math.cos(t) < -0.5 else 0.0)
            p = (q[0] * (1.0 if abs(math.sin(t)) < 0.6 else 0.93), q[1], z)
        r5.append(add(p))
    rows.append(r5)
    rows.append([add(p) for p in neckline_xyz(TH)])
    nrow = len(rows)
    arm_faces = {4: (2, 3, 8, 9)}
    for i in range(nrow - 1):
        for k in range(12):
            if i == 4 and k in (2, 3, 8, 9):
                continue
            F.append((rows[i][k], rows[i][(k + 1) % 12], rows[i + 1][(k + 1) % 12], rows[i + 1][k]))
            reg.append(R_BAND if i == 0 else R_TORSO)
    # seams: side seams, band seam, shoulder seams, neckline
    for k in (3, 9):
        for i in range(4):
            seams[tuple(sorted((rows[i][k], rows[i + 1][k])))] = "side"
        seams[tuple(sorted((rows[5][k], rows[6][k])))] = "cover"
    for k in range(12):
        seams[tuple(sorted((rows[1][k], rows[1][(k + 1) % 12])))] = "rib"
        seams[tuple(sorted((rows[6][k], rows[6][(k + 1) % 12])))] = "cover"
    # sleeves
    cuff_loops = []
    for s, ks in (("l", (2, 3, 4)), ("r", (10, 9, 8))):
        sh, el, wr = arm_axis(s)
        L = [rows[4][ks[0]], rows[4][ks[1]], rows[4][ks[2]], rows[5][ks[2]], rows[5][ks[1]], rows[5][ks[0]]]
        for i in range(6):
            seams[tuple(sorted((L[i], L[(i + 1) % 6])))] = "cover"
        ud = (el - sh) / np.linalg.norm(el - sh); fd = (wr - el) / np.linalg.norm(wr - el)
        fl = np.linalg.norm(wr - el)
        centres = [(sh + 0.25 * (el - sh), ud, 7.5), (sh + 0.84 * (el - sh), ud, 7.0),
                   (el, (ud + fd) / np.linalg.norm(ud + fd), 6.8), (el + 0.17 * (wr - el), fd, 6.6),
                   (wr - 6.5 * fd, fd, 4.5), (wr + 0.6 * fd, fd, 4.2)]
        # armhole angles in the upper-arm frame
        ref = np.array([0, -1.0, 0])
        e1 = ref - ref @ ud * ud; e1 /= np.linalg.norm(e1); e2 = np.cross(ud, e1)
        c0 = np.mean([V[i] for i in L], 0)
        ang = [math.atan2((V[i] - c0) @ e2, (V[i] - c0) @ e1) for i in L]
        prev = L
        for ri, (c, d, r) in enumerate(centres):
            e1 = e1 - (e1 @ d) * d; e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
            ring = [add(c + r * (math.cos(a) * e1 + math.sin(a) * e2), 1.0) for a in ang]
            for i in range(6):
                F.append((prev[i], prev[(i + 1) % 6], ring[(i + 1) % 6], ring[i]))
                reg.append(R_CUFF if ri == len(centres) - 1 else R_SLEEVE)
            seams[tuple(sorted((prev[1], ring[1])))] = "side"      # under-sleeve seam continues the side seam
            if ri == len(centres) - 2:
                for i in range(6):
                    seams[tuple(sorted((ring[i], ring[(i + 1) % 6])))] = "rib"   # cuff seam
            prev = ring
        cuff_loops.append(prev)
    V = np.array(V); F = np.array(F)
    # orient sleeve faces like the torso (outward)
    Fn = F.copy()
    cen = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 3]] - V[F[:, 0]])
    reg = np.array(reg)
    for s in "lr":
        sh, el, wr = arm_axis(s)
        sel = np.where(((reg == R_SLEEVE) | (reg == R_CUFF)) & (np.sign(cen[:, 0]) == (1 if s == "l" else -1)))[0]
        t = np.clip(B.cone_t(cen[sel], sh, el), 0, 1)
        ax = sh + t[:, None] * (el - sh)
        if np.mean(np.sum(fn[sel] * (cen[sel] - ax), 1)) < 0:
            Fn[sel] = F[sel][:, ::-1]
    loops = [(rows[6], True), (rows[0], True)] + [(c, True) for c in cuff_loops]
    return V, Fn, {"sl": np.array(sl)}, loops, seams, reg


def body_mesh(level):
    V, F, at, loops, seams, reg = body_cage()
    for _ in range(level):
        V, F, at, loops, seams, reg = catmull_clark(V, F, at, loops, seams, reg)
    fixed = np.zeros(len(V), bool)
    for lp, _ in loops:
        fixed[lp] = True
    neck = loops[0][0]
    hem, cuffs = loops[1][0], (loops[2][0], loops[3][0])
    keep = V.copy()
    V = B.project(B.garment_sdf, V, 0.0, iters=10)
    V[hem] = project_in_plane(B.garment_sdf, keep[hem], (0, 0, 1))
    for s, c in zip("lr", cuffs):
        sh, el, wr = arm_axis(s)
        V[c] = project_in_plane(B.garment_sdf, keep[c], wr - el)
    th = np.linspace(0, 2 * np.pi, len(neck), endpoint=False)
    V[neck] = neckline_xyz(th)
    V = relax(V, F, fixed, B.garment_sdf, iters=8 if level <= 2 else 14)
    return dict(V=V, F=F, at=at, loops=loops, seams=seams, reg=reg)


# ------------------------------------------------------------------------------------------------------------- hood
NECK_AX = np.array([0.0, 0.9, 0.0])
PHI_V = math.radians(21.0)                  # half-angle of the front V where the hood edges meet the neckline
O1, DO = 1.3, 1.6                           # layer-1 offset over the garment, layer-2 extra offset (two fabrics + air)
CHAN_W = 2.5                                # drawstring channel width


def support_sdf(P):
    """What the hood lies on: the hoodie outer surface, and the neck with 1.4 cm clearance."""
    return B.smin(B.garment_sdf(P), B.neck_sdf(P) - 1.4, 1.5)


def surf(P, iters=6):
    """Project onto the support; return (point, unit normal)."""
    sh = P.shape
    Q = B.project(support_sdf, P.reshape(-1, 3), 0.0, iters=iters)
    n = B.grad(support_sdf, Q); n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
    return Q.reshape(sh), n.reshape(sh)


def bez(p0, p1, p2, p3, t):
    t = t[..., None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


R1, R2 = 0.6, 0.85                          # roll-over radius at the neckline, crown-fold radius (layer gaps 1.2 / 1.7 cm)


def hood_paths(N, phi, npts=30):
    """Every hood column is a 'turtle' path in the vertical plane through its neckline point:
    rise -> roll-over turn (outwards) -> layer 1 hanging down the back -> crown fold (turning back out) -> layer 2 up.
    Turn angles and lengths vary smoothly with the column angle: tall straight collar at the front V, a lip that curls
    outwards at the front sides, a thick roll at the sides, the full two-layer bag on the back. Fixed turn radii keep
    every layer gap >= 2 fabric thicknesses, so layers never cross.  Returns P (c, 5*npts, 3), segment ids, bag weight."""
    c = len(phi)
    up = np.array([0, 0, 1.0])
    ph = np.minimum(phi, 2 * np.pi - phi)
    wr = sstep(math.radians(30), math.radians(85), ph)          # roll-over
    w = sstep(math.radians(55), math.radians(120), ph)          # bag
    psi = phi - np.pi
    backw = np.maximum(np.cos(psi), 0)
    psi_h = PHI_V + (2 * np.pi - 2 * PHI_V) * phi / (2 * np.pi)
    out_h = np.stack([np.sin(psi_h), -np.cos(psi_h), np.zeros(c)], 1)
    back = np.array([0, 1.0, 0])
    dd = out_h + back[None] * (0.8 * np.abs(np.sin(phi)) * (np.cos(phi) < 0.3))[:, None]
    dd /= np.linalg.norm(dd, axis=1)[:, None]
    Nh = neckline_xyz(psi_h)                                    # where each column sits once past the V fan
    h = 4.4 - 3.4 * wr
    th0 = -0.25 + 0.45 * wr
    Lbag = 3.5 + 23.0 * backw ** 1.3
    L1, L2 = w * Lbag, w * Lbag * 0.99
    T1, T2 = np.pi * wr, np.pi * sstep(0.0, 0.3, w)
    P = np.zeros((c, 5 * npts, 3)); sid = np.repeat(np.arange(5), npts)
    mind = np.zeros((c, 5 * npts))
    for i in range(c):
        pts, th, pos = [], th0[i], np.zeros(2)                    # pos = (along dd, up)
        def straight(L):
            nonlocal pos
            out = []
            for t in np.linspace(0, 1, npts, endpoint=False):
                out.append(pos + L * t * np.array([math.sin(th), math.cos(th)]))
            pos = pos + L * np.array([math.sin(th), math.cos(th)])
            return out
        def turn(ang, r, sgn):
            nonlocal pos, th
            out = []
            ctr = pos + sgn * r * np.array([math.cos(th), -math.sin(th)])
            for t in np.linspace(0, 1, npts, endpoint=False):
                a_ = th + sgn * ang * t
                out.append(ctr - sgn * r * np.array([math.cos(a_), -math.sin(a_)]))
            th = th + sgn * ang
            pos = ctr - sgn * r * np.array([math.cos(th), -math.sin(th)])
            return out
        seg = straight(h[i]) + turn(T1[i], R1, 1) + straight(L1[i]) + turn(T2[i], R2, -1) + straight(L2[i])
        seg = np.array(seg + [pos])[:5 * npts]
        q = np.linspace(0, 1, len(seg))
        P[i] = N[i] + seg[:, :1] * dd[i] + seg[:, 1:2] * up
        # V fan: slide from the neckline point to the column's own angle over the rise
        fan = sstep(0.0, 0.6, np.clip(seg[:, 1] / max(h[i], 1e-3), 0, 1))
        fan[npts:] = 1.0
        P[i, :, :2] += fan[:, None] * (Nh[i, :2] - N[i, :2])
        # minimum clearance over the support for the hanging layers
        k1 = 2 * npts
        mind[i, k1:3 * npts] = 2 * R1
        tt = np.linspace(0, 1, npts)
        mind[i, 3 * npts:4 * npts] = 2 * R1 + 2 * R2 * (1 - np.cos(np.pi * tt)) / 2
        mind[i, 4 * npts:] = 2 * R1 + 2 * R2
    # keep hanging layers off the back (push along the support gradient)
    flat = P.reshape(-1, 3); md = (mind * w[:, None]).reshape(-1)
    sel = md > 0
    for _ in range(4):
        dsup = support_sdf(flat[sel])
        need = md[sel] - dsup
        g = B.grad(support_sdf, flat[sel]); g /= np.maximum(np.linalg.norm(g, axis=1), 1e-9)[:, None]
        tmp = flat[sel]; tmp += g * np.maximum(need, 0)[:, None]; flat[sel] = tmp
    return flat.reshape(P.shape), sid, w


def resample(P, sid, w, rows_mult=1):
    """Rows by arclength: channel = last CHAN_W cm; before it, rows blend between uniform (front) and segment-aligned (back)."""
    c, n, _ = P.shape
    seglen = np.linalg.norm(np.diff(P, axis=1), axis=2)
    S = np.concatenate([np.zeros((c, 1)), np.cumsum(seglen, 1)], 1)
    L = S[:, -1]
    body_end = np.maximum(L - CHAN_W, 0.45 * L)
    sb = np.stack([S[:, np.argmax(sid == k)] for k in range(5)], 1)
    al = [0, 1, 3, 6, 8, 10]                     # rows at segment starts: rise 1, roll 2, layer 1 3, fold 2, layer 2 2
    ks = np.arange(0, 10 * rows_mult + 1) / rows_mult
    al_s = np.stack([np.interp(ks, al, [sb[i, 0], sb[i, 1], sb[i, 2], sb[i, 3], sb[i, 4], body_end[i]]) for i in range(c)])
    al_s = np.minimum(al_s, body_end[:, None])
    al_s = np.maximum.accumulate(al_s, axis=1)
    un_s = body_end[:, None] * ks[None, :] / 10
    tgt = un_s + (al_s - un_s) * w[:, None]
    ch = body_end[:, None] + (L - body_end)[:, None] * (np.arange(1, 2 * rows_mult + 1) / (2 * rows_mult))[None, :]
    tgt = np.concatenate([tgt, ch], 1)
    out = np.stack([np.stack([np.interp(tgt[i], S[i], P[i, :, k]) for k in range(3)], 1) for i in range(c)])
    u = tgt / L[:, None]
    return out, u


def hood_mesh(body, mult=1):
    """Hood grid hung off the body's neckline loop. Returns verts (new only), faces using body indices for row 0."""
    V = body["V"]
    neck = body["loops"][0][0]
    n = len(neck)
    phi = np.linspace(0, 2 * np.pi, n + 1)
    N = V[np.array(neck + [neck[0]])]
    P, sid, w = hood_paths(N, phi)
    R, u = resample(P, sid, w, mult)
    rows = R.shape[1]                                     # includes row 0 (the neckline)
    base = len(V)
    idx = np.full((n + 1, rows), -1, int)
    idx[:n, 0] = neck; idx[n, 0] = neck[0]
    newV = R[:, 1:].reshape(-1, 3)
    idx[:, 1:] = base + np.arange((n + 1) * (rows - 1)).reshape(n + 1, rows - 1)
    F = []
    for j in range(n):
        for r in range(rows - 1):
            F.append((idx[j, r], idx[j + 1, r], idx[j + 1, r + 1], idx[j, r + 1]))
    F = np.array(F)
    reg = np.where(np.tile(np.arange(rows - 1), n) >= rows - 1 - 2 * mult, R_CHAN, R_HOOD)
    seams = {}
    centre = n // 2
    for r in range(rows - 1):
        seams[tuple(sorted((idx[centre, r], idx[centre, r + 1])))] = "cover"
    chan_row = rows - 1 - 2 * mult
    for j in range(n):
        seams[tuple(sorted((idx[j, chan_row], idx[j + 1, chan_row])))] = "chan"
        for r in (3 * mult, 7 * mult):                         # UV cuts after the roll and through the crown fold
            seams[tuple(sorted((idx[j, r], idx[j + 1, r])))] = "cut"
    params = dict(phi=np.repeat(phi, rows - 1), u=u[:, 1:].ravel(), w=np.repeat(w, rows - 1))
    return dict(V=newV, F=F, reg=reg, seams=seams, idx=idx, params=params, chan_row=chan_row)


# ------------------------------------------------------------------------------------------------------------ folds
def arm_coords(P, s):
    """Arclength along the arm polyline (shoulder joint -> elbow -> wrist), radial unit vector, closest axis point."""
    sh, el, wr = arm_axis(s)
    l1, l2 = np.linalg.norm(el - sh), np.linalg.norm(wr - el)
    t1 = np.clip(B.cone_t(P, sh, el), 0, 1); t2 = np.clip(B.cone_t(P, el, wr), 0, 1)
    q1 = sh + t1[:, None] * (el - sh); q2 = el + t2[:, None] * (wr - el)
    d1 = np.linalg.norm(P - q1, axis=1); d2 = np.linalg.norm(P - q2, axis=1)
    use2 = d2 < d1
    q = np.where(use2[:, None], q2, q1)
    sarc = np.where(use2, l1 + t2 * l2, t1 * l1)
    r = P - q; r /= np.maximum(np.linalg.norm(r, axis=1), 1e-9)[:, None]
    return sarc, r, l1, l2


def pocket_mask(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return (1 - sstep(15.5, 18.0, np.abs(x))) * (1 - sstep(113.0, 116.5, z)) * (1 - sstep(-4.0, 0.0, y)) * (z > 86)


def folds(P, sl, level_hi=False, hood=None):
    """Displacement (cm, along the outward normal) of the cloth folds. Low = sculpted primary folds; hi adds secondary folds
    and crumple. Same function on the game mesh and on the bake source, so they agree."""
    n = len(P)
    d = np.zeros(n)
    x, y, z = P.T
    pm = pocket_mask(P)
    prof = lambda a: np.abs(np.sin(a)) - 0.6              # rounded ridges, creased valleys
    for s, sg in (("l", 1), ("r", -1)):
        side = (np.sign(x) == sg) & (sl > 0.02)
        if not side.any():
            continue
        Q = P[side]
        sarc, r, l1, l2 = arm_coords(Q, s)
        sh, el, wr = arm_axis(s)
        ud = (el - sh) / l1; fd = (wr - el) / l2
        inner = fd - (fd @ ud) * ud; inner /= np.linalg.norm(inner)
        e2 = np.cross(ud, inner)
        c = r @ inner
        th = np.arctan2(r @ e2, c)
        w_sl = sl[side]
        dd = np.zeros(len(Q))
        # elbow: compression folds on the inside of the bend, softer rings around the back
        win = np.exp(-((sarc - l1) / 6.0) ** 2)
        a = 2 * np.pi * (sarc - l1) / 3.4 + 1.3 * c + 0.7 * np.sin(2 * th)
        dd += 0.62 * (0.3 + 0.7 * np.maximum(c, 0)) * win * prof(a)
        # stacked bunching above the ribbed cuff
        sc = l1 + l2 - 6.5
        win = np.exp(-((sarc - (sc - 4.5)) / 3.6) ** 2) * (sarc < sc + 0.3)
        a = 2 * np.pi * sarc / 2.8 + 0.9 * np.sin(2 * th + 0.4 * sarc) + 0.6 * th
        dd += 0.48 * win * prof(a)
        # upper sleeve: diagonal drag folds from the armpit along the underside
        under = np.maximum(-(r @ np.array([0, 0, 1.0])), 0)
        win = sstep(4, 9, sarc) * (1 - sstep(l1 - 8, l1 - 4, sarc))
        a = 2 * np.pi * (sarc + 3.0 * th) / 7.0
        dd += 0.26 * under * win * prof(a)
        # forearm: a long soft twist
        win = sstep(l1 + 3, l1 + 6, sarc) * (1 - sstep(sc - 9, sc - 6, sarc))
        dd += 0.18 * win * prof(2 * np.pi * (sarc - 4.0 * th) / 9.0)
        if level_hi:
            win = np.exp(-((sarc - l1) / 7.0) ** 2) + np.exp(-((sarc - (sc - 4)) / 4.0) ** 2)
            dd += 0.12 * win * prof(2 * np.pi * sarc / 1.5 + 2.1 * th + 0.8 * np.sin(3 * th))
        d[side] += dd * np.clip(w_sl, 0, 1)
        # armpit: radial folds fanning onto the torso side and under the sleeve
        ap = np.array([sg * 18.6, 1.2, 129.0])
        v = P - ap
        rr = np.linalg.norm(v, axis=1)
        al = np.arctan2(v[:, 1], v[:, 2] + 1e-9)
        win = np.exp(-((rr - 7.0) / 4.5) ** 2) * sstep(1.5, 3.5, rr) * (np.abs(x) > 12)
        d += 0.52 * win * prof(2.6 * al + 0.22 * rr)
        if level_hi:
            d += 0.10 * win * prof(5.3 * al + 0.5 * rr + 1.0)
    torso = sl < 0.5
    # waist: gathers above the ribbed band (vertical) + the blouse rolling over the band (horizontal)
    cy = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 3])
    th = np.arctan2(x, -(y - cy))
    win = np.exp(-((z - 96.0) / 4.0) ** 2) * (z > B.Z_BAND)
    d += torso * (1 - pm) * 0.50 * win * prof(7.0 * th + 0.9 * np.sin(3 * th) + 0.12 * z)
    win = np.exp(-((z - 93.2) / 2.0) ** 2) * (z > B.Z_BAND + 0.3)
    d += torso * (1 - pm) * 0.22 * win * prof(2 * np.pi * (z - B.Z_BAND) / 3.0 + 0.5 * np.sin(4 * th))
    # side seams pull a few diagonal folds over the hips
    win = np.exp(-((z - 101.0) / 6.0) ** 2) * sstep(0.75, 0.95, np.abs(np.sin(th)))
    d += torso * (1 - pm) * 0.24 * win * prof(2 * np.pi * (z + 6 * np.abs(th)) / 6.5)
    # soft large-scale drape everywhere (keeps panels from looking inflated)
    d += (1 - pm) * 0.16 * np.sin(0.23 * x + 0.11 * z + 0.7) * np.sin(0.19 * y + 0.13 * z + 1.9)
    if level_hi:
        d += (1 - pm) * 0.05 * np.sin(0.9 * x + 0.5 * z) * np.sin(0.8 * y - 0.7 * z + 0.4) * np.sin(0.6 * z + 1.3 * x)
    if hood is not None:
        phi, u, w = hood
        bag = sstep(0.10, 0.25, u) * (1 - sstep(0.80, 0.92, u)) * w
        a = 5.4 * (phi - np.pi) + 0.9 * np.sin(3.0 * u + phi)
        d += bag * (0.55 + 0.25 * np.sin(2 * phi)) * prof(a)
        if level_hi:
            d += bag * 0.12 * prof(11 * (phi - np.pi) + 4 * u)
    return d


def displace(V, F, d, keep=None):
    n = vertex_normals(V, F)
    Vd = V + n * d[:, None]
    if keep is not None:
        Vd[keep] = V[keep]
    return Vd


def clear_body(V, F, thick, margin=0.35, mask=None):
    """Push outer verts out so the inner fabric face keeps `margin` from the proxy body."""
    d = B.body_sdf(V)
    need = (thick + margin) - d
    sel = need > 0
    if mask is not None:
        sel &= mask
    if sel.any():
        g = B.grad(B.body_sdf, V[sel]); g /= np.linalg.norm(g, axis=1)[:, None]
        V = V.copy(); V[sel] += g * need[sel][:, None]
    return V, int(sel.sum())


# ------------------------------------------------------------------------------------------------------------ pocket
PK_Z0, PK_Z1, PK_ZO, PK_W0, PK_W1 = 91.9, 113.5, 97.0, 16.0, 9.5


def pocket_edge_x(z):
    t = np.clip((z - PK_ZO) / (PK_Z1 - PK_ZO), 0, 1)
    return PK_W0 + (PK_W1 - PK_W0) * (t + 0.18 * np.sin(np.pi * t))      # slightly bellied opening curve


def front_hit(x, z, f):
    """Ray from the front (+Y) onto f = 0: y of the first crossing."""
    ys = np.arange(-30.0, 5.0, 0.02)
    out = []
    for xi, zi in zip(x, z):
        Q = np.stack([np.full(len(ys), xi), ys, np.full(len(ys), zi)], 1)
        k = int(np.argmax(f(Q) < 0))
        out.append(ys[k])
    return np.array(out)


def pocket_mesh(mult=1, surf_f=None):
    ns, nv = 16 * mult, 8 * mult
    s = np.linspace(-1, 1, ns + 1); v = np.linspace(0, 1, nv + 1)
    S, Vv = np.meshgrid(s, v, indexing="ij")
    z = PK_Z0 + (PK_Z1 - PK_Z0) * Vv
    x = S * pocket_edge_x(z)
    y = front_hit(x.ravel(), z.ravel(), surf_f or B.garment_sdf).reshape(x.shape)
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    nrm = B.grad(surf_f or B.garment_sdf, P); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    zs = z.ravel(); a = np.abs(S.ravel())
    tz = np.clip((zs - PK_ZO) / (PK_Z1 - PK_ZO), 0, 1)
    lift = 1.15 * sstep(0.45, 1.0, a) * np.sin(np.pi * tz) ** 0.8 * (zs > PK_ZO)
    puff = 0.35 * np.sin(np.pi * Vv.ravel()) * np.cos(np.pi * S.ravel() / 2.2)
    off = 0.30 + lift + puff
    P = P + nrm * off[:, None]
    idx = np.arange(len(P)).reshape(ns + 1, nv + 1)
    F = np.array([(idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]) for i in range(ns) for j in range(nv)])
    # boundary in order (for rim cuts) and the opening edges (for hem / wear)
    opening = np.concatenate([idx[0, :], idx[-1, :]])
    return dict(V=P, F=F, idx=idx, opening=opening, S=S.ravel(), Vv=Vv.ravel())


# ------------------------------------------------------------------------------------------------------- solidify
INNER_OF = {R_TORSO: R_INNER, R_BAND: R_INNER, R_SLEEVE: R_INNER, R_CUFF: R_INNER, R_HOOD: R_LINING, R_CHAN: R_LINING,
            R_POCKET: R_POCK_IN}
THICK_OF = {R_TORSO: T_BODY, R_BAND: T_RIB, R_SLEEVE: T_BODY, R_CUFF: T_RIB, R_HOOD: T_HOOD, R_CHAN: T_CHAN, R_POCKET: T_POCKET}


def vertex_thickness(nv, F, reg):
    t = np.zeros(nv)
    for k in range(4):
        np.maximum.at(t, F[:, k], np.vectorize(THICK_OF.get)(reg))
    return t


def solidify(V, F, reg, seams, thick, cut_verts=()):
    n = len(V)
    vn = vertex_normals(V, F)
    Vi = V - vn * thick[:, None]
    faces = [tuple(f) for f in F] + [tuple(int(x) + n for x in f[::-1]) for f in F]
    fr = list(reg) + [INNER_OF[r] for r in reg]
    finner = [0] * len(F) + [1] * len(F)
    E, FE = edges_of(F)
    cnt = np.bincount(FE.ravel(), minlength=len(E))
    bset = {tuple(e) for e in E[cnt == 1]}
    new_seams = set(seams) | {(a + n, b + n) for a, b in seams}
    bverts = set()
    for f in F:
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            if (min(a, b), max(a, b)) in bset:
                faces.append((b, a, a + n, b + n)); fr.append(R_RIM); finner.append(2)
                new_seams |= {(min(a, b), max(a, b)), (min(a, b) + n, max(a, b) + n)}
                bverts |= {a, b}
    seam_v = {v for e in seams for v in e} | set(cut_verts)
    for v in bverts & seam_v:
        new_seams.add((v, v + n))
    return np.concatenate([V, Vi]), faces, fr, finner, new_seams


# ----------------------------------------------------------------------------------------------- tubes and lathes
def frames(path):
    t = np.gradient(path, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    ref = np.array([0, 0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    n0 = np.cross(t[0], ref); n0 /= np.linalg.norm(n0)
    N = [n0]
    for i in range(1, len(path)):
        m = N[-1] - (N[-1] @ t[i]) * t[i]; m /= np.linalg.norm(m); N.append(m)
    N = np.array(N); Bn = np.cross(t, N)
    return t, N, Bn


def tube(path, radius, sides, closed=False, caps=True, squash=1.0):
    path = np.asarray(path, float); m = len(path)
    r = np.broadcast_to(np.asarray(radius, float), (m,))
    t, N, Bn = frames(path)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = (path[:, None] + r[:, None, None] * (np.cos(a)[None, :, None] * N[:, None] + squash * np.sin(a)[None, :, None] * Bn[:, None])).reshape(-1, 3)
    F, seams = [], set()
    rings = m if closed else m - 1
    for i in range(rings):
        i2 = (i + 1) % m
        for k in range(sides):
            k2 = (k + 1) % sides
            F.append((i * sides + k, i * sides + k2, i2 * sides + k2, i2 * sides + k))
        seams.add(tuple(sorted((i * sides, i2 * sides))))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    UVc = [(k / sides * 2 * np.pi * r[i], arc[i]) for i in range(m) for k in range(sides)]
    V = list(V)
    if not closed and caps:
        for end, sgn in ((0, -1), (m - 1, 1)):
            c = len(V); V.append(path[end] + t[end] * sgn * r[end] * 0.35); UVc.append((0.0, arc[end]))
            ring = [end * sides + k for k in range(sides)]
            for k in range(sides):
                a_, b_ = ring[k], ring[(k + 1) % sides]
                F.append((a_, b_, c) if sgn > 0 else (b_, a_, c))
                seams.add(tuple(sorted((a_, b_))))
    tube.coords = np.array(UVc)
    return np.array(V), F, seams


def lathe(profile, origin, axis, segs):
    """Closed (r, h) profile revolved about axis -> torus-topology shell."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(axis, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    p = np.asarray(profile, float); k = len(p)
    a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    V = (np.asarray(origin)[None, None] + p[None, :, 0:1] * (np.cos(a)[:, None, None] * e1 + np.sin(a)[:, None, None] * e2)
         + p[None, :, 1:2] * axis).reshape(-1, 3)
    F, seams = [], set()
    for i in range(segs):
        i2 = (i + 1) % segs
        for j in range(k):
            j2 = (j + 1) % k
            F.append((i * k + j2, i2 * k + j2, i2 * k + j, i * k + j))
        seams.add(tuple(sorted((i * k, i2 * k))))
    for j in range(k):
        seams.add(tuple(sorted((j, (j + 1) % k))))
    return V, F, seams


GROMMET = [(0.30, -0.95), (0.30, 0.08), (0.42, 0.19), (0.58, 0.15), (0.63, 0.05), (0.42, 0.0), (0.42, -0.80), (0.58, -0.86),
           (0.52, -0.96), (0.34, -0.98)]


def trefoil(centre, axis, scale, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    q = np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -1.6 * np.sin(3 * t)], 1) * scale
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    return np.asarray(centre) + q[:, 0:1] * e1 + q[:, 1:2] * e2 + q[:, 2:3] * axis


# ------------------------------------------------------------------------------------------------------- drawstrings
STR_R, STR_SQ = 0.26, 0.75


def rope(start_pts, length, bvh, clear, seg=0.5, iters=260):
    """PBD rope: first points pinned (inside the channel, out of the eyelet), the rest hangs under gravity and rests on the
    garment (bvh = mathutils BVHTree of the outer surface)."""
    from mathutils import Vector
    pin = [np.asarray(p, float) for p in start_pts]
    pl = sum(np.linalg.norm(b - a) for a, b in zip(pin, pin[1:]))
    m = int(round((length - pl) / seg))
    d0 = pin[-1] - pin[-2]; d0 /= np.linalg.norm(d0)
    X = [pin[-1] + (d0 * 0.5 + np.array([0, 0, -1.0])) / np.linalg.norm(d0 * 0.5 + np.array([0, 0, -1.0])) * seg * (i + 1) for i in range(m)]
    X = np.array(pin + X)
    k = len(pin)
    L = np.r_[[np.linalg.norm(b - a) for a, b in zip(pin, pin[1:])], np.full(m, seg)]
    for it in range(iters):
        X[k:, 2] -= 0.06
        for _ in range(6):
            for i in range(k - 1, len(X) - 1):
                a, b = X[i], X[i + 1]
                v = b - a; l = np.linalg.norm(v) + 1e-12
                corr = (l - L[i]) * v / l
                if i < k - 1:
                    continue
                if i == k - 1:
                    X[i + 1] -= corr
                else:
                    X[i] += corr * 0.5; X[i + 1] -= corr * 0.5
        for i in range(k, len(X)):
            q, nrm, _, dist = bvh.find_nearest(Vector(X[i]))
            if q is None:
                continue
            q, nrm = np.array(q), np.array(nrm)
            h = (X[i] - q) @ nrm
            if h < clear:
                X[i] += nrm * (clear - h)
    return X


def resample_poly(P, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


def string_paths(Vo, Fo, eyelets):
    """eyelets: [(point on the channel outer face, outward normal), ...] -> dense centre-lines (left, right)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons([Vector(v) for v in Vo], [list(map(int, f)) for f in Fo])
    out = []
    for (p, n), length in zip(eyelets, (31.0, 28.5)):
        start = [p - n * 0.55, p - n * 0.1, p + n * 0.55, p + n * 1.0 + np.array([0, 0, -0.3])]
        X = rope(start, length, bvh, STR_R + 0.12)
        Xs = X.copy()
        for _ in range(3):                                   # light smoothing (keeps the pins)
            Xs[4:-1] = 0.25 * Xs[3:-2] + 0.5 * Xs[4:-1] + 0.25 * Xs[5:]
        out.append(Xs)
    return out


def string_parts(paths, hi=False):
    """Cord tube + overhand knot + aglet for each centre-line."""
    parts = []
    for P in paths:
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))][-1]
        cord = resample_poly(P, 90 if hi else 34)
        V, F, sm = tube(cord, STR_R, 12 if hi else 6, squash=STR_SQ)
        parts.append((V, F, sm, R_STRING, tube.coords))
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        at = lambda d: np.stack([np.interp(d, s, P[:, k]) for k in range(3)], -1)
        kc = at(L - 3.0); kt = at(L - 2.6) - at(L - 3.4)
        kn = trefoil(kc, kt, 0.15, 72 if hi else 24)
        V, F, sm = tube(kn, 0.22, 8 if hi else 5, closed=True)
        parts.append((V, F, sm, R_KNOT, tube.coords))
        ag = np.stack([at(d) for d in np.linspace(L - 2.1, L, 8 if hi else 4)])
        ag[-1] += (ag[-1] - ag[-2]) / np.linalg.norm(ag[-1] - ag[-2]) * 0.1
        V, F, sm = tube(ag, 0.31, 16 if hi else 6)
        parts.append((V, F, sm, R_AGLET, tube.coords))
    return parts


def grommet_parts(eyelets, hi=False):
    parts = []
    for p, n in eyelets:
        prof = np.array(GROMMET)
        if hi:                                                # densify the profile for the bake source
            prof = np.concatenate([np.linspace(a, b, 4, endpoint=False) for a, b in zip(prof, np.roll(prof, -1, 0))])
        V, F, sm = lathe(prof, p, n, 36 if hi else 10)
        parts.append((V, F, sm, R_GROM, np.zeros((len(V), 2))))
    return parts


# ----------------------------------------------------------------------------------------------------------- build
def build(level, eyelets=None, paths=None, log=print):
    """level 2 = game mesh, 4 = bake source. Returns merged mesh dict + extras."""
    hi = level > 2
    mult = 2 ** (level - 2)
    body = body_mesh(level)
    hood = hood_mesh(body, mult)
    nb = len(body["V"])
    V = np.concatenate([body["V"], hood["V"]]); F = np.concatenate([body["F"], hood["F"]])
    reg = np.concatenate([body["reg"], hood["reg"]])
    seams = set(body["seams"]) | set(hood["seams"])
    sl = np.r_[body["at"]["sl"], np.zeros(len(hood["V"]))]
    d = np.zeros(len(V))
    d[:nb] = folds(V[:nb], sl[:nb], hi)
    hp = hood["params"]
    d[nb:] = folds(V[nb:], sl[nb:], hi, hood=(hp["phi"], hp["u"], hp["w"]))
    if not hi:                                   # the game mesh can't carry 3 cm folds: keep their low-pass, bake the rest
        E = edges_of(F)[0]
        deg = np.bincount(E.ravel(), minlength=len(V)).astype(float)
        for _ in range(3):
            acc = np.zeros(len(V)); np.add.at(acc, E[:, 0], d[E[:, 1]]); np.add.at(acc, E[:, 1], d[E[:, 0]])
            d = 0.5 * d + 0.5 * acc / np.maximum(deg, 1)
    # channel: rounded bulge between the stitch row and the edge
    u_ch = np.zeros(len(V)); rows = hood["idx"].shape[1]
    for j in range(hood["idx"].shape[0]):
        for r in range(hood["chan_row"] + 1, rows - 1):
            u_ch[hood["idx"][j, r]] = 1
    d += 0.22 * u_ch
    V = displace(V, F, d)
    thick = vertex_thickness(len(V), F, reg)
    V, npush = clear_body(V, F, thick, mask=np.r_[np.ones(nb, bool), np.zeros(len(hood["V"]), bool)])
    log(f"  level {level}: body {len(body['F'])} + hood {len(hood['F'])} quads, {npush} verts pushed clear of the body")
    pk = pocket_mesh(mult)
    # eyelets from the game-mesh hood (column 2 / n-2, middle of the channel)
    idx, cr = hood["idx"], hood["chan_row"]
    if eyelets is None:
        vn = vertex_normals(V, F)
        eyelets = []
        nn = idx.shape[0] - 1
        for j in (2 * mult, nn - 2 * mult):
            a, b = idx[j, cr], idx[j, rows - 1]
            p = (V[a] + V[b]) / 2; n = vn[a] + vn[b]; n /= np.linalg.norm(n)
            eyelets.append((p + n * 0.0, n))
    if paths is None:
        paths = string_paths(V, F, eyelets)
    # shells
    centre_col = idx.shape[0] // 2
    Vs, Fs, Rs, Is, Ss = solidify(V, F, reg, seams, thick, cut_verts=[idx[0, 0], idx[centre_col, rows - 1]])
    pkreg = np.full(len(pk["F"]), R_POCKET)
    pidx = pk["idx"]
    Vp, Fp, Rp, Ip, Sp = solidify(pk["V"], pk["F"], pkreg, set(), np.full(len(pk["V"]), T_POCKET),
                                  cut_verts=[pidx[0, 0], pidx[-1, 0], pidx[0, -1], pidx[-1, -1]])
    parts = [(Vs, Fs, Ss, Rs, Is, np.zeros((len(Vs), 2))), (Vp, Fp, Sp, Rp, Ip, np.zeros((len(Vp), 2)))]
    for Vx, Fx, Sx, rx, cx in grommet_parts(eyelets, hi) + string_parts(paths, hi):
        parts.append((Vx, Fx, Sx, [rx] * len(Fx), [3] * len(Fx), cx))
    # merge
    allV, allF, allR, allI, allS, allC, ranges, off = [], [], [], [], set(), [], [], 0
    for Vx, Fx, Sx, Rx, Ix, Cx in parts:
        allV.append(np.asarray(Vx)); allF += [tuple(int(i) + off for i in f) for f in Fx]
        allR += list(Rx); allI += list(Ix); allS |= {(a + off, b + off) for a, b in Sx}
        allC.append(np.asarray(Cx)); ranges.append((off, off + len(Vx), int(Rx[0])))
        off += len(Vx)
    sew = dict(body["seams"]); sew.update(hood["seams"])
    # pocket outline: sewn edges vs the two openings (in pocket-local indices, offset to the merged mesh)
    pk_off = len(Vs)
    pi_ = pk["idx"]; ns_, nv_ = pi_.shape[0] - 1, pi_.shape[1] - 1
    pk_sew = {}
    for i in range(ns_):
        pk_sew[(pi_[i, 0] + pk_off, pi_[i + 1, 0] + pk_off)] = "pocket"
        pk_sew[(pi_[i, nv_] + pk_off, pi_[i + 1, nv_] + pk_off)] = "pocket"
    for j in range(nv_):
        z = pk["V"][pi_[0, j], 2]
        t = "pocket" if z < PK_ZO - 0.3 else "opening"
        pk_sew[(pi_[0, j] + pk_off, pi_[0, j + 1] + pk_off)] = t
        pk_sew[(pi_[ns_, j] + pk_off, pi_[ns_, j + 1] + pk_off)] = t
    return dict(V=np.concatenate(allV), F=allF, reg=np.array(allR), inner=np.array(allI), seams=allS,
                eyelets=eyelets, paths=paths, outer=(V, F), body=body, hood=hood, pocket=pk, sew=sew, pk_sew=pk_sew,
                n_outer=len(V), sl=sl, hood_base=nb, ranges=ranges, tube_uv=np.concatenate(allC))
