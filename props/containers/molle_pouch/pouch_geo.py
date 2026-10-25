"""6x6 utility pouch geometry (bpy + numpy). Units mm in here, metres in Blender. Front = -Y, pivot = base centre.

Body: a 152 x 76 x 152 mm rounded box described in *cube parameters* C (a point on the axis-aligned box surface).
  base(C)      -> point on the rounded box (edge radius R) + its normal
  puff(C)      -> low-frequency pillowing (the game mesh carries this)
  detail(C)    -> seams (piping bead + pinch), zipper groove, webbing compression, folds, noise (high only)
Seams sit on the cube edges (= the 45 deg line of the rounded edges), so the box-projected UV seams follow the sewn seams.

Clamshell: the zipper runs up the side gussets' centre line (y = 0) from ZCUT, over the top. SM_GPPouch_Flap is
everything in front of the zipper and above ZCUT (front panel above the fold line hidden under PALS row 2); it hinges
forward about X at HINGE. Every part is built twice from the same functions: high (bake source) and low (game mesh)."""
import bpy, bmesh, math
import numpy as np
from mathutils import Vector, Matrix

MM = 0.001
HX, HY, HZ = 76.0, 38.0, 76.0          # half extents; the body spans z 0..152
R = 11.0                               # edge radius of the base box
WALL = 4.0                             # padded wall (liner offset)
ZCUT = 69.0                            # zipper ends / side seam / flap fold line
HINGE = (0.0, -41.0, ZCUT)             # flap pivot (front face of the webbing at the fold line)
PALS_Z = (30.0, 82.0, 132.0)           # front webbing row centres
WEB = 25.0                             # webbing width
WEB_T = 1.5                            # webbing thickness
TACK_F = (-76.0, -38.0, 0.0, 38.0, 76.0)          # front bar-tacks (cube-param x; +-76 = in the seam)
GRID_Z = (52.0, 125.0)                 # back grid (3 contiguous rows)
TACK_B = (-66.0, -33.0, 0.0, 33.0, 66.0)
STRAP_X = (-50.0, 50.0)
SNAP_Z = 40.0
LOOP_Z = (49.0, ZCUT)                  # side pull loop
SLIDERS = (-15.0, 15.0)                # slider centres along the top zipper (x)
CAPS = ((-52.0, 13.0, -0.35), (50.0, 15.0, 0.30))   # cord end caps on the rear half of the top: (x, y, yaw rad)

# ID colours (baked from the highs; the texture pass classifies texels by nearest colour)
IDS = {"fabric": (0.5, 0.5, 0.5), "webh": (1, 0, 0), "webv": (0, 1, 0), "tape": (0, 0, 1), "teeth": (1, 1, 0),
       "slider": (1, 0, 1), "cord": (0, 1, 1), "cap": (1, 0.5, 0), "stud": (0.3, 0.6, 0.3), "stiff": (0.5, 0, 1)}


# ------------------------------------------------------------------------------------------------ body fields
def labels(C):
    """cube face of each param point: 0 front(-y) 1 back(+y) 2 -x 3 +x 4 top 5 bottom"""
    x, y, z = C[:, 0], C[:, 1], C[:, 2] - HZ
    r = np.stack([np.abs(x) / HX, np.abs(y) / HY, np.abs(z) / HZ], 1)
    ax = np.argmax(r, 1)
    lab = np.where(ax == 1, np.where(y < 0, 0, 1), np.where(ax == 0, np.where(x < 0, 2, 3), np.where(z > 0, 4, 5)))
    return lab


def base(C):
    P = C - np.array([0, 0, HZ])
    h = np.array([HX - R, HY - R, HZ - R])
    Q = np.clip(P, -h, h)
    D = P - Q
    L = np.linalg.norm(D, axis=1, keepdims=True)
    N = D / np.maximum(L, 1e-9)
    return Q + N * R + np.array([0, 0, HZ]), N


def seam_dists(C):
    """geodesic-ish (cube-param) distances to: front perimeter seam, back perimeter seam, zipper line, side seam"""
    x, y, z = C[:, 0], C[:, 1], C[:, 2]
    lab = labels(C)
    big = 1e3
    face_edge = np.minimum.reduce([HX - np.abs(x), z, 2 * HZ - z])
    dF = np.where(lab == 0, face_edge, np.where(lab == 1, big, y + HY))
    dB = np.where(lab == 1, face_edge, np.where(lab == 0, big, HY - y))
    side = (lab == 2) | (lab == 3)
    dZ = np.where(side, np.where(z >= ZCUT, np.abs(y), np.hypot(y, ZCUT - z)), np.where(lab == 4, np.abs(y), big))
    dS = np.where(side, np.abs(z - ZCUT), big)
    return lab, dF, dB, dZ, dS


def _f(a, e=5):
    return 1 - np.clip(np.abs(a), 0, 1) ** e


def puff(C, N):
    x, y, z = C[:, 0], C[:, 1], C[:, 2] - HZ
    lab, dF, dB, dZ, dS = seam_dists(C)
    terms = ((1, -1, 0.8, _f(x / HX) * _f(z / HZ)), (1, 1, 0.6, _f(x / HX) * _f(z / HZ)),
             (0, -1, 1.6, _f(y / HY) * _f(z / HZ)), (0, 1, 1.6, _f(y / HY) * _f(z / HZ)),
             (2, 1, 2.6, _f(x / HX) * _f(y / HY)), (2, -1, 0.5, _f(x / HX) * _f(y / HY)))
    out = np.zeros(len(C))
    for ax, sg, amp, f in terms:
        w = np.clip(N[:, ax] * sg, 0, 1) ** 2
        out += w * amp * f
    out *= 1 - np.exp(-(dZ / 9.0) ** 2)                 # the zipper pulls each half of the gusset in
    out *= 1 - 0.35 * np.exp(-(dS / 6.0) ** 2)
    return out


def surface(C, high=False, extra=None):
    """positions + normals of the pouch skin at cube params C (N,3)"""
    C = np.asarray(C, float).reshape(-1, 3)
    B, N = base(C)
    d = puff(C, N)
    if high:
        d = d + detail(C, N)
    if extra is not None:
        d = d + extra
    return B + N * d[:, None], N


def _bead(d, h=0.55, s=0.75, v=0.28, vo=2.4, vs=1.0):
    return h * np.exp(-(d / s) ** 2) - v * np.exp(-((d - vo) / vs) ** 2)


_RNG = np.random.default_rng(7)
_WAVES = [(_RNG.normal(size=3), _RNG.uniform(0, 6.3)) for _ in range(14)]


def wnoise(C, scale, seed=0):
    out = np.zeros(len(C))
    r = np.random.default_rng(seed)
    for k in range(10):
        d = r.normal(size=3); d /= np.linalg.norm(d)
        out += np.sin(C @ d * (2 * np.pi / scale) * r.uniform(0.7, 1.4) + r.uniform(0, 6.3))
    return out / math.sqrt(10 / 2)


def folds():
    """fold ridges (start, dir, length, width, amp) in cube-param space, seeded from the seams"""
    r = np.random.default_rng(11)
    out = []
    def add(p, d, n, L=(10, 30), w=(1.6, 3.2), a=(0.35, 0.85), jitter=0.45):
        for _ in range(n):
            t = r.uniform(0, 1)
            s = p(t)
            ang = r.normal(0, jitter)
            dd = np.array(d(t), float)
            # rotate dd within the face plane: find the face normal = the axis dd and the seam both lack
            out.append((s, dd, ang, r.uniform(*L), r.uniform(*w), r.uniform(*a) * r.choice([1, 1, -0.6])))
    # front panel from its four edges, inward
    add(lambda t: (-HX, -HY, 6 + 140 * t), lambda t: (1, 0, 0), 7)
    add(lambda t: (HX, -HY, 6 + 140 * t), lambda t: (-1, 0, 0), 7)
    add(lambda t: (-70 + 140 * t, -HY, 2 * HZ), lambda t: (0, 0, -1), 6, L=(8, 18))
    add(lambda t: (-70 + 140 * t, -HY, 0), lambda t: (0, 0, 1), 5, L=(8, 16))
    # gussets: from front and back seams toward the zipper, and from the zipper outward
    for sx in (-HX, HX):
        add(lambda t, sx=sx: (sx, -HY, 4 + 144 * t), lambda t: (0, 1, 0), 9, L=(8, 26))
        add(lambda t, sx=sx: (sx, HY, 4 + 144 * t), lambda t: (0, -1, 0), 8, L=(8, 24))
        add(lambda t, sx=sx: (sx, 0, ZCUT + 6 + 76 * t), lambda t: (0, r.choice([-1, 1]), 0), 10, L=(6, 16), w=(1.2, 2.4), a=(0.3, 0.7))
        add(lambda t, sx=sx: (sx, -HY + 76 * t, 2), lambda t: (0, 0, 1), 4, L=(10, 30), a=(0.4, 0.9))
    # top: along both seams and from the zipper
    add(lambda t: (-72 + 144 * t, -HY, 2 * HZ), lambda t: (0, 1, 0), 9, L=(8, 22))
    add(lambda t: (-72 + 144 * t, HY, 2 * HZ), lambda t: (0, -1, 0), 8, L=(8, 22))
    add(lambda t: (-72 + 144 * t, 0, 2 * HZ), lambda t: (0, r.choice([-1, 1]), 0), 10, L=(6, 14), w=(1.2, 2.2), a=(0.3, 0.6))
    # back panel corners
    add(lambda t: (r.choice([-HX, HX]), HY, 4 + 144 * t), lambda t: (0, 0, 0), 0)
    return out


_FOLDS = None


def fold_field(C):
    global _FOLDS
    if _FOLDS is None:
        _FOLDS = folds()
    out = np.zeros(len(C))
    for s, d, ang, L, w, a in _FOLDS:
        s = np.array(s, float)
        d = d / max(np.linalg.norm(d), 1e-9)
        # the face the fold runs into: its normal is the axis where s sits on the box and d is zero
        n = np.zeros(3)
        ax = int(np.argmax(np.abs(np.array([s[0] / HX, s[1] / HY, (s[2] - HZ) / HZ])) * (np.abs(d) < 0.5)))
        n[ax] = 1
        d = d * math.cos(ang) + np.cross(n, d) * math.sin(ang)
        v = C - s
        along = v @ d
        perp = np.linalg.norm(v - along[:, None] * d, axis=1)
        m = (along > -2) & (along < L + 2) & (perp < 4 * w)
        if not m.any():
            continue
        t = np.clip(along[m] / L, 0, 1)
        taper = np.sin(np.pi * np.clip(t * 0.85 + 0.15, 0, 1)) ** 0.7 * (1 - t) ** 0.5
        q = (perp[m] / w) ** 2
        out[m] += a * (1 - 0.6 * q) * np.exp(-q) * taper
    return out


def detail(C, N):
    """high-frequency skin: seam beads, zipper groove, PALS compression, folds, noise (mm along N)"""
    lab, dF, dB, dZ, dS = seam_dists(C)
    x, y, z = C[:, 0], C[:, 1], C[:, 2]
    d = _bead(dF) + _bead(dB)
    d += _bead(dS, 0.5, 1.0, 0.35, 2.8, 1.2)
    zip_ = np.exp(-(dZ / 25.0) ** 2) > 1e-3
    gz = -2.8 * np.exp(-(dZ / 1.9) ** 2) + 1.0 * np.exp(-((dZ - 3.9) / 1.6) ** 2)
    gz += -0.22 * (np.exp(-((dZ - 9.0) / 0.7) ** 2) + np.exp(-((dZ - 12.5) / 0.7) ** 2))
    d += np.where(zip_, gz, 0)
    front = lab == 0
    band = np.zeros(len(C))
    for zc in PALS_Z:
        band = np.maximum(band, np.clip((WEB / 2 + 1.0 - np.abs(z - zc)) / 2.0, 0, 1))
    between = np.clip(1 - band, 0, 1)
    d += np.where(front, -0.35 * band + 0.15 * between * _f(x / HX, 4), 0)
    back = lab == 1
    gb = np.clip((np.minimum(z - GRID_Z[0], GRID_Z[1] - z) + 1.0) / 2.0, 0, 1) * (np.abs(x) < 69)
    d += np.where(back, -0.45 * gb, 0)
    d += 0.2 * wnoise(C, 46, 3) + 0.04 * wnoise(C, 14, 4)
    d += 0.6 * fold_field(C)
    return d


# ------------------------------------------------------------------------------------------------ mesh helpers
def cube_grid(xs, ys, zs):
    """welded quad grid over the box surface. Returns verts (N,3) cube params, faces [[i,j,k,l]], face labels"""
    keys, verts, faces, flab = {}, [], [], []

    def vid(p):
        k = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        i = keys.get(k)
        if i is None:
            i = keys[k] = len(verts)
            verts.append(p)
        return i
    X0, X1, Y0, Y1, Z0, Z1 = xs[0], xs[-1], ys[0], ys[-1], zs[0], zs[-1]
    spec = [(0, lambda a, b: (a, Y0, b), xs, zs, (0, -1, 0)), (1, lambda a, b: (a, Y1, b), xs, zs, (0, 1, 0)),
            (2, lambda a, b: (X0, a, b), ys, zs, (-1, 0, 0)), (3, lambda a, b: (X1, a, b), ys, zs, (1, 0, 0)),
            (4, lambda a, b: (a, b, Z1), xs, ys, (0, 0, 1)), (5, lambda a, b: (a, b, Z0), xs, ys, (0, 0, -1))]
    for lab, fn, A, Bs, n in spec:
        idx = [[vid(fn(a, b)) for b in Bs] for a in A]
        pa, pb = np.array(fn(A[1], Bs[0])) - np.array(fn(A[0], Bs[0])), np.array(fn(A[0], Bs[1])) - np.array(fn(A[0], Bs[0]))
        out = np.cross(pa, pb) @ np.array(n) > 0
        for i in range(len(A) - 1):
            for j in range(len(Bs) - 1):
                q = [idx[i][j], idx[i + 1][j], idx[i + 1][j + 1], idx[i][j + 1]]
                faces.append(q if out else q[::-1])
                flab.append(lab)
    return np.array(verts, float), faces, flab


def mesh_obj(name, verts_mm, faces, smooth=True, col=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata((np.asarray(verts_mm) * MM).tolist(), [], faces)
    me.validate(clean_customdata=False)
    o = bpy.data.objects.new(name, me)
    (col or bpy.context.scene.collection).objects.link(o)
    if smooth:
        me.shade_smooth()
    return o


def set_attr(o, name, values):
    """per-vertex float colour attribute (3 floats stored in rgb)"""
    a = o.data.color_attributes.new(name, "FLOAT_COLOR", "POINT")
    v = np.zeros((len(o.data.vertices), 4), np.float32)
    v[:, :3] = values
    v[:, 3] = 1
    a.data.foreach_set("color", v.ravel())


def set_uv(o, per_loop):
    """per_loop: (n_loops, 2) array"""
    me = o.data
    uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.asarray(per_loop, np.float32).ravel())


def frames(P, side_hint):
    """per-station tangent, side, up for a path; side_hint (N,3) or (3,)"""
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    S = np.broadcast_to(np.asarray(side_hint, float), P.shape).copy()
    S -= (S * T).sum(1, keepdims=True) * T
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)
    U = np.cross(T, S)
    return T, S, U


def transport_frames(P):
    """parallel-transport frames for tubes"""
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    S = [np.cross(T[0], ref)]
    S[0] /= np.linalg.norm(S[0])
    for i in range(1, len(P)):
        s = S[-1] - (S[-1] @ T[i]) * T[i]
        S.append(s / np.linalg.norm(s))
    S = np.array(S)
    return T, S, np.cross(T, S)


def rrect(w, t, rc, n=3):
    """rounded-rectangle profile (a across, b up), bottom edge at b=0, CCW, starting mid-bottom.
    Returns points and a per-segment 'hidden' flag (bottom segments)."""
    rc = min(rc, t / 2 - 1e-3, w / 2 - 1e-3)
    if n == 0 or rc <= 0:
        pts = [(0.0, 0.0), (w / 2, 0.0), (w / 2, t), (-w / 2, t), (-w / 2, 0.0)]
        return pts, [True, False, False, False, True]
    pts = []
    cs = [(w / 2 - rc, rc, -90), (w / 2 - rc, t - rc, 0), (-w / 2 + rc, t - rc, 90), (-w / 2 + rc, rc, 180)]
    for cx, cy, a0 in cs:
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + rc * math.cos(a), cy + rc * math.sin(a)))
    pts = [(0.0, 0.0)] + pts
    hid = [abs(pts[i][1]) < 1e-6 and abs(pts[(i + 1) % len(pts)][1]) < 1e-6 for i in range(len(pts))]
    return pts, hid


def circle(r, n):
    pts = [(r * math.cos(2 * math.pi * k / n - math.pi / 2), r * math.sin(2 * math.pi * k / n - math.pi / 2)) for k in range(n)]
    return pts, [False] * n


def sweep(name, P, S, U, prof, hid=None, caps=True, attr=None, uv=True, hid_scale=0.15, col=None):
    """Sweep a closed profile (a along S, b along U) through centres P. attr(station, a, b) -> (N,3) values (high)."""
    P, S, U = (np.asarray(v, float) for v in (P, S, U))
    nP, nK = len(P), len(prof)
    A = np.array([p[0] for p in prof]); Bv = np.array([p[1] for p in prof])
    V = P[:, None, :] + S[:, None, :] * A[None, :, None] + U[:, None, :] * Bv[None, :, None]
    verts = V.reshape(-1, 3)
    faces = []
    for i in range(nP - 1):
        for k in range(nK):
            k1 = (k + 1) % nK
            faces.append([i * nK + k, i * nK + k1, (i + 1) * nK + k1, (i + 1) * nK + k])
    nside = len(faces)
    if caps:
        faces.append([k for k in range(nK)][::-1])
        faces.append([(nP - 1) * nK + k for k in range(nK)])
    o = mesh_obj(name, verts, faces, col=col)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    if attr is not None:
        st = np.repeat(np.arange(nP), nK); aa = np.tile(A, nP); bb = np.tile(Bv, nP)
        set_attr(o, "lab_p", attr(st, aa, bb))
    if uv:
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        hid = hid or [False] * nK
        seg = [math.dist(prof[k], prof[(k + 1) % nK]) for k in range(nK)]
        # visible chain and hidden chain cumulative v
        vpos, hpos, cv, ch = {}, {}, 0.0, 0.0
        for k in range(nK):
            if hid[k]:
                hpos[k] = ch; ch += seg[k]
            else:
                vpos[k] = cv; cv += seg[k]
        me = o.data
        loops = np.zeros((len(me.loops), 2))
        for fi, f in enumerate(me.polygons):
            if fi >= nside:                                  # caps: planar a/b, own island far away
                for li in f.loop_indices:
                    vi = me.loops[li].vertex_index
                    loops[li] = (5000 + (fi - nside) * 400 + A[vi % nK] * hid_scale * 4, 5000 + Bv[vi % nK] * hid_scale * 4)
                continue
            i, k = divmod(fi, nK)
            for li in f.loop_indices:
                vi = me.loops[li].vertex_index
                st, kk = divmod(vi, nK)
                if hid[k]:
                    off = hpos[k] + (seg[k] if kk != k else 0)
                    loops[li] = (s[st] * hid_scale + 3000, off * hid_scale + 3000)
                else:
                    off = vpos[k] + (seg[k] if kk != k else 0)
                    loops[li] = (s[st], off)
        set_uv(o, loops)
    return o


def solid_box(name, w, d, h, r=0.0, segs=2, bevel=0.0, center=(0, 0, 0), col=None):
    """rounded box (mm) centred at center; r rounds the vertical edges"""
    o = __import__("lab.geo", fromlist=["geo"]).rounded_box(name, w, d, h, r, -h / 2, bevel, segs=segs)
    o.location = Vector(center) * MM
    bake_xform(o)
    if col:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    return o


def bake_xform(o):
    bpy.context.view_layer.update()
    o.data.transform(o.matrix_world)
    o.matrix_world = Matrix()
    return o


# ------------------------------------------------------------------------------------------------ the parts
def pals_front(k, high):
    """front PALS row k: seam to seam, bowed between bar-tacks, ends dive into the seams"""
    zc = PALS_Z[k]
    n = 609 if high else 33
    xs = np.linspace(-HX, HX, n)
    if not high:                                          # stations at the tacks and channel middles
        xs = np.unique(np.concatenate([np.linspace(-HX, HX, 33), TACK_F]))
    C = np.stack([xs, np.full_like(xs, -HY), np.full_like(xs, zc)], 1)
    Sf, N = surface(C)
    u = np.mod(xs + HX, 38.0) / 38.0
    bow = 0.1 + 0.32 * np.sin(np.pi * u) ** 0.8
    endd = np.clip((np.abs(xs) - 70.0) / 6.0, 0, 1)
    off = bow * (1 - endd) - 1.6 * endd
    P = Sf + N * off[:, None]
    T, S, U = frames(P, (0, 0, 1))
    if np.mean(U @ np.array([0, -1, 0])) < 0:
        S = -S; U = -U
    prof, hid = rrect(WEB, WEB_T, 0.55 if high else 0.0, 3 if high else 0)
    attr = (lambda st, a, b: np.stack([xs[st], a, np.full_like(a, 1.0 + k)], 1)) if high else None
    return sweep(f"pals{k}", P, S, U, prof, hid, attr=attr, uv=not high)


def back_grid(high):
    """3 contiguous horizontal rows over the vertical straps, bar-tacked into 4 columns"""
    rows = []
    zs = np.linspace(GRID_Z[0], GRID_Z[1], 4)
    for r in range(3):
        zc = (zs[r] + zs[r + 1]) / 2
        w = zs[r + 1] - zs[r]
        xs = np.linspace(-68, 68, 545) if high else np.unique(np.concatenate([np.linspace(-68, 68, 17), TACK_B, [x + s for x in STRAP_X for s in (-14, 14)]]))
        C = np.stack([xs, np.full_like(xs, HY), np.full_like(xs, zc)], 1)
        Sf, N = surface(C)
        over = sum(np.clip((15.5 - np.abs(xs - sx)) / 3.0, 0, 1) for sx in STRAP_X) * 1.7
        u = np.mod(xs + 66.0, 33.0) / 33.0
        bow = 0.1 + 0.25 * np.sin(np.pi * u) ** 0.8
        endd = np.clip((np.abs(xs) - 66.0) / 2.0, 0, 1)
        off = (bow + over) * (1 - endd) - 0.5 * endd
        P = Sf + N * off[:, None]
        T, S, U = frames(P, (0, 0, 1))
        if np.mean(U @ np.array([0, 1, 0])) < 0:
            S = -S; U = -U
        prof, hid = rrect(w + (0.0 if high else 0.02), WEB_T, 0.5 if high else 0.0, 3 if high else 0)
        attr = (lambda st, a, b, xs=xs, r=r: np.stack([xs[st], a, np.full_like(a, 4.0 + r)], 1)) if high else None
        rows.append(sweep(f"grid{r}", P, S, U, prof, hid, attr=attr, uv=not high))
    return rows


def strap_path(sx, high):
    """vertical strap centre-line (on the back), up past the top, folded over into a loop tab"""
    zs = np.linspace(4, 141, 275 if high else 15)
    C = np.stack([np.full_like(zs, sx), np.full_like(zs, HY), zs], 1)
    Sf, N = surface(C)
    off = np.where(zs < 8, -2.6 + (zs - 4) / 4 * 2.8, 0.2)
    P = list(Sf + N * off[:, None])
    y_top = P[-1][1]
    zt = 167.0
    for z in np.linspace(141, zt - 2.0, 52 if high else 5)[1:]:
        P.append(np.array([sx, y_top + 0.15 * (z - 141) / 24, z]))
    rr = 1.0 + WEB_T / 2
    yc = P[-1][1] - rr
    for a in np.linspace(0, math.pi, 13 if high else 5)[1:]:
        P.append(np.array([sx, yc + rr * math.cos(a), zt - 2.0 + rr * math.sin(a)]))
    for z in np.linspace(zt - 2.0, 147.0, 40 if high else 4)[1:]:
        P.append(np.array([sx, yc - rr, z]))
    return np.array(P)


def strap(i, high):
    sx = STRAP_X[i]
    P = strap_path(sx, high)
    T, S, U = frames(P, (1, 0, 0))
    # the outward side of the first stretch must face +y
    if U[5] @ np.array([0, 1, 0]) < 0:
        S = -S; U = -U
    prof, hid = rrect(WEB, WEB_T, 0.55 if high else 0.0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b: np.stack([s[st], a, np.full_like(a, 7.0 + i)], 1)) if high else None
    return sweep(f"strap{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def snap_tab(i, high):
    sx = STRAP_X[i]
    zs = np.linspace(53, 27, 60 if high else 4)
    C = np.stack([np.full_like(zs, sx), np.full_like(zs, HY), zs], 1)
    Sf, N = surface(C)
    off = np.full_like(zs, 0.2 + WEB_T - 0.15)
    off[zs > 50] = 0.2 + WEB_T - 0.15 - (zs[zs > 50] - 50) / 3 * 1.0     # tucks under the grid
    P = Sf + N * off[:, None]
    T, S, U = frames(P, (1, 0, 0))
    if U[len(U) // 2] @ np.array([0, 1, 0]) < 0:
        S = -S; U = -U
    prof, hid = rrect(WEB + 0.6, 2.6, 0.8 if high else 0.0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b: np.stack([s[st], a, np.full_like(a, 9.0 + i)], 1)) if high else None
    return sweep(f"snaptab{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def snap_top(i):
    """outer face centre of the snap tab at the stud (mm)"""
    sx = STRAP_X[i]
    Sf, N = surface(np.array([[sx, HY, SNAP_Z]]))
    return Sf[0] + N[0] * (0.2 + WEB_T - 0.15 + 2.6), N[0]


def stud(i, high):
    from lab import geo
    seg = 48 if high else 12
    prof = [(0, 0.35), (1.9, 0.35), (2.0, 1.4), (2.7, 2.0), (3.7, 2.15), (4.5, 1.7), (6.0, 1.25), (6.6, 0.8), (6.5, -0.4), (0, -0.4)]
    if not high:
        prof = [(0, 0.35), (2.0, 0.35), (2.2, 1.9), (3.8, 2.15), (6.2, 1.2), (6.6, -0.4), (0, -0.4)]
    o = geo.lathe(f"stud{i}", [dict(pts=[(r * MM, z * MM) for r, z in prof], uv=(0, 0, 1, 1))], segs=seg)
    p, n = snap_top(i)
    o.data.transform(Matrix.Rotation(-math.pi / 2, 4, "X"))
    o.location = Vector(p) * MM
    return bake_xform(o)


def stiffener(i, high):
    sx = STRAP_X[i]
    Sf, N = surface(np.array([[sx, HY, 141.0]]))
    y = Sf[0][1] + 0.2 + WEB_T + 0.15 * 9 / 24 + 1.1
    o = solid_box(f"stiff{i}", 44, 6.5, 2.6, r=0, segs=1)
    bm = bmesh.new(); bm.from_mesh(o.data)
    if high:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.8 * MM, segments=3, affect="EDGES", profile=0.5)
    else:
        bmesh.ops.bevel(bm, geom=[e for e in bm.edges if abs(e.verts[0].co.y - e.verts[1].co.y) > 1e-6], offset=1.6 * MM, segments=1, affect="EDGES")
    bm.to_mesh(o.data); bm.free()
    o.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))          # 6.5 tall in z, 2.6 thick in y
    o.location = Vector((sx, y, 150.0)) * MM
    return bake_xform(o)


def side_loop(i, high):
    """25 mm webbing pull loop on each gusset: hangs from the side seam, bar-tacked at the top"""
    sx = -HX if i == 0 else HX
    sg = -1 if i == 0 else 1
    z0, z1 = LOOP_Z
    n = 40 if high else 4
    P = []
    zs = np.linspace(z1 + 1.0, z0 + 2.4, n)
    C = np.stack([np.full_like(zs, sx), np.zeros_like(zs), zs], 1)
    Sf, N = surface(C)
    for p, nn, z in zip(Sf, N, zs):
        P.append(p + nn * (0.15 if z < z1 - 0.5 else -0.6))
    base_x = P[-1][0]
    rr = 1.4 + WEB_T / 2
    for a in np.linspace(0, math.pi, 12 if high else 4)[1:]:
        P.append(np.array([base_x + sg * (rr - rr * math.cos(a)), 0.0, z0 + 2.4 - rr * math.sin(a)]))
    xo = P[-1][0]
    zs2 = np.linspace(z0 + 2.4, z1 - 4.0, n)[1:]
    for z in zs2:
        P.append(np.array([xo - sg * 0.9 * np.clip((z - (z1 - 12)) / 8, 0, 1), 0.0, z]))
    P.append(np.array([xo - sg * 1.1, 0.0, z1 - 1.0]))
    P = np.array(P)
    T, S, U = frames(P, (0, 1, 0))
    if U[2] @ np.array([sg, 0, 0]) < 0:
        S = -S; U = -U
    prof, hid = rrect(WEB, WEB_T, 0.55 if high else 0.0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b: np.stack([s[st], a, np.full_like(a, 11.0 + i)], 1)) if high else None
    return sweep(f"sideloop{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def zip_path(step):
    """zipper centre line in cube params: up the -x gusset, over the top, down the +x gusset"""
    L1 = 2 * HZ - ZCUT
    pts = [(-HX, 0, z) for z in np.arange(ZCUT, 2 * HZ, step)]
    pts += [(x, 0, 2 * HZ) for x in np.arange(-HX, HX, step)]
    pts += [(HX, 0, z) for z in np.arange(2 * HZ, ZCUT - 1e-6, -step)]
    return np.array(pts, float)


def zipper_tape(high=True):
    C = zip_path(0.5)
    Sf, N = surface(C)
    P = Sf - N * 2.6
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 0.05]
    P, N, C = P[keep], N[keep], C[keep]
    T, S, U = frames(P, (0, 1, 0))
    U = N - (N * T).sum(1, keepdims=True) * T
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    S = np.cross(U, T)
    prof, hid = rrect(24.0, 0.6, 0.2, 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return sweep("ziptape", P, S, U, prof, hid, attr=lambda st, a, b: np.stack([s[st], a, np.full_like(a, 13.0)], 1), uv=False)


def zipper_teeth():
    """coil-zipper elements: two interleaved rows along the zipper, 1.6 mm pitch (one mesh)"""
    C = zip_path(0.25)
    Sf, N = surface(C)
    P = Sf - N * 2.0
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    T = np.gradient(P, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    bm = bmesh.new()
    pitch = 1.6
    tpl = bmesh.new()
    bmesh.ops.create_cube(tpl, size=1.0)
    bmesh.ops.bevel(tpl, geom=list(tpl.edges), offset=0.22, segments=2, affect="EDGES", profile=0.6)
    tv = [v.co.copy() for v in tpl.verts]
    tf = [[v.index for v in f.verts] for f in tpl.faces]
    tpl.free()
    for side in (-1, 1):
        for sv in np.arange(pitch * (0.25 if side < 0 else 0.75), s[-1] - 0.5, pitch):
            i = int(np.searchsorted(s, sv))
            i = min(max(i, 1), len(P) - 2)
            t = T[i]; n = N[i]
            n = n - (n @ t) * t; n /= np.linalg.norm(n)
            a = np.cross(n, t)
            c = P[i] + a * side * 1.1
            # element: 3.0 across (a), 1.05 along (t), 1.25 thick (n)
            M = np.stack([a * 2.6, t * 1.05, n * 1.2], 1)
            vs = [bm.verts.new(((M @ np.array(v)) + c) * MM) for v in tv]
            for f in tf:
                bm.faces.new([vs[k] for k in f])
    me = bpy.data.meshes.new("teeth"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("teeth", me); bpy.context.scene.collection.objects.link(o)
    me.shade_smooth()
    return o


def top_point(x, y):
    Sf, N = surface(np.array([[x, y, 2 * HZ]]))
    return Sf[0], N[0]


def slider(i, high):
    """slider on the top zipper, nose toward the centre, bail at the rear. Plate 24 x 11 x 3, body 5 mm tall."""
    x = SLIDERS[i]
    sg = 1 if i == 0 else -1                              # nose direction (+x for the left slider)
    p, n = top_point(x, 0)
    zb = p[2] - 1.2
    parts = []
    from lab import geo
    body = geo.rounded_box(f"sl{i}", 24, 11, 4.2, 2.2 if high else 1.5, zb * 1.0, 0.8 if high else 0.0, segs=4 if high else 1)
    body.location = (x * MM, 0, 0); bake_xform(body)
    bm = bmesh.new(); bm.from_mesh(body.data)
    for v in bm.verts:                                     # taper: the nose is narrower and lower
        t = ((v.co.x / MM - x) * sg + 12) / 24             # 0 rear .. 1 nose
        v.co.y *= 1 - 0.32 * t
        if v.co.z / MM > zb + 2:
            v.co.z -= 0.9 * t * MM
    bm.to_mesh(body.data); bm.free()
    parts.append(body)
    # bail: an arch on top at the rear
    bx = x - sg * 6.0
    if high:
        tor = bpy.data.meshes.new("bail")
        bm = bmesh.new()
        rings, nr, nt = [], 24, 10
        for k in range(nr + 1):
            a = math.pi * k / nr
            c = np.array([bx + 4.2 * math.cos(a) * sg, 0, zb + 4.2 + 3.2 * math.sin(a)])
            tt = np.array([-math.sin(a) * sg, 0, math.cos(a)])
            nn = np.array([math.cos(a) * sg, 0, math.sin(a)])
            bb = np.cross(tt, nn)
            rings.append([bm.verts.new(((c + 1.0 * math.cos(2 * math.pi * j / nt) * nn + 1.6 * math.sin(2 * math.pi * j / nt) * bb)) * MM) for j in range(nt)])
        for k in range(nr):
            for j in range(nt):
                bm.faces.new((rings[k][j], rings[k][(j + 1) % nt], rings[k + 1][(j + 1) % nt], rings[k + 1][j]))
        bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(tor); bm.free()
        b = bpy.data.objects.new(f"bail{i}", tor); bpy.context.scene.collection.objects.link(b)
        tor.shade_smooth()
    else:
        b = geo.rounded_box(f"bail{i}", 10.0, 3.2, 3.6, 0, zb + 3.6, 0, segs=1)
        b.location = (bx * MM, 0, 0); bake_xform(b)
    parts.append(b)
    return parts


def cord_path(i, high):
    """paracord pull: a loop from the slider bail to the end cap lying on the top's rear half"""
    x = SLIDERS[i]
    sg = 1 if i == 0 else -1
    zb = top_point(x, 0)[0][2] - 1.2
    bail = np.array([x - sg * 6.0, 0.0, zb + 6.0])
    cx, cy, yaw = CAPS[i]
    cp, cn = top_point(cx, cy)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0]) * (1 if i == 1 else -1)    # cap axis, pointing away from the cord
    cap_c = cp + np.array([0, 0, 4.0])
    cord_end = cap_c - d * 6.0                            # cords enter the cap's inner end
    n = 90 if high else 16
    pts = []
    L = np.linalg.norm(bail - cord_end)
    ax = (bail - cord_end) / L
    lat = np.cross(ax, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
    for t in np.linspace(0.12, 2 * math.pi - 0.12, n):
        s = (1 - math.cos(t)) / 2                         # 0 at the cap, 1 at the bail
        w = 2.4 * math.sin(t) * (0.35 + 0.65 * s)          # strands spread a little toward the bail
        arch = 14.0 * math.sin(math.pi * s) ** 1.1 + 3.0 * s
        pts.append(cord_end + ax * (L * s) + lat * w + np.array([0, 0, arch * (1 if s < 0.97 else 1)]))
    return np.array(pts), cap_c, d


def cord(i, high):
    P, _, _ = cord_path(i, high)
    T, S, U = transport_frames(P)
    prof, hid = circle(1.9, 16 if high else 6)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = None
    if high:
        nK = len(prof)
        th = np.array([math.atan2(b, a) for a, b in prof])
        attr = lambda st, a, b: np.stack([s[st], np.tile(th, len(P)), np.full_like(a, 14.0 + i)], 1)
    return sweep(f"cord{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def cap(i, high):
    """molded cord end: 23 x 11 x 8.5 mm, tapered toward the cord, chamfered tip, grip groove"""
    from lab import geo
    _, c, d = cord_path(i, high)
    o = geo.rounded_box(f"cap{i}", 23, 11, 8.5, 3.0 if high else 2.0, -4.25, 1.0 if high else 0.0, segs=4 if high else 1)
    bm = bmesh.new(); bm.from_mesh(o.data)
    for v in bm.verts:
        t = (v.co.x / MM + 11.5) / 23                     # 0 cord end .. 1 tip
        v.co.y *= 0.82 + 0.18 * t
        v.co.z *= 0.86 + 0.14 * min(1, t * 2)
        if high and 0.62 < t < 0.70:                      # grip groove ring
            v.co.y *= 0.94; v.co.z *= 0.94
    bm.to_mesh(o.data); bm.free()
    yaw = math.atan2(d[1], d[0])
    o.rotation_euler = (0, -0.38, yaw)                   # tip lifted off the top, cord end resting on it
    o.location = (Vector(c) + Vector((0, 0, 3.2))) * MM
    bake_xform(o)
    if high:
        bm = bmesh.new(); bm.from_mesh(o.data)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True)
        bm.to_mesh(o.data); bm.free()
        o.modifiers.new("S", "SUBSURF").levels = 1
    return o


# ------------------------------------------------------------------------------------------------ body shells
def body_high():
    xs = np.linspace(-HX, HX, 254); ys = np.linspace(-HY, HY, 128); zs = np.linspace(0, 2 * HZ, 254)
    C, faces, _ = cube_grid(xs, ys, zs)
    P, _ = surface(C, high=True)
    o = mesh_obj("body_hi", P, faces)
    set_attr(o, "lab_p", C)
    return o


def edge_samples(lo, hi, flat_lo, flat_hi, step, must=()):
    """axis samples: 3 per rounded strip (equal angles), roughly `step` in the flat, always including `must`"""
    ang = [math.tan(math.radians(a)) for a in (15, 30)]
    s = [lo] + [flat_lo - R * t for t in ang[::-1]] + [flat_lo]
    pts = sorted({flat_lo, flat_hi, *[m for m in must if flat_lo < m < flat_hi]})
    for a, b in zip(pts, pts[1:]):
        k = max(1, round((b - a) / step))
        s += list(np.linspace(a, b, k + 1)[1:])
    s += [flat_hi + R * t for t in ang] + [hi]
    return np.array(sorted(set(round(v, 5) for v in s)))


def flap_face(C, lab):
    c = C.mean(0)
    if lab in (2, 3):
        return c[1] < 0 and c[2] > ZCUT
    if lab == 4:
        return c[1] < 0
    if lab == 0:
        return c[2] > ZCUT
    return False


def shell_low(step=12.0):
    """outer skin split into body / flap; returns (body_obj, flap_obj) open shells with mm cube-param UVs"""
    xs = edge_samples(-HX, HX, -HX + R, HX - R, step, (0.0,) + tuple(TACK_F) + tuple(STRAP_X))
    ys = edge_samples(-HY, HY, -HY + R, HY - R, step * 0.8, (0.0,))
    zs = edge_samples(0, 2 * HZ, R, 2 * HZ - R, step, (ZCUT,))
    C, faces, flab = cube_grid(xs, ys, zs)
    P, _ = surface(C)
    out = []
    for want in (False, True):
        sel = [k for k, f in enumerate(faces) if flap_face(C[f], flab[k]) == want]
        used = sorted({v for k in sel for v in faces[k]})
        remap = {v: i for i, v in enumerate(used)}
        o = mesh_obj("flap_shell" if want else "body_shell", P[used], [[remap[v] for v in faces[k]] for k in sel])
        Cu = C[used]
        # box-projected UVs from the cube params (mm), right-handed per face
        loops = np.zeros((len(o.data.loops), 2))
        for pi, (k, poly) in enumerate(zip(sel, o.data.polygons)):
            lab = flab[k]
            for li in poly.loop_indices:
                x, y, z = Cu[o.data.loops[li].vertex_index]
                loops[li] = {0: (x, z), 1: (-x, z), 2: (-y, z), 3: (y, z), 4: (x, y), 5: (x, -y)}[lab]
                loops[li] += (lab * 400.0, 0)
        set_uv(o, loops)
        out.append(o)
    return out
