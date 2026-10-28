"""Triple AR magazine pouch with covers (bpy + numpy). mm in here, metres in Blender. Front = -Y, pivot = base centre.

Layout (sized for three 5.56 STANAG 30-rd mags, 25 x 81 x 190 mm each, spine across X):
  back MOLLE panel (full width) carrying a 3-row PALS web on its rear face, six polymer stiffener straps woven through it
  (snap tabs + press-studs at the bottom, plastic clip + webbing loop at the top);
  three pillowed sleeves (front wall with a loop-velcro panel), each with a tall cordura back (centre stitch, eyelet);
  three flaps (moving parts): wide ribbed webbing outside, hook strip at the free end, pull tab on top.
Generic helpers (sweep, grid, frames, attributes) come from ../molle_pouch/pouch_geo.py."""
import bpy, bmesh, math, os, sys
import numpy as np
from mathutils import Vector, Matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../molle_pouch"))
import pouch_geo as PG
from pouch_geo import MM, sweep, rrect, circle, frames, transport_frames, mesh_obj, set_attr, set_uv, cube_grid, bake_xform

XC = (-91.0, 0.0, 91.0)          # sleeve centres
SHX, SHY = 43.0, 16.5            # sleeve half width / half depth
SY0 = -SHY                       # sleeve centre y (sleeve spans y -33 .. 0)
FRONT_H = 108.0                  # sleeve front wall height
BACK_H = 196.0                   # tall back top
SR = 9.0                         # sleeve edge radius
WALL = 2.6
PANEL = dict(hx=136.0, y0=0.0, y1=7.0, z0=6.0, z1=170.0)
ROWS_Z = (147.0, 101.0, 55.0)    # rear PALS rows
TACKS = tuple(np.arange(-133.35, 134, 38.1).round(2))
STRAP_X = tuple(x + s for x in XC for s in (-24.0, 24.0))
STUD_Z = 28.0
FLAP_W = 80.0
FLAP_T = 3.0
HOOK_Z = (40.0, 60.0)           # flap free end (hook strip) on the loop panel
WEB = 25.0
WEB_T = 1.5
POLY_T = 1.8                     # polymer strap thickness
MAG = dict(w=81.0, d=25.0, h=190.0, z0=4.0)

IDS = {"fabric": (0.5, 0.5, 0.5), "webh": (1, 0, 0), "webv": (0, 1, 0), "flapweb": (0, 0, 1), "loop": (1, 1, 0),
       "poly": (1, 0, 1), "clip": (0, 1, 1), "stud": (0.3, 0.6, 0.3), "eyelet": (1, 0.5, 0), "hook": (0.5, 0, 1),
       "mag": (0.2, 0.2, 0.8)}


class SoftBox:
    """rounded box with pillowed panels; cube params C are points on the axis-aligned box (mm)"""

    def __init__(self, c, h, r, puffs, front_seam=True):
        self.c, self.h, self.r, self.p = np.array(c, float), np.array(h, float), r, puffs

    def labels(self, C):
        q = (C - self.c) / self.h
        ax = np.argmax(np.abs(q), 1)
        return np.where(ax == 1, np.where(q[:, 1] < 0, 0, 1), np.where(ax == 0, np.where(q[:, 0] < 0, 2, 3), np.where(q[:, 2] > 0, 4, 5)))

    def base(self, C):
        P = C - self.c
        hi = self.h - self.r
        Q = np.clip(P, -hi, hi)
        D = P - Q
        N = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        return Q + N * self.r + self.c, N

    def seam(self, C):
        """distance (cube-param) to the front (-y face) perimeter and to the back perimeter"""
        q = C - self.c
        lab = self.labels(C)
        fe = np.minimum.reduce([self.h[0] - np.abs(q[:, 0]), self.h[2] - np.abs(q[:, 2])])
        dF = np.where(lab == 0, fe, np.where(lab == 1, 1e3, q[:, 1] + self.h[1]))
        dB = np.where(lab == 1, fe, np.where(lab == 0, 1e3, self.h[1] - q[:, 1]))
        return lab, dF, dB

    def puff(self, C, N):
        q = (C - self.c) / self.h
        f = lambda a: 1 - np.clip(np.abs(a), 0, 1) ** 5
        out = np.zeros(len(C))
        for ax, sg, a, b in ((1, -1, 0, 2), (1, 1, 0, 2), (0, -1, 1, 2), (0, 1, 1, 2), (2, 1, 0, 1), (2, -1, 0, 1)):
            amp = self.p.get((ax, sg), 0.0)
            out += np.clip(N[:, ax] * sg, 0, 1) ** 2 * amp * f(q[:, a]) * f(q[:, b])
        return out

    def surface(self, C, high=False, seed=0):
        C = np.asarray(C, float).reshape(-1, 3)
        B, N = self.base(C)
        d = self.puff(C, N)
        if high:
            lab, dF, dB = self.seam(C)
            d += PG._bead(dF, 0.7, 1.0, 0.45, 2.8, 1.2) + PG._bead(dB, 0.7, 1.0, 0.45, 2.8, 1.2)
            d += 0.35 * PG.wnoise(C, 36, 3 + seed) + 0.06 * PG.wnoise(C, 10, 4 + seed)
        return B + N * d[:, None], N


def sleeve_box(i, h_top=FRONT_H):
    return SoftBox((XC[i], SY0, h_top / 2), (SHX, SHY, h_top / 2), SR, {(1, -1): 2.0, (0, -1): 1.2, (0, 1): 1.2, (2, -1): 0.6})


def back_box(i):
    """the tall cordura back of sleeve i (above the sleeve, to BACK_H)"""
    return SoftBox((XC[i], 0.5, (FRONT_H - 30 + BACK_H) / 2), (SHX - 2.0, 3.0, (BACK_H - FRONT_H + 30) / 2), 2.9, {(1, -1): 0.8, (1, 1): 0.5})


def panel_box():
    p = PANEL
    return SoftBox((0, (p["y0"] + p["y1"]) / 2, (p["z0"] + p["z1"]) / 2), (p["hx"], (p["y1"] - p["y0"]) / 2, (p["z1"] - p["z0"]) / 2), 3.4,
                   {(1, 1): 1.2, (1, -1): 0.4})


def grid_axes(box, step, must_x=(), must_z=(), high=False):
    c, h, r = box.c, box.h, box.r
    if high:
        n = lambda L: max(4, int(round(L / step)) + 1)
        return (np.linspace(c[0] - h[0], c[0] + h[0], n(2 * h[0])), np.linspace(c[1] - h[1], c[1] + h[1], n(2 * h[1])),
                np.linspace(c[2] - h[2], c[2] + h[2], n(2 * h[2])))
    r = min(r, *h)
    ax = []
    for k, must in ((0, must_x), (1, ()), (2, must_z)):
        ax.append(PG_edge_samples(c[k] - h[k], c[k] + h[k], c[k] - h[k] + r, c[k] + h[k] - r, step, must, r))
    return ax


def PG_edge_samples(lo, hi, flat_lo, flat_hi, step, must, r):
    ang = [math.tan(math.radians(a)) for a in (22.5,)]
    s = [lo] + [flat_lo - r * t for t in ang[::-1]] + [flat_lo]
    pts = sorted({flat_lo, flat_hi, *[m for m in must if flat_lo < m < flat_hi]})
    for a, b in zip(pts, pts[1:]):
        k = max(1, round((b - a) / step))
        s += list(np.linspace(a, b, k + 1)[1:])
    s += [flat_hi + r * t for t in ang] + [hi]
    return np.array(sorted(set(round(v, 5) for v in s)))


def box_mesh(name, box, high, step, drop_top=False, seed=0, must_x=(), must_z=(), hidden=()):
    """skin of a SoftBox; drop_top removes the +z face (open sleeve). Returns obj (+ cube-param UVs in mm on the low)"""
    xs, ys, zs = grid_axes(box, 0.7 if high else step, must_x, must_z, high)
    C, faces, flab = cube_grid(xs, ys, zs)
    if drop_top:
        keep = [k for k, l in enumerate(flab) if l != 4]
        faces = [faces[k] for k in keep]; flab = [flab[k] for k in keep]
        used = sorted({v for f in faces for v in f})
        rm = {v: i for i, v in enumerate(used)}
        C = C[used]; faces = [[rm[v] for v in f] for f in faces]
    P, _ = box.surface(C, high, seed)
    o = mesh_obj(name, P, faces)
    if high:
        set_attr(o, "lab_p", C)
        return o
    loops = np.zeros((len(o.data.loops), 2))
    for poly, lab in zip(o.data.polygons, flab):
        for li in poly.loop_indices:
            x, y, z = C[o.data.loops[li].vertex_index]
            loops[li] = {0: (x, z), 1: (-x, z), 2: (-y, z), 3: (y, z), 4: (x, y), 5: (x, -y)}[lab]
            if lab in hidden:                            # faces pressed against another part: low texel density
                loops[li] *= 0.2
            loops[li] += (lab * 700.0 + seed * 4000, 0)
    set_uv(o, loops)
    return o


# ------------------------------------------------------------------------------------------------ parts
def sleeve(i, high):
    return box_mesh(f"sleeve{i}", sleeve_box(i), high, 11.0, drop_top=True, seed=i, must_x=(XC[i],), hidden=(1,))


def tall_back(i, high):
    return box_mesh(f"back{i}", back_box(i), high, 12.0, seed=10 + i, hidden=(1,))


def panel(high):
    return box_mesh("panel", panel_box(), high, 16.0, seed=20, must_x=STRAP_X, hidden=(0,))


def loop_panel(i, high):
    """loop-velcro panel on the sleeve front: thin slab hugging the front wall"""
    b = sleeve_box(i)
    zs = np.linspace(10, FRONT_H - 6, 60 if high else 5)
    xs = np.linspace(XC[i] - 37, XC[i] + 37, 2)
    out = []
    P, N = b.surface(np.stack([np.full_like(zs, XC[i]), np.full_like(zs, SY0 - SHY), zs], 1))
    P = P + N * 0.15
    T, S, U = frames(P, (1, 0, 0))
    if U[len(U) // 2] @ np.array([0, -1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(74.0, 1.3, 0.4 if high else 0, 2 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 20.0 + i)], 1)) if high else None
    return sweep(f"loop{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def web_row(k, high):
    """rear PALS row on the panel, bowed over the polymer straps where they pass underneath (rows 0 and 2)"""
    b = panel_box()
    zc = ROWS_Z[k]
    xs = np.linspace(-134, 134, 671) if high else np.unique(np.concatenate([np.linspace(-134, 134, 15), TACKS, [x + s for x in STRAP_X for s in (-14, 14)]]))
    xs = xs[(xs >= -134) & (xs <= 134)]
    P, N = b.surface(np.stack([xs, np.full_like(xs, PANEL["y1"]), np.full_like(xs, zc)], 1))
    under = k in (0, 2)
    over = sum(np.clip((15.5 - np.abs(xs - sx)) / 3.0, 0, 1) for sx in STRAP_X) * (POLY_T + 0.2) * under
    u = np.mod(xs + 133.35, 38.1) / 38.1
    bow = 0.12 + 0.5 * np.sin(np.pi * u) ** 0.8
    endd = np.clip((np.abs(xs) - 132.5) / 1.5, 0, 1)
    off = (bow + over) * (1 - endd) - 0.6 * endd
    P = P + N * off[:, None]
    T, S, U = frames(P, (0, 0, 1))
    if np.mean(U @ np.array([0, 1, 0])) < 0:
        S, U = -S, -U
    prof, hid = rrect(WEB, WEB_T, 0.5 if high else 0, 3 if high else 0)
    attr = (lambda st, a, b_: np.stack([xs[st], a, np.full_like(a, 1.0 + k)], 1)) if high else None
    return sweep(f"row{k}", P, S, U, prof, hid, attr=attr, uv=not high)


def strap_path(j, high):
    """polymer stiffener strap j: from the snap tab up the back, woven (under rows 0/2, over row 1), up through the clip"""
    sx = STRAP_X[j]
    b = panel_box()
    zs = np.linspace(14, 166, 305 if high else 21)
    P, N = b.surface(np.stack([np.full_like(zs, sx), np.full_like(zs, PANEL["y1"]), zs], 1))
    over_row1 = np.clip((WEB / 2 + 3.5 - np.abs(zs - ROWS_Z[1])) / 3.5, 0, 1) * (WEB_T + 0.3)
    off = 0.15 + over_row1
    P = list(P + N * off[:, None])
    top = P[-1].copy()
    for z in np.linspace(166, BACK_H + 4, 16 if high else 3)[1:]:
        P.append(np.array([sx, top[1] - (z - 166) * 0.08, z]))
    return np.array(P)


def poly_strap(j, high):
    P = strap_path(j, high)
    T, S, U = frames(P, (1, 0, 0))
    if U[3] @ np.array([0, 1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(25.0, POLY_T, 0.6 if high else 0, 2 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 10.0 + j)], 1)) if high else None
    return sweep(f"poly{j}", P, S, U, prof, hid, attr=attr, uv=not high)


def snap_tab(j, high):
    """the strap's lower end: a wider moulded tab carrying the press-stud"""
    sx = STRAP_X[j]
    b = panel_box()
    zs = np.linspace(42, 9, 40 if high else 4)
    P, N = b.surface(np.stack([np.full_like(zs, sx), np.full_like(zs, PANEL["y1"]), zs], 1))
    P = P + N * (0.15 + POLY_T - 0.2)
    T, S, U = frames(P, (1, 0, 0))
    if U[1] @ np.array([0, 1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(27.0, 2.2, 1.0 if high else 0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 16.0 + j)], 1)) if high else None
    return sweep(f"tab{j}", P, S, U, prof, hid, attr=attr, uv=not high)


def stud(j, high):
    from lab import geo
    seg = 48 if high else 12
    prof = [(0, 4.2), (2.5, 4.1), (5.0, 3.6), (6.8, 2.6), (7.6, 1.4), (8.0, 0.6), (7.9, -0.4), (0, -0.4)]
    if not high:
        prof = [(0, 4.2), (4.0, 3.8), (7.0, 2.4), (8.0, 0.6), (7.9, -0.4), (0, -0.4)]
    o = geo.lathe(f"stud{j}", [dict(pts=[(r * MM, z * MM) for r, z in prof], uv=(0, 0, 1, 1))], segs=seg)
    b = panel_box()
    P, N = b.surface(np.array([[STRAP_X[j], PANEL["y1"], STUD_Z]]))
    p = P[0] + N[0] * (0.15 + POLY_T - 0.2 + 2.2)
    o.data.transform(Matrix.Rotation(-math.pi / 2, 4, "X"))
    o.location = Vector(p) * MM
    return bake_xform(o)


def clip(j, high):
    """moulded clip / slide at the top of each strap (rectangular loop the webbing tab runs through)"""
    sx = STRAP_X[j]
    P = strap_path(j, high)
    y = P[-1][1] + POLY_T / 2
    o = PG.solid_box(f"clip{j}", 31, 5.0, 7.0, r=0, segs=1)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.bevel(bm, geom=list(bm.edges) if high else [e for e in bm.edges if abs(e.verts[0].co.y - e.verts[1].co.y) > 1e-7],
                    offset=(0.9 if high else 1.6) * MM, segments=3 if high else 1, affect="EDGES", profile=0.5)
    bm.to_mesh(o.data); bm.free()
    o.location = Vector((sx, y, BACK_H - 1.0)) * MM
    return bake_xform(o)


def top_loop(j, high):
    """folded webbing loop standing up above the clip"""
    sx = STRAP_X[j]
    y0 = strap_path(j, high)[-1][1] + POLY_T / 2
    n = 30 if high else 4
    P = [np.array([sx, y0 + 1.6, z]) for z in np.linspace(BACK_H - 4, BACK_H + 26, n)]
    rr = 1.3 + WEB_T / 2
    for a in np.linspace(0, math.pi, 12 if high else 4)[1:]:
        P.append(np.array([sx, y0 + 1.6 - rr + rr * math.cos(a), BACK_H + 26 + rr * math.sin(a)]))
    for z in np.linspace(BACK_H + 26, BACK_H - 4, n)[1:]:
        P.append(np.array([sx, y0 + 1.6 - 2 * rr, z]))
    P = np.array(P)
    T, S, U = frames(P, (1, 0, 0))
    if U[2] @ np.array([0, 1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(22.0, WEB_T, 0.5 if high else 0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 30.0 + j)], 1)) if high else None
    return sweep(f"tloop{j}", P, S, U, prof, hid, attr=attr, uv=not high)


def eyelet(i, high):
    """grommet on the front face of the tall back (seen with the flap open)"""
    from lab import geo
    prof = [(2.6, -0.6), (2.6, 0.9), (3.2, 1.4), (5.0, 1.3), (5.6, 0.6), (5.5, -0.6)]
    if not high:
        prof = [(2.6, -0.6), (2.6, 1.0), (5.0, 1.3), (5.6, -0.6)]
    prof = prof + [prof[0]]
    # closed ring: lathe a closed profile
    o = geo.lathe(f"eye{i}", [dict(pts=[(r * MM, z * MM) for r, z in prof], uv=(0, 0, 1, 1))], segs=40 if high else 10)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    b = back_box(i)
    P, N = b.surface(np.array([[XC[i], b.c[1] - b.h[1], 120.0]]))
    o.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))           # axis -> -Y
    o.location = Vector(P[0] + N[0] * 0.3) * MM
    return bake_xform(o)


def channel_tab(k, high):
    """short webbing bridge across the channel between sleeves at the front-wall top"""
    x = (XC[k] + XC[k + 1]) / 2
    xs = np.linspace(x - 9, x + 9, 20 if high else 3)
    P = np.stack([xs, np.full_like(xs, -2.0), np.full_like(xs, FRONT_H - 8)], 1)
    T, S, U = frames(P, (0, 0, 1))
    if U[1] @ np.array([0, -1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(12.0, WEB_T, 0.4 if high else 0, 2 if high else 0)
    s = xs - xs[0]
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 40.0 + k)], 1)) if high else None
    return sweep(f"ctab{k}", P, S, U, prof, hid, attr=attr, uv=not high)


# ------------------------------------------------------------------------------------------------ flap
HINGE_Y = 3.0


def flap_path(i, high):
    """centre line (bottom of the profile = inner face) of a closed flap: from the hinge on the tall back's top,
    forward over the mag, round the corner and down the front onto the loop panel"""
    yf = SY0 - SHY - 2.0 - 1.3 - 0.3          # front of the loop panel
    ztop = BACK_H + 1.5
    rc = 9.0
    P = []
    n1 = 20 if high else 4
    for y in np.linspace(HINGE_Y, yf + rc, n1):
        P.append((y, ztop))
    for a in np.linspace(0, math.pi / 2, 14 if high else 4)[1:]:
        P.append((yf + rc - rc * math.sin(a), ztop - rc + rc * math.cos(a)))
    for z in np.linspace(ztop - rc, HOOK_Z[0], 60 if high else 6)[1:]:
        # slight belly over the mag, then onto the loop panel
        t = (ztop - rc - z) / (ztop - rc - HOOK_Z[0])
        P.append((yf - 1.6 * math.sin(math.pi * min(1, t * 1.15)) ** 1.2 * (t < 0.87), z))
    return np.array([[XC[i], y, z] for y, z in P])


def flap(i, high):
    P = flap_path(i, high)
    T, S, U = frames(P, (1, 0, 0))
    # outward = up on the top run
    if U[1] @ np.array([0, 0, 1]) < 0:
        S, U = -S, -U
    prof, hid = rrect(FLAP_W, FLAP_T, 1.2 if high else 0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 50.0 + i)], 1)) if high else None
    o = sweep(f"flap{i}", P, S, U, prof, None, attr=attr, uv=not high)
    return o


def hook_strip(i, high):
    """hook strip on the flap's inner face at the free end (just proud of it, presses on the loop panel)"""
    yf = SY0 - SHY - 2.0 - 1.3 - 0.3
    zs = np.linspace(HOOK_Z[1], HOOK_Z[0] + 0.5, 20 if high else 3)
    P = np.stack([np.full_like(zs, XC[i]), np.full_like(zs, yf + 1.0), zs], 1)
    T, S, U = frames(P, (1, 0, 0))
    if U[0] @ np.array([0, -1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(FLAP_W - 6, 1.2, 0.3 if high else 0, 2 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 60.0 + i)], 1)) if high else None
    return sweep(f"hook{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def pull_tab(i, high):
    """small folded webbing pull tab sewn on the flap top, near the hinge"""
    y0, z0 = HINGE_Y - 6.0, BACK_H + 1.5 + FLAP_T
    n = 12 if high else 3
    P = [np.array([XC[i], y0 + 2.0, z]) for z in np.linspace(z0 - 1.0, z0 + 12, n)]
    rr = 1.0 + WEB_T / 2
    for a in np.linspace(0, math.pi, 10 if high else 4)[1:]:
        P.append(np.array([XC[i], y0 + 2.0 - rr + rr * math.cos(a), z0 + 12 + rr * math.sin(a)]))
    for z in np.linspace(z0 + 12, z0 - 1.0, n)[1:]:
        P.append(np.array([XC[i], y0 + 2.0 - 2 * rr, z]))
    P = np.array(P)
    T, S, U = frames(P, (1, 0, 0))
    if U[1] @ np.array([0, 1, 0]) < 0:
        S, U = -S, -U
    prof, hid = rrect(20.0, WEB_T, 0.5 if high else 0, 3 if high else 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    attr = (lambda st, a, b_: np.stack([s[st], a, np.full_like(a, 70.0 + i)], 1)) if high else None
    return sweep(f"ptab{i}", P, S, U, prof, hid, attr=attr, uv=not high)


def hinge(i):
    return (XC[i], HINGE_Y, BACK_H + 1.5)


# ------------------------------------------------------------------------------------------------ magazine proxy
def magazine(i, high):
    """STANAG 30-rd proxy (25 x 81 x 190): curved body, ribbed sides, feed lips + top round, flared baseplate"""
    from lab import geo
    w, d, h, z0 = MAG["w"], MAG["d"], MAG["h"], MAG["z0"]
    nz = 40 if high else 8
    zs = np.linspace(0, 1, nz)
    curve = lambda t: 4.0 * t ** 2
    rings = []
    bm = bmesh.new()
    prof = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    if high:
        prof = []
        for cx, cy, a0 in ((w / 2 - 2, -d / 2 + 2, -90), (w / 2 - 2, d / 2 - 2, 0), (-w / 2 + 2, d / 2 - 2, 90), (-w / 2 + 2, -d / 2 + 2, 180)):
            for k in range(4):
                a = math.radians(a0 + 90 * k / 3)
                prof.append((cx + 2 * math.cos(a), cy + 2 * math.sin(a)))
    for t in zs:
        z = z0 + 6 + t * (h - 16)
        dx = curve(1 - t)
        rib = 0.0
        ring = [bm.verts.new(((XC[i] + x + dx) * MM, (SY0 + y * (1 + rib)) * MM, z * MM)) for x, y in prof]
        rings.append(ring)
    # baseplate (flared) and top (feed lips)
    bot = [bm.verts.new(((XC[i] + x * 1.06 + curve(1)) * MM, (SY0 + y * 1.14) * MM, z0 * MM)) for x, y in prof]
    top = [bm.verts.new(((XC[i] + x * 0.92) * MM, (SY0 + y * 0.9) * MM, (z0 + h) * MM)) for x, y in prof]
    seq = [bot] + rings + [top]
    for a, b in zip(seq, seq[1:]):
        n = len(a)
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bm.faces.new(bot[::-1]); bm.faces.new(top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if not high:
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    me = bpy.data.meshes.new(f"mag{i}"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(f"mag{i}", me); bpy.context.scene.collection.objects.link(o)
    me.shade_smooth()
    return o
