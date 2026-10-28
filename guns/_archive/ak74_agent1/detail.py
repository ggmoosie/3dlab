"""detail.py - the AK-74M surface-detail catalogue and the SDF helpers that apply it. Every feature names where it was
read: R = right profile master px, L = left profile px (ak_ref.lp2b), close-ups by file. Domed rivets, pins and the
selector detents are bake-only shells (they feed the normal/AO bakes, not the game mesh); dimples, ribs, recesses, the
checkering panel, the side rail and the sling hardware are real geometry."""
import math
import numpy as np
import ak_ref as R
import ak_spec as P
from lab import sdf as S

bp, lp = R.bp, R.lp2b
MM = R.S_MM


def yb(x, row=None):
    return bp(x, 285.98 + R.BORE_SLOPE * (x - 1030.0) if row is None else row)[0]


def zb(row, x):
    return bp(x, row)[1]


def pt(side, x, row):
    return bp(x, row) if side == "R" else lp(x, row)


def stadium(Y, Z, a, b, r):
    return S.capsule2(Y, Z, a, b, r)


# ------------------------------------------------------------------ rivets / pins (bake-only heads)
# (part, side, x, row, head radius mm, kind)  kind: dome (domed rivet), pin (flush pin end, bright), ring (pin + washer)
RIVETS = [
    # right side (R photo, R_recA / R_recB 8x grids)
    ("receiver", "R", 468.0, 281.2, 1.4, "dome"),            # rear trunnion small rivets
    ("receiver", "R", 531.5, 302.5, 1.6, "dome"),            # selector detent rivet
    ("receiver", "R", 543.5, 311.5, 2.6, "pin"),             # trigger pin (bright)
    ("receiver", "R", 585.5, 304.5, 2.6, "pin"),             # hammer pin (bright)
    ("receiver", "R", 600.0, 307.0, 2.6, "pin"),             # trigger-guard front rivet (bright)
    ("receiver", "R", 665.0, 300.5, 2.4, "dome"),            # magwell rivet
    ("receiver", "R", 716.0, 283.0, 2.4, "dome"),            # front trunnion rivet
    ("receiver", "R", 698.0, 283.0, 2.4, "dome"),
    ("receiver", "R", 707.0, 298.0, 2.4, "dome"),
    # left side (L photo, L_c 4x grid; 163320 / 163334 close-ups)
    ("receiver", "L", 751.0, 301.0, 2.6, "pin"),
    ("receiver", "L", 766.0, 297.0, 2.6, "pin"),
    ("receiver", "L", 807.5, 304.0, 2.6, "pin"),
    ("receiver", "L", 685.0, 294.5, 2.3, "dome"),
    ("receiver", "L", 656.0, 293.0, 1.8, "dome"),
    ("receiver", "L", 610.0, 289.0, 2.6, "dome"),            # front trunnion rivets (163320: three domes)
    ("receiver", "L", 610.0, 274.0, 2.6, "dome"),
    ("receiver", "L", 628.0, 274.0, 2.6, "dome"),
    ("receiver", "L", 858.0, 275.0, 1.4, "dome"),            # rear trunnion small rivets
    ("receiver", "L", 864.0, 275.0, 1.4, "dome"),
    ("gasblock", "R", 990.0, 272.0, 1.5, "dome"),            # gas block pins (163337: two)
    ("gasblock", "R", 990.0, 283.0, 1.5, "dome"),
    ("fsb", "R", 1062.0, 284.0, 1.4, "pin"),                 # FSB taper pins
    ("fsb", "R", 1088.0, 284.0, 1.4, "pin"),
    ("fsb", "R", 1075.2, 272.4, 1.6, "pin"),                 # brake detent pin (bright dot)
]
# receiver dimples / embossings: (side, x0, r0, x1, r1, height mm)  + = raised (163320 shows a lit top rim)
DIMPLES = [("R", 627.5, 297.5, 655.0, 306.0, 0.7), ("L", 697.0, 291.0, 722.0, 299.0, 0.7)]
# selector stop dimples above the selector on the right (163300: three small ones) and the left-side 'stop' marks
SMALL_DIMPLES = [("R", 506.0, 274.0), ("R", 512.0, 276.0), ("L", 862.0, 285.0), ("L", 846.0, 286.0)]


def dome(X, Y, Z, c, r, hw, sign, h=0.9):
    """domed rivet head on the side wall |x| = hw (sign +1 left / -1 right)"""
    y, z = c
    rho = np.hypot(Y - y, Z - z)
    W = X * sign
    k = (r * r + h * h) / (2 * h)                                             # sphere through the rim and the apex
    return np.maximum(np.sqrt(rho * rho + (W - (hw + h - k)) ** 2) - k, hw - 0.3 - W)


def heads(X, Y, Z, part, hw):
    out = []
    for p, side, x, row, r, kind in RIVETS:
        if p != part:
            continue
        c = pt(side, x, row); sg = -1 if side == "R" else 1
        if kind == "dome":
            out.append(dome(X, Y, Z, c, r, hw, sg, 0.9))
        else:
            W = X * sg
            out.append(np.maximum(np.hypot(Y - c[0], Z - c[1]) - r, S.slab(W, hw - 0.6, hw + 0.25)))
    return S.union(*out) if out else np.full_like(X, 1e3)


# ------------------------------------------------------------------ per-part features
def apply(d, X, Y, Z, part, hw):
    ax = np.abs(X)
    if part == "receiver":
        for side, x0, r0, x1, r1, h in DIMPLES:
            (ya, za), (yb_, zb_) = pt(side, x0, r1), pt(side, x1, r0)
            zc = (za + zb_) / 2; rr = (zb_ - za) / 2
            d2 = S.capsule2(Y, Z, (ya + rr, zc), (yb_ - rr, zc), rr)
            sg = -1 if side == "R" else 1
            boss = S.extrude(d2 + 0.6, X * sg, hw - 0.5, hw + h, 0.5)
            d = S.runion(d, boss, 0.6)
        c = pt("L", 875.0, 289.0)
        btn = np.maximum(np.hypot(Y - c[0], Z - c[1]) - 3.6, S.slab(X, hw - 0.5, hw + 1.6))
        btn = S.rdiff(btn, np.maximum(np.hypot(Y - c[0], Z - c[1]) - 1.5, hw + 1.0 - X), 0.2)
        d = S.runion(d, btn, 0.4)
        for side, x, row in SMALL_DIMPLES:
            c = pt(side, x, row); sg = -1 if side == "R" else 1
            d = S.rdiff(d, np.maximum(np.hypot(Y - c[0], Z - c[1]) - 1.1, hw - 0.5 - X * sg), 0.3)
        # stamped serial on the left lip (generic digits; the lettering itself is in the print mask) - a shallow band
        # trunnion seam lines: the rear trunnion edge and the front trunnion edge are visible as fine grooves
        for side, x in (("R", 489.0), ("R", 690.0), ("L", 618.0), ("L", 852.0)):
            y = pt(side, x, 290.0)[0]; sg = -1 if side == "R" else 1
            g = np.maximum(np.abs(Y - y) - 0.25, np.maximum(hw - 0.35 - X * sg, S.slab(Z, zb(314.0, x), zb(270.0, x))))
            d = S.rdiff(d, g, 0.15)
        return d
    if part == "dustcover":
        return d
    if part == "rsb":
        # the big round pin end on both sides of the RSB (163320: lower rear) and the leaf-spring rivet
        for side, x, row in (("R", 711.0, 262.0), ("L", 600.0, 255.0)):
            c = pt(side, x, row); sg = -1 if side == "R" else 1
            ring = np.maximum(np.abs(np.hypot(Y - c[0], Z - c[1]) - 3.0) - 0.45, hw - 2.5 - X * sg)
            d = S.rdiff(d, np.maximum(ring, X * sg - (hw + 2)), 0.2)
        return d
    if part == "grip":
        # checkering panel: shallow rectangular recess on both sides following the grip's lean (163309 / 163334)
        a, b = bp(495.0, 330.0), bp(477.0, 405.0)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        u = (Y - a[0]) * math.cos(ang) + (Z - a[1]) * math.sin(ang)
        v = -(Y - a[0]) * math.sin(ang) + (Z - a[1]) * math.cos(ang)
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        pan = S.rect2(u, v, (0.0, -6.8), (L, 6.8), 1.2)
        d = S.rdiff(d, S.extrude(pan, ax, hw - 3.2, hw + 8, 0.4), 0.5)
        return d
    if part == "mag":
        return mag_detail(d, X, Y, Z, hw)
    if part == "stock":
        return stock_detail(d, X, Y, Z, hw)
    return d


def _arc_center(p0, p1, sag):
    """centre and radius (B mm) of the outline arc p0 -> p1 (px) with sagitta sag (px)"""
    a = np.array(bp(*p0)); b = np.array(bp(*p1))
    mid = (a + b) / 2; ch = b - a; L = np.linalg.norm(ch)
    # photo 'left normal' (x right, row down) maps to B with z flipped: rotate the chord the other way
    nrm = np.array([-ch[1], ch[0]]) / L
    s = sag * MM
    Rr = (s * s + (L / 2) ** 2) / (2 * s)
    return mid + nrm * (s - Rr), abs(Rr)


def _circ3(a, b, c):
    """circle through three px points -> (centre B mm, radius mm)"""
    A, B_, C = (np.array(bp(*p)) for p in (a, b, c))
    M = np.array([[2 * (B_ - A)[0], 2 * (B_ - A)[1]], [2 * (C - A)[0], 2 * (C - A)[1]]])
    rhs = np.array([B_ @ B_ - A @ A, C @ C - A @ A])
    cc = np.linalg.solve(M, rhs)
    return cc, float(np.linalg.norm(cc - A))


def mag_detail(d, X, Y, Z, hw):
    """30-rd polymer magazine (163300 + profiles): the side panel is a raised field 0.7 mm proud, bounded by a front band
    (6 mm) and the rear spine (7 mm); short transverse tick ribs along both bands (11 mm pitch); floorplate lip"""
    global _MAGARC
    if "_MAGARC" not in globals():
        from outlines import fitted_table
        o = fitted_table()["mag"]
        _MAGARC = (_circ3(o.p[2], o.p[3], o.p[4]), _circ3(o.p[11], o.p[9], o.p[7]))
    (c, Rf), (cr, Rr) = _MAGARC
    rho = np.hypot(Y - c[0], Z - c[1]); phi = np.arctan2(Z - c[1], Y - c[0])
    rho_r = np.hypot(Y - cr[0], Z - cr[1])
    ax = np.abs(X)
    zfl = bp(700.0, 482.0)[1]                                         # above the floorplate lip
    ztop = bp(650.0, 318.0)[1]
    inband = (Z > zfl + 2.0) & (Z < ztop)
    # panel field: inside the front band / spine -> raised
    fld = np.maximum(Rf + 6.0 - rho, np.maximum((rho_r - (Rr - 7.0)) * np.sign(Rr), np.maximum(zfl + 4.0 - Z, Z - (ztop - 2.0))))
    panel = S.extrude(fld + 0.8, ax, hw - 0.5, hw + 0.7, 0.4)
    d = S.runion(d, panel, 0.5)
    # tick ribs along the front band (radial short ribs every 11 mm of arc)
    pitch = 11.0 / max(Rf, 1.0)
    t = np.abs(((phi / pitch) % 1.0) - 0.5) * pitch * rho
    tick = np.maximum.reduce([t - 0.45, np.abs(rho - (Rf + 3.0)) - 1.9, -(Z - (zfl + 6.0)), Z - (ztop - 4.0)])
    d = S.runion(d, S.extrude(tick, ax, hw - 0.5, hw + 0.45, 0.2), 0.2)
    return d


def stock_detail(d, X, Y, Z, hw):
    ax = np.abs(X)
    # right side: long recessed groove (R_a: from (250, 330) to (345, 313), 10 px wide, rounded ends)
    a, b = bp(252.0, 329.5), bp(343.0, 313.5)
    g = S.capsule2(Y, Z, a, b, 4.6 * MM)
    d = S.rdiff(d, S.extrude(g, -X, hw - 2.8, hw + 10, 0.0), 1.6)
    # left side: framed panel (L_d): raised frame groove around a recessed trapezoid panel
    q = [lp(933.0, 283.5), lp(1071.0, 300.5), lp(1069.0, 332.0), lp(936.0, 306.5)]
    pan = _quad(Y, Z, q)
    d = S.rdiff(d, S.extrude(pan + 1.6, X, hw - 1.3, hw + 10, 0.0), 0.6)                   # recessed panel
    d = S.rdiff(d, S.extrude(np.abs(pan) - 0.45, X, hw - 0.6, hw + 10, 0.0), 0.25)        # frame groove
    return d


def _quad(Y, Z, q):
    area = sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))
    sg = 1.0 if area > 0 else -1.0
    dd = np.full_like(Y, -1e3)
    for (y0, z0), (y1, z1) in zip(q, q[1:] + q[:1]):
        ny, nz = (z1 - z0) * sg, -(y1 - y0) * sg; n = math.hypot(ny, nz)
        dd = np.maximum(dd, ((Y - y0) * ny + (Z - z0) * nz) / n)
    return dd


# ------------------------------------------------------------------ side rail + sling hardware
def side_rail(X, Y, Z):
    """left-side optics rail (163334 close-up + L_c grid: x 732..852, rows 282..297): a stepped bar, front and rear
    dovetail lugs, rivets through it"""
    hw = P.HW_REC
    q = [lp(733.0, 282.5), lp(852.0, 282.5), lp(852.0, 297.0), lp(733.0, 297.0)]
    y0, y1 = sorted([q[0][0], q[1][0]]); z0, z1 = q[2][1], q[0][1]
    bar = S.box(X, Y, Z, (hw - 0.4, y0, z0), (hw + 3.2, y1, z1), 0.7)
    # stepped top (163334: a raised dovetail strip along the top, with the notch at the middle step)
    strip = S.box(X, Y, Z, (hw + 2.6, y0 + 2.0, z1 - 4.0), (hw + 5.6, y1 - 2.0, z1 + 0.2), 0.6)
    ym = lp(790.0, 0)[1] if False else lp(790.0, 0)[0]
    strip = S.rdiff(strip, S.box(X, Y, Z, (hw + 2, ym - 4.0, z1 - 5), (hw + 7, ym + 4.0, z1 + 1), 0.6), 0.4)
    d = S.runion(bar, strip, 0.5)
    # lightening slot (lower middle)
    d = S.rdiff(d, S.box(X, Y, Z, (hw + 1.8, ym - 9.0, z0 + 2.5), (hw + 6, ym + 9.0, z0 + 6.0), 1.2), 0.4)
    return d


def slings(X, Y, Z):
    """sling hardware: right stock-front swivel (R_a: loop x 425..450 rows 298..322 on a boss) + left stock-front swivel
    boss with its ring (L_d (918, 290))"""
    out = []
    hw = P.HW_STOCK_WRIST
    # right: boss at (437, 304) + D-loop hanging forward-down in the side plane
    yb_, zb_ = bp(437.0, 303.0)
    boss = S.cyl(X, Y, Z, "x", (yb_, zb_), 3.0, -hw - 2.6, -hw + 1.0, 0.6)
    loop = _dloop(Y, Z, bp(427.0, 307.0), bp(449.0, 321.5), 1.0)
    out.append(boss); out.append(np.maximum(loop, S.slab(X, -hw - 2.6, -hw - 0.6)))
    # left: round swivel boss + ring standing out
    yl, zl = lp(918.0, 290.0)
    out.append(S.cyl(X, Y, Z, "x", (yl, zl), 4.8, hw - 1.0, hw + 1.6, 0.6))
    th = np.arctan2(Z - zl, Y - yl)
    tor = np.hypot(np.hypot(Y - yl, Z - zl) - 3.4, X - (hw + 2.4)) - 0.9
    out.append(tor)
    return S.union(*out)


def _dloop(Y, Z, a, b, r):
    """stadium-shaped wire loop in the (y, z) plane between corners a (top-left) and b (bottom-right), wire radius r"""
    cy, cz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    hy, hz = abs(b[0] - a[0]) / 2, abs(b[1] - a[1]) / 2
    rr = min(hy, hz)
    d2 = S.rect2(Y, Z, (cy - hy, cz - hz), (cy + hy, cz + hz), rr * 0.9)
    return np.abs(d2) - r


def bake_shells():
    """bake-only shells: rivet / pin heads (one per assembly so moving parts keep their own)"""
    T = {}
    def rec(X, Y, Z): return heads(X, Y, Z, "receiver", P.HW_REC)
    def gb(X, Y, Z): return S.union(heads(X, Y, Z, "gasblock", 9.2), heads(X, Y, Z, "fsb", 9.6))
    T["rivets_rec"] = (rec, (-P.HW_REC - 2, yb(455.0), zb(320.0, 600) - 3), (P.HW_REC + 2, yb(730.0), zb(255.0, 600) + 3), 0.1, "park")
    T["rivets_front"] = (gb, (-11.0, yb(980.0), zb(300.0, 1040) - 3), (11.0, yb(1095.0), zb(265.0, 1040) + 3), 0.1, "park")
    return T
