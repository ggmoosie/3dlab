"""parts.py - AK-74M high-poly: one SDF per part (lab/sdf.py narrow-band marching cubes) -> work/high.npz (B frame mm).
Side outlines are the fitted parametric outlines (outlines.py); cross-sections, widths and every 3-D feature are clean
parametric shapes measured off the profile master, the 3/4 shots and the close-ups (never traced pixel blobs).
usage: .venv/bin/python guns/ak74/parts.py [h_scale=1.0] [parts=a,b] [out=path]"""
import sys, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import ak_ref as R
import ak_spec as P
import outlines as OL
import detail as DT

MAT = dict(park=0, poly=1, steel=2, bright=3, rubber=4, brass=5, magpoly=6, paint=7)
HS = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("h_scale=")), 1.0))
ONLY = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("parts=")), None)
bp = R.bp
lerp = S.lerp
_F = {}


def F(name, soften=0.0):
    """side-profile signed distance (y, z) of a fitted outline"""
    if name not in _F:
        T = OL.fitted_table()
        _F[name] = T[name].field(0.08, soften=soften)
    return _F[name]


def bore_row(x):
    return 285.98 + R.BORE_SLOPE * (x - 1030.0)


def yb(x, row=None):
    """y of photo column x (on the bore line unless a row is given)"""
    return bp(x, bore_row(x) if row is None else row)[0]


def zb(row, x):
    """z of photo row at column x (the master is rotated 0.55 deg: z depends on the column)"""
    return bp(x, row)[1]


def rect_px(Y, Z, x0, r0, x1, r1, r=0.0):
    """axis-aligned (in B) rectangle through the px box corners (centre column for the rows, centre row for the columns)"""
    xc, rc = (x0 + x1) / 2, (r0 + r1) / 2
    return S.rect2(Y, Z, (yb(x0, rc), zb(r1, xc)), (yb(x1, rc), zb(r0, xc)), r)


def box_px(X, Y, Z, x0, r0, x1, r1, xlo, xhi, r=0.0):
    xc, rc = (x0 + x1) / 2, (r0 + r1) / 2
    return S.box(X, Y, Z, (xlo, yb(x0, rc), zb(r1, xc)), (xhi, yb(x1, rc), zb(r0, xc)), r)


def cap_px(Y, Z, p0, p1, rpx):
    return S.capsule2(Y, Z, bp(*p0), bp(*p1), rpx * R.S_MM)


def circ_px(Y, Z, x, row, rpx):
    y, z = bp(x, row)
    return np.hypot(Y - y, Z - z) - rpx * R.S_MM


def poly2(Y, Z, pts_px):
    """2-D SDF of a convex px polygon (any winding) in (Y, Z)"""
    Pm = [bp(*q) for q in pts_px]
    ys = [p[0] for p in Pm]; zs = [p[1] for p in Pm]
    area = sum(ys[i] * zs[(i + 1) % len(Pm)] - ys[(i + 1) % len(Pm)] * zs[i] for i in range(len(Pm)))
    sg = 1.0 if area > 0 else -1.0
    d = np.full_like(Y, -1e3)
    for (y0, z0), (y1, z1) in zip(Pm, Pm[1:] + Pm[:1]):
        ny, nz = (z1 - z0) * sg, -(y1 - y0) * sg; n = math.hypot(ny, nz)
        d = np.maximum(d, ((Y - y0) * ny + (Z - z0) * nz) / n)
    return d


def lathe(Y, Rr, prof):
    """radius profile [(y, r)] about the axis (exact enough for gentle tapers; steps get their own rounding)"""
    ys, rs = zip(*prof)
    return np.maximum(Rr - np.interp(Y, ys, rs), S.slab(Y, ys[0], ys[-1]))


def side_feature(X, d2, x_in, x_out, r=0.4, sign=0):
    """extrude a 2-D (y, z) feature through |x| (sign 0), the left side (+1) or the right side (-1)"""
    W = np.abs(X) if sign == 0 else X * sign
    return S.extrude(d2, W, x_in, x_out, r)


# ================================================================== receiver group
REC_SLOT = (612.0, 689.0, 263.6, 271.5)              # ejection port px (x0, x1, row top, row bottom)


def receiver(X, Y, Z):
    hw = P.HW_REC
    d = S.rinter(F("receiver")(Y, Z), np.abs(X) - hw, 1.5)
    # magwell (the magazine's top seats in it), trigger slot
    d = S.diff(d, box_px(X, Y, Z, 600.5, 286.0, 683.0, 318.0, -13.9, 13.9, 1.0))
    d = S.diff(d, box_px(X, Y, Z, 539.5, 300.0, 557.0, 321.0, -3.6, 3.6, 0.8))
    # ejection port (right side, below the dust cover skirt): open window into the receiver; the carrier shows through
    x0, x1, r0, r1 = REC_SLOT
    port = box_px(X, Y, Z, x0, r0, x1, r1, -hw - 5, -hw + 4.5, 0.8)
    d = S.rdiff(d, port, 0.5)
    # the receiver's top-edge bead (stamped rail lip) just below the dust-cover skirt, both sides (163320)
    lip = S.extrude(S.slab(Z, zb(266.6, 580), zb(264.4, 580)), np.abs(X), hw - 0.5, hw + 0.45, 0.4)
    lip = np.maximum(lip, S.slab(Y, yb(470.0), yb(702.0)))
    lip = np.maximum(lip, -port)
    d = S.runion(d, lip, 0.3)
    return DT.apply(d, X, Y, Z, "receiver", hw)


def dustcover(X, Y, Z):
    """stamped cover: flat top, big shoulder roll (r 7), vertical skirts outside the receiver, a stamped step along the
    side (163320: the lower skirt sits 0.6 inboard below the step line) and the deeper front skirt (last 40 %)"""
    hw = P.HW_DC
    prof = F("dustcover")(Y, Z)
    d = S.rinter(prof, np.abs(X) - hw, 7.0)
    zstep = zb(256.6, 600)
    d = np.maximum(d, np.where(Z < zstep, np.abs(X) - (hw - 0.6), -1e3))
    ext = box_px(X, Y, Z, 611.0, 262.0, 703.0, 267.0, -hw + 0.6, hw - 0.6, 0.6)
    ext = np.maximum(ext, -(np.abs(X) - (hw - 1.6)))
    d = S.union(d, ext)
    inner = S.rinter(prof + 1.0, np.abs(X) - (hw - 1.0), 6.0)
    d = S.diff(d, np.maximum(inner, zb(262.0, 600) - Z))
    yc, zc = bp(461.0, 265.0)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, zc), 4.6, yb(452.0), yb(472.0)))
    return d


def guidebtn(X, Y, Z):
    """recoil-spring guide rod rear button through the dust-cover rear"""
    return S.rinter(F("guidebtn")(Y, Z), np.abs(X) - 5.2, 1.2)


def rsb(X, Y, Z):
    """rear sight block: trapezoid section (wide at the trunnion, narrower at the top), the leaf channel, the gas-tube
    lock lever on the right"""
    zt, zbm = zb(236.0, 740), zb(270.0, 740)
    t = np.clip((Z - zbm) / (zt - zbm), 0, 1)
    hw = lerp(P.HW_RSB, 10.8, t)
    d = S.rinter(F("rsb")(Y, Z), np.abs(X) - hw, 1.0)
    d = S.rdiff(d, box_px(X, Y, Z, 724.0, 230.0, 757.5, 244.6, -7.6, 7.6, 0.6), 0.5)
    lev = S.extrude(np.minimum(cap_px(Y, Z, (764.0, 247.5), (772.5, 264.5), 3.0), circ_px(Y, Z, 764.0, 247.5, 4.2)),
                    -X, P.HW_RSB - 3.0, P.HW_RSB + 1.4, 0.6)
    d = S.runion(d, lev, 0.4)
    return DT.apply(d, X, Y, Z, "rsb", P.HW_RSB)


def leaf(X, Y, Z):
    d = S.rinter(F("leaf")(Y, Z), np.abs(X) - 7.0, 0.5)
    y, z = bp(698.8, 231.7)
    notch = np.maximum(np.abs(X) - 1.0 - 0.5 * np.maximum(Z - (z - 2.4), 0), z - 2.4 - Z)
    return S.rdiff(d, np.maximum(notch, S.slab(Y, y - 6, y + 6)), 0.15)


def slider(X, Y, Z):
    d = S.rinter(F("slider")(Y, Z), np.abs(X) - 9.0, 0.8)
    y, z = bp(716.0, 238.5)
    knobs = S.cyl(X, Y, Z, "x", (y, z), 3.1, -13.6, 13.6, 0.8)
    th = np.arctan2(Z - z, Y - y)
    kn = np.maximum(np.abs(((th * 24 / (2 * math.pi)) % 1.0) - 0.5) * 2 * math.pi * 3.1 / 24 - 0.22,
                    np.maximum(2.6 - np.hypot(Y - y, Z - z), 9.5 - np.abs(X)))
    return S.union(d, S.diff(knobs, kn))


# ================================================================== handguards, gas system
GT_Z, GT_R = 24.3, 8.35                       # gas tube axis / radius (rotated-frame crossings x 918..936)


def upperhg(X, Y, Z):
    """gas-tube cover: rounded-square polymer body (r 8 top corners), tapering top"""
    d = S.rinter(F("upperhg")(Y, Z), np.abs(X) - P.HW_UHG, 8.0)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, GT_Z), GT_R + 0.3, yb(770.0), yb(900.0)))
    return d


def _sec_u(X, Z, grow, x):
    return S.rinter(S.slab(Z, zb(270.5, x), zb(240.4, x) + grow), np.abs(X) - (P.HW_UHG + grow), 8.0 + grow)


def hg_rings_u(X, Y, Z):
    """steel ferrules of the upper handguard (gas tube assembly): rear (x 775.6..785) and front (880.5..893.5)"""
    rr = np.maximum(_sec_u(X, Z, 0.5, 780), S.slab(Y, yb(775.6), yb(785.0)))
    rf = S.rinter(S.slab(Z, zb(270.5, 887), zb(243.6, 887)), np.abs(X) - (P.HW_UHG + 0.4), 7.6)
    rf = np.maximum(rf, S.slab(Y, yb(880.5), yb(893.5)))
    d = np.minimum(rr, rf)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, GT_Z), GT_R + 0.2, yb(770.0), yb(900.0)))


def hg_ring_l(X, Y, Z):
    """lower handguard front retainer (steel, x 884..897)"""
    lw = np.maximum(S.rinter(S.slab(Z, zb(305.6, 890), zb(268.5, 890)), np.abs(X) - (P.HW_LHG - 1.5), 8.0),
                    S.slab(Y, yb(884.0), yb(897.0)))
    return S.diff(lw, S.cyl(X, Y, Z, "y", (0, 0), 7.0, yb(870.0), yb(905.0)))


def lowerhg(X, Y, Z):
    """lower handguard: rounded-rectangle section, sides taper in at the top to meet the upper; 6 longitudinal finger
    ribs at the rear with staggered fronts; 4 vertical slots at the front; thumb bulge at the rear bottom"""
    zt = zb(270.5, 800)
    hw = np.where(Z > zt - 6.0, P.HW_LHG - 2.5 * S.sstep(zt - 6.0, zt, Z), P.HW_LHG)
    d = S.rinter(F("lowerhg")(Y, Z), np.abs(X) - hw, 7.5)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.4, yb(700.0), yb(900.0)))
    ax = np.abs(X)
    fr = [802.0, 806.0, 806.0, 802.0, 796.0, 788.0]
    for k, xf in enumerate(fr):
        row = 273.8 + 4.75 * k
        g = S.extrude(np.maximum(S.slab(Z, zb(row + 1.25, 760), zb(row - 1.25, 760)), S.slab(Y, yb(731.0), yb(xf))), ax,
                      P.HW_LHG - 1.6, P.HW_LHG + 5, 0.35)
        d = S.rdiff(d, g, 0.3)
    for xs in (854.5, 861.0, 867.5, 874.0):
        g = S.extrude(np.maximum(S.slab(Y, yb(xs - 1.4), yb(xs + 1.4)), S.slab(Z, zb(292.5, xs), zb(276.0, xs))), ax,
                      P.HW_LHG - 1.4, P.HW_LHG + 5, 0.3)
        d = S.rdiff(d, g, 0.3)
    return d


def gastube(X, Y, Z):
    """gas tube from the RSB to the gas-block cup: plain tube r 8.35, rear collar by the handguard cap (r 10.6), two vent
    holes per side and shallow longitudinal flutes on the exposed part (163337)"""
    Rr = np.hypot(X, Z - GT_Z)
    prof = [(yb(770.0), GT_R), (yb(893.0), GT_R), (yb(893.2), 10.5), (yb(904.0), 10.5), (yb(905.6), GT_R), (yb(943.0), GT_R)]
    d = lathe(Y, Rr, prof)
    for xh in (912.0, 920.0):
        d = S.diff(d, S.cyl(X, Y, Z, "x", (yb(xh), GT_Z + 1.0), 1.4, -12, 12))
    th = np.arctan2(Z - GT_Z, X)
    fl = np.maximum(np.abs(((th * 10 / (2 * math.pi)) % 1.0) - 0.5) * 2 * math.pi * GT_R / 10 - 0.5,
                    np.maximum(GT_R - 0.35 - Rr, S.slab(Y, yb(907.5), yb(940.0))))
    return S.diff(d, fl)


def gasblock(X, Y, Z):
    """gas block: cup (axis 25.4, r 9.0) taking the tube, web down to the barrel ring (r 9.4), bayonet lug below with the
    rod notch, the arch over the barrel; outline = the photo's (fitted)"""
    side = F("gasblock")(Y, Z)
    cup = S.rinter(side, np.hypot(X, Z - 25.4) - 9.0, 0.6)
    web = S.rinter(side, np.maximum(np.abs(X) - 8.8, Z - 25.4), 1.2)
    ring = S.rinter(side, np.hypot(X, Z) - 9.4, 1.0)
    lug = S.rinter(side, np.maximum(np.abs(X) - 5.8, Z + 8.0), 0.8)
    d = S.runion(S.runion(cup, web, 1.5), ring, 1.5)
    d = S.union(d, lug)
    zr = zb(299.0, 999)
    d = S.rdiff(d, np.maximum(np.hypot(X, Z - zr) - 2.0, S.slab(Y, yb(990.0), yb(1010.0))), 0.3)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 6.7, yb(950.0), yb(1010.0)))
    d = S.diff(d, np.maximum(np.hypot(X, Z - 25.4) - GT_R - 0.05, S.slab(Y, yb(930.0), yb(983.0))))
    return d


def barrel(X, Y, Z):
    Rr = np.hypot(X, Z)
    # front handguard retainer collar on the barrel (x 896..905.5, bottom row 294.5 = r 9.4)
    collar = np.maximum(np.minimum(Rr - 9.4, S.box(X, Y, Z, (-7.5, yb(896.0), 0.0), (7.5, yb(905.5), zb(271.4, 901)), 2.0)),
                        S.slab(Y, yb(896.0), yb(905.5)))
    prof = [(yb(698.0), 7.6), (yb(890.0), 7.0), (yb(905.0), 6.7), (yb(966.0), 6.7), (yb(1004.0), 6.6), (yb(1015.0), 6.48),
            (yb(1058.0), 6.4), (yb(1094.0), 6.2), (yb(1100.0), 6.2)]
    d = S.runion(lathe(Y, Rr, prof), S.rinter(collar, -1e3 + 0 * Rr, 0.0), 0.6)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 2.8, yb(690.0), yb(1110.0)))


def rod(X, Y, Z):
    """cleaning rod under the barrel, slightly tilted as photographed (z -13.8 at x 905 -> -12.75 at x 1045)"""
    y0, y1 = yb(905.0), yb(1045.0)
    zr = -13.8 + (Y - y0) * (1.05 / (y1 - y0))
    d = np.maximum(np.hypot(X, Z - zr) - 1.72, S.slab(Y, yb(760.0), yb(1092.0)))
    head = np.maximum(np.hypot(X, Z - zr) - 2.3, S.slab(Y, yb(1086.0), yb(1092.0)))
    return S.union(d, head)


def fsb(X, Y, Z):
    """front sight block (outline fitted): barrel ring, tower leaning back with the ears (U seen from the front, tips
    narrowing), the post, the drift window, rod lugs below"""
    side = F("fsb")(Y, Z)
    ztop = zb(232.2, 1085)
    ring = S.rinter(side, np.maximum(np.hypot(X, Z) - 8.3, Z - 8.0), 0.8)
    tw = S.rinter(side, np.maximum(np.abs(X) - (9.4 - 2.4 * S.sstep(ztop - 14.0, ztop, Z)), 2.0 - Z), 0.9)
    lug = S.rinter(side, np.maximum(np.abs(X) - 5.2, Z + 6.0), 0.8)
    d = S.union(S.runion(ring, tw, 1.2), lug)
    slot = np.maximum(np.abs(X) - 5.0, ztop - 12.5 - Z)
    d = S.rdiff(d, np.maximum(slot, S.slab(Y, yb(1060.0), yb(1100.0))), 1.4)
    post = S.cyl(X, Y, Z, "z", (0, yb(1084.5)), 1.05, ztop - 13.5, ztop - 1.5, 0.45)
    base = S.box(X, Y, Z, (-2.4, yb(1081.5), ztop - 16.0), (2.4, yb(1087.5), ztop - 10.5), 0.6)
    d = S.union(d, post, base)
    yw, zw = bp(1081.2, 264.7)
    d = S.rdiff(d, S.cyl(X, Y, Z, "x", (yw, zw), 6.4 * R.S_MM, -20, 20), 0.3)
    zr = zb(299.6, 1075)
    d = S.rdiff(d, np.maximum(np.hypot(X, Z - zr) - 2.0, S.slab(Y, yb(1050.0), yb(1100.0))), 0.3)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 6.25, yb(1050.0), yb(1100.0)))
    return d


def brake(X, Y, Z):
    """AK-74 brake (rotated-frame crossings): threaded collar r 12.75 (x 1094..1116), chamber r 11.8 (1118..1168) with the
    two big side windows (x 1152..1163.2), a blind side vent each side at 1117, front lip r 8.0 with a chamfered face and
    the bottom slot, wide chamber + bore (163339)"""
    Rr = np.hypot(X, Z - 0.1)
    prof = [(yb(1093.0), 12.2), (yb(1094.0), 12.75), (yb(1116.0), 12.75), (yb(1118.5), 11.85), (yb(1150.0), 11.75),
            (yb(1168.0), 11.8), (yb(1170.8), 8.2), (yb(1178.6), 7.9), (yb(1179.6), 7.0)]
    d = lathe(Y, Rr, prof)
    d = S.rinter(d, np.maximum(Rr - 11.8, Y - yb(1170.8)), 0.0) if False else d
    d = S.diff(d, np.maximum(Rr - 6.3, S.slab(Y, yb(1090.0), yb(1112.0))))
    d = S.diff(d, np.maximum(Rr - 2.9, S.slab(Y, yb(1100.0), yb(1190.0))))
    d = S.diff(d, np.maximum(Rr - 8.6, S.slab(Y, yb(1121.0), yb(1168.5))))
    d = S.diff(d, np.maximum(Rr - 5.4, S.slab(Y, yb(1168.0), yb(1190.0))))
    yw0, yw1 = yb(1152.2, 286), yb(1163.2, 286)
    zw0, zw1 = zb(294.0, 1157), zb(279.4, 1157)
    d = S.rdiff(d, S.box(X, Y, Z, (-30, yw0, zw0), (30, yw1, zw1), 1.2), 0.4)
    yv, zv = bp(1117.5, 286.6)
    d = S.rdiff(d, np.maximum(np.hypot(Y - yv, Z - zv) - 1.45, 9.6 - np.abs(X)), 0.2)
    d = S.rdiff(d, S.box(X, Y, Z, (-2.6, yb(1170.0), -20), (2.6, yb(1182.0), -5.6), 0.4), 0.3)
    # inner baffle ring seen through the front opening (163339): a thin conical ring inside the chamber
    baffle = np.maximum(np.maximum(Rr - 8.7, 4.2 - Rr), S.slab(Y - (Rr - 4.2) * 0.35, yb(1163.5), yb(1165.0)))
    return S.union(d, baffle)


# ================================================================== controls, trigger group
def carrier(X, Y, Z):
    """bolt carrier with the long gas piston and the charging handle (right side). Inside the receiver: body x +-10.8,
    rounded top under the dust cover, its side seen through the ejection port; the piston (r 5.0, head r 8.0) runs forward
    through the RSB in the gas tube to the gas block; the handle sticks out of the port front (R_rs knob x 683..705,
    163300: a flat paddle with a rounded end)"""
    hw = P.HW_REC
    y0, y1 = yb(505.0), yb(689.5)
    body = S.rinter(S.slab(Z, zb(276.0, 600), zb(246.5, 600)), np.abs(X) - 10.8, 6.0)
    body = np.maximum(body, S.slab(Y, y0, y1))
    body = S.rdiff(body, S.box(X, Y, Z, (-6.2, y0 - 1, -20), (6.2, yb(650.0) + 2, zb(271.0, 600)), 1.0), 0.8)   # bolt raceway
    # side faces visible in the port: a guide rail along the lower edge
    rail = S.box(X, Y, Z, (-11.6, y0 + 4, zb(273.5, 600)), (11.6, y1 - 2, zb(271.0, 600)), 0.6)
    body = S.runion(body, rail, 0.4)
    # piston: carrier front collar -> rod -> head (the gas cylinder axis)
    zp = 24.6
    collar = S.cyl(X, Y, Z, "y", (0, zp), 7.2, y1 - 1, yb(706.0), 1.0)
    rodp = S.cyl(X, Y, Z, "y", (0, zp), 4.9, yb(704.0), yb(966.0), 0.6)
    head = S.cyl(X, Y, Z, "y", (0, zp), 7.9, yb(958.0), yb(968.0), 0.8)
    for xg in (960.5, 963.5):                                                         # gas-ring grooves on the head
        head = S.diff(head, np.maximum(np.abs(Y - yb(xg)) - 0.35, 7.4 - np.hypot(X, Z - zp)))
    web = S.box(X, Y, Z, (-5.0, y1 - 8, zp - 6), (5.0, y1 + 2, zp + 2), 1.0)
    d = S.union(S.runion(S.runion(body, web, 1.5), collar, 1.5), rodp, head)
    # charging handle: stem out of the port front + flat paddle with a rounded end
    yc, zc = bp(694.0, 265.0)
    stem = S.cyl(X, Y, Z, "x", (yc, zc), 3.2, -hw - 6.0, -6.0, 0.6)
    pad = S.rinter(S.rect2(Y, Z, (yc - 5.6, zc - 6.0), (yc + 5.6, zc + 6.0), 5.0), S.slab(X, -hw - 21.0, -hw - 5.0), 1.6)
    d = S.union(d, S.runion(stem, pad, 1.2))
    return d


def bolt(X, Y, Z):
    """rotating bolt in the carrier raceway: body r 6.0, head with two locking lugs, cam pin up into the carrier,
    extractor claw, firing-pin hole"""
    y0, y1 = yb(642.0), yb(706.0)
    body = S.cyl(X, Y, Z, "y", (0, 0), 6.0, y0, y1, 0.6)
    lugs = np.maximum(S.box(X, Y, Z, (-3.4, y1 - 9.0, -9.2), (3.4, y1, 9.2), 0.8), 5.0 - np.abs(Z))
    d = S.union(body, lugs)
    cam = S.cyl(X, Y, Z, "z", (0, yb(668.0)), 2.4, 0.0, 12.5, 0.6)
    ext = S.box(X, Y, Z, (-7.4, y1 - 22, -1.6), (-5.4, y1 + 0.6, 1.6), 0.5)
    d = S.union(d, cam, ext)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 1.2, y1 - 4, y1 + 2))


def recoil(X, Y, Z):
    """recoil-spring assembly: the guide button (rear, through the dust cover), guide rod and the coil spring (helix SDF)
    running forward into the carrier"""
    btn = S.rinter(F("guidebtn")(Y, Z), np.abs(X) - 5.2, 1.2)
    yc, zc = bp(461.0, 265.0)
    rodr = S.cyl(X, Y, Z, "y", (0, zc), 1.6, yc, yb(640.0), 0.5)
    ya, yb_ = yb(468.0), yb(612.0)
    rho = np.hypot(X, Z - zc); th = np.arctan2(Z - zc, X)
    pitch = 3.4
    ph = ((Y - pitch * th / (2 * math.pi)) / pitch) % 1.0
    ax = (np.minimum(ph, 1 - ph)) * pitch
    coil = np.maximum(np.hypot(rho - 4.2, ax) - 0.7, S.slab(Y, ya, yb_))
    return S.union(btn, rodr, coil)


def selector(X, Y, Z):
    """safety/selector lever on the right: hub with the axis nut, long tapered stamped lever up-forward to the finger tab
    that covers the charging-handle slot (R_b / 163300)"""
    hw = P.HW_REC
    hub = circ_px(Y, Z, 517.0, 288.0, 8.6)
    arm = np.minimum(cap_px(Y, Z, (517.0, 288.0), (600.0, 268.5), 5.2), cap_px(Y, Z, (560.0, 278.0), (624.0, 264.5), 3.6))
    tab = poly2(Y, Z, [(598.0, 262.0), (628.0, 261.5), (628.0, 271.0), (598.0, 274.5)])
    lev = S.extrude(np.minimum.reduce([hub, arm, tab]), -X, hw - 0.2, hw + 1.6, 0.5)
    lip = S.extrude(rect_px(Y, Z, 620.0, 261.0, 628.0, 271.0, 1.0), -X, hw + 1.0, hw + 4.2, 0.6)
    yh, zh = bp(517.0, 288.0)
    nut = S.cyl(X, Y, Z, "x", (yh, zh), 3.4, -hw - 3.4, -hw, 0.6)
    return S.union(S.runion(lev, lip, 0.8), nut)


def trigger(X, Y, Z):
    return S.rinter(F("trigger")(Y, Z), np.abs(X) - 3.2, 0.9)


def guard(X, Y, Z):
    """stamped trigger guard: 1.8 mm strip, 10 mm wide (row crossings: rear leg x 529.8, bottom 356.5 at x 540..548, front
    leg x 584.2 into the catch block)"""
    tang = S.extrude(rect_px(Y, Z, 529.0, 315.4, 537.4, 320.2, 0.6), np.abs(X), 0, 5.0, 0.6)
    path = [(529.8, 316.0), (529.8, 346.5), (532.6, 352.5), (538.0, 355.6), (548.0, 355.6), (570.0, 354.0), (578.0, 352.0),
            (583.4, 348.0), (585.0, 343.0), (585.0, 316.0)]
    d2 = np.full_like(Y, 1e3)
    for a, b in zip(path[:-1], path[1:]):
        d2 = np.minimum(d2, cap_px(Y, Z, a, b, 0.95))
    return S.union(S.extrude(d2, np.abs(X), 0, 5.0, 0.6), tang)


def magcatch(X, Y, Z):
    d = S.rinter(F("magcatch")(Y, Z), np.abs(X) - 7.5, 1.0)
    d = np.maximum(d, np.where(Z < zb(346.0, 596), np.abs(X) - 5.5, -1e3))
    yp, zp = bp(598.0, 331.0)
    return S.union(d, S.cyl(X, Y, Z, "x", (yp, zp), 2.2, -8.6, 8.6, 0.5))


def grip(X, Y, Z):
    hw = P.HW_GRIP
    d = S.rinter(F("grip")(Y, Z), np.abs(X) - hw, 7.0)
    d = DT.apply(d, X, Y, Z, "grip", hw)
    # grip screw from below (163304): round counterbore in the bottom face with the slotted screw head inside
    (ya, za), (yb2, zb2) = bp(464.0, 423.0), bp(489.0, 427.8)
    ang = math.atan2(zb2 - za, yb2 - ya); yc, zc = (ya + yb2) / 2, (za + zb2) / 2
    u = (Y - yc) * math.cos(ang) + (Z - zc) * math.sin(ang); v = -(Y - yc) * math.sin(ang) + (Z - zc) * math.cos(ang)
    bore = np.maximum(np.hypot(X, u) - 3.4, -v - 7.0)
    d = S.rdiff(d, bore, 0.4)
    head = np.maximum(np.maximum(np.hypot(X, u) - 2.9, S.slab(v, -7.5, -2.8)), -np.maximum(np.abs(X) - 0.45, -v - 3.6))
    return S.union(d, head)


def magazine(X, Y, Z):
    hw = P.HW_MAG
    d = S.rinter(F("mag")(Y, Z), np.abs(X) - hw, 2.2)
    d = DT.apply(d, X, Y, Z, "mag", hw)
    # top (inside the magwell when inserted; seen in 163300 and the exploded / field-strip views): feed lips, the front
    # locking lug and the rear catch tab
    ztop = zb(312.0, 640)
    lips = S.box(X, Y, Z, (-7.5, yb(612.0), ztop - 2), (7.5, yb(660.0), ztop + 2.6), 1.0)
    lips = S.rdiff(lips, S.box(X, Y, Z, (-4.8, yb(608.0), ztop - 1), (4.8, yb(664.0), ztop + 4), 0.8), 0.4)
    flug = S.box(X, Y, Z, (-6.0, yb(675.0), ztop - 1.0), (6.0, yb(680.0), ztop + 5.0), 0.8)        # inside the magwell
    rtab = S.box(X, Y, Z, (-5.0, yb(606.5), ztop - 1.0), (5.0, yb(610.5), ztop + 4.0), 0.6)
    return S.union(d, lips, flug, rtab)


# ================================================================== stock
def stock_hw(Y):
    t = np.clip((Y - P.BUTT_Y) / (yb(452.0) - P.BUTT_Y), 0, 1)
    return lerp(P.HW_STOCK_BUTT, P.HW_STOCK_WRIST, S.sstep(0.25, 0.75, t))


def stock(X, Y, Z):
    hw = stock_hw(Y)
    d = S.rinter(F("stock")(Y, Z), np.abs(X) - hw, 6.5)
    d = DT.apply(d, X, Y, Z, "stock", hw)
    # hinge knuckle: the stock's steel-lined knuckle around the hinge barrel (middle band between the receiver knuckles)
    za, zc = zb(301.6, 456), zb(278.4, 456)
    kn = S.cyl(X, Y, Z, "z", (P.HINGE[0], P.HINGE[1]), 5.6, za, zc, 0.6)
    web = S.box(X, Y, Z, (P.HW_STOCK_WRIST - 6.0, yb(449.0), za), (P.HINGE[0], P.HINGE[1], zc), 0.8)
    kn = S.runion(kn, web, 1.0)
    kn = S.diff(kn, S.cyl(X, Y, Z, "z", (P.HINGE[0], P.HINGE[1]), 4.65, za - 1, zc + 1))
    return S.runion(d, kn, 0.8)


def buttplate(X, Y, Z):
    d = S.rinter(F("buttplate")(Y, Z), np.abs(X) - (P.HW_STOCK_BUTT - 0.3), 5.5)
    # folding latch hook on the left of the butt (L_d: x 1125..1145, rows 300..345; 163304): plate + hook lip
    (y0, z1), (y1, z0) = R.lp2b(1145.0, 300.0), R.lp2b(1125.0, 345.0)
    hook = S.box(X, Y, Z, (P.HW_STOCK_BUTT - 1.5, min(y0, y1), z0), (P.HW_STOCK_BUTT + 2.2, max(y0, y1), z1), 0.6)
    lipp = S.box(X, Y, Z, (P.HW_STOCK_BUTT + 0.5, max(y0, y1) - 3.0, z0), (P.HW_STOCK_BUTT + 5.0, max(y0, y1), z0 + 7.0), 0.6)
    d = S.union(d, S.runion(hook, lipp, 0.6))
    yk, zk = bp(196.0, 348.6)
    knob = S.cyl(X, Y, Z, "y", (0, zk), 5.9 * R.S_MM, yb(192.6, 348.6), yb(200.5, 348.6), 0.8)
    return S.union(d, knob)


def hinge(X, Y, Z):
    """stock hinge block between the stock front (x 452) and the receiver (x 458.5), rows 286..311; vertical hinge barrel
    on the left side"""
    blk = S.rinter(rect_px(Y, Z, 451.6, 285.6, 459.0, 311.2, 0.6), np.abs(X) - (P.HW_REC - 0.4), 1.0)
    z0, z1 = zb(318.0, 456), zb(262.0, 456)
    barrelh = S.cyl(X, Y, Z, "z", (P.HINGE[0], P.HINGE[1]), 4.6, z0, z1, 0.8)
    web = S.box(X, Y, Z, (P.HW_REC - 1.0, P.HINGE[1] - 3.6, z0 + 1.0), (P.HINGE[0], P.HINGE[1] + 3.6, z1 - 1.0), 0.8)
    # knuckle gaps: the stock's two knuckles ride between the receiver's (3 knuckles, 2 gaps)
    d = S.union(blk, S.runion(barrelh, web, 1.0))
    for zc in (zb(278.0, 456), zb(302.0, 456)):
        d = S.rdiff(d, np.maximum(np.abs(Z - zc) - 0.35, np.maximum(np.hypot(X - P.HINGE[0], Y - P.HINGE[1]) - 5.0,
                                                                      -np.hypot(X - P.HINGE[0], Y - P.HINGE[1]) + 2.0)), 0.15)
    return d


def sidrail(X, Y, Z):
    return DT.side_rail(X, Y, Z)


def slings(X, Y, Z):
    return DT.slings(X, Y, Z)


# ================================================================== build table
def table():
    T = {}
    def add(name, fn, xr, rr, hw, h, mat):
        cs = [bp(x, r) for x in xr for r in rr]
        y0, y1 = min(c[0] for c in cs) - 2, max(c[0] for c in cs) + 2
        z0, z1 = min(c[1] for c in cs) - 2, max(c[1] for c in cs) + 2
        T[name] = (fn, (-hw, y0, z0), (hw, y1, z1), h, mat)
    add("receiver", receiver, (452, 732), (252, 322), 18.0, 0.22, "park")
    add("dustcover", dustcover, (455, 706), (238, 272), 18.0, 0.2, "park")
    add("recoil", recoil, (452, 645), (254, 276), 7.0, 0.14, "park")
    add("rsb", rsb, (698, 782), (230, 275), 19.5, 0.18, "park")
    add("leaf", leaf, (693, 762), (227, 245), 9.0, 0.12, "park")
    add("slider", slider, (703, 729), (231, 248), 15.5, 0.12, "park")
    add("upperhg", upperhg, (778, 888), (236, 274), 18.0, 0.24, "poly")
    add("hg_rings_u", hg_rings_u, (772, 897), (236, 274), 17.5, 0.2, "park")
    add("hg_ring_l", hg_ring_l, (880, 900), (264, 310), 22.5, 0.2, "park")
    add("lowerhg", lowerhg, (722, 890), (265, 323), 23.5, 0.24, "poly")
    add("gastube", gastube, (766, 990), (244, 274), 12.0, 0.16, "park")
    add("gasblock", gasblock, (936, 1007), (245, 309), 12.0, 0.16, "park")
    add("barrel", barrel, (694, 1103), (270, 298), 11.0, 0.2, "park")
    add("rod", rod, (756, 1095), (294, 304), 4.0, 0.12, "park")
    add("fsb", fsb, (1054, 1097), (228, 310), 11.5, 0.14, "park")
    add("brake", brake, (1090, 1182), (270, 304), 14.5, 0.14, "park")
    add("carrier", carrier, (500, 972), (238, 281), 38.0, 0.18, "bright")
    add("bolt", bolt, (638, 710), (270, 300), 11.0, 0.14, "park")
    add("selector", selector, (505, 633), (255, 300), 22.0, 0.14, "park")
    add("trigger", trigger, (536, 562), (313, 348), 4.5, 0.1, "park")
    add("guard", guard, (524, 590), (313, 359), 6.5, 0.1, "park")
    add("magcatch", magcatch, (584, 610), (313, 361), 10.0, 0.1, "park")
    add("grip", grip, (450, 535), (313, 432), 16.0, 0.2, "poly")
    add("magazine", magazine, (594, 738), (308, 503), 16.0, 0.22, "magpoly")
    add("stock", stock, (196, 460), (274, 393), 28.0, 0.26, "poly")
    add("buttplate", buttplate, (190, 230), (281, 392), 27.0, 0.14, "poly")
    add("hinge", hinge, (446, 470), (257, 323), 27.0, 0.16, "park")
    add("siderail", sidrail, (720, 870), (270, 310), 0.0, 0.14, "park")
    T["siderail"] = (sidrail, (P.HW_REC - 1.0, yb(560.0) - 4, zb(312.0, 620) - 3), (P.HW_REC + 8.0, yb(700.0) + 4, zb(262.0, 620) + 3), 0.14, "park")
    add("slings", slings, (410, 460), (290, 335), 30.0, 0.12, "park")
    for k, v in DT.bake_shells().items():
        T[k] = v
    return T


def main():
    out = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("out=")), os.path.join(HERE, "work", "high.npz"))
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, (fn, lo, hi, h, mat) in table().items():
        if ONLY and name not in ONLY:
            continue
        off = h * HS * 0.3713
        V, Fc = S.mesh(fn, np.add(lo, -off), np.add(hi, off), h * HS, lip=3.5, name=name)
        if len(V) == 0:
            print("[parts] EMPTY", name, flush=True)
            continue
        m = DT.material_of(name, V, MAT[mat]) if hasattr(DT, "material_of") else np.full(len(V), MAT[mat])
        parts[name] = (V, Fc, {"mat": m})
    S.save(out + ".tmp.npz", parts)
    new = dict(np.load(out + ".tmp.npz"))
    if old:
        for k in list(old):
            if k.split("__")[0] in parts:
                old.pop(k)
        new.update(old)
    np.savez_compressed(out, **new)
    os.remove(out + ".tmp.npz")
    print("HIGH_DONE", sorted({k.split("__")[0] for k in new}))


if __name__ == "__main__":
    main()
