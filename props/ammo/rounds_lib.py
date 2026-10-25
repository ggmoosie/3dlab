"""Loose rounds and shotgun shells (bpy + numpy). Every calibre is a lathe profile in mm, built twice:
  detailed  SM_<Item>[_HP|_Slug]        hero / inspect / first person (~1-1.5k tris): rim, extractor groove, primer in its
                                        pocket, headstamp recess, case-mouth crimp, cannelure, ogive / HP cavity, star or
                                        roll crimp; normal map = painted relief (headstamp, knurl, stab crimp, skives)
  low       SM_<Item>[...]_Low          world pickup / in-magazine (~150-300 tris); normal + AO baked from the detailed mesh
Variants that share a shape (FMJ / green tip / AP) are material instances of the universal mask material:
  Mask R = tip paint zone (ColorA) · G = case / brass head (ColorB) · B = sealant ring or hull print (Label) · A = wear
Shotgun shells: R = plastic hull (ColorA), G = brass head (ColorB), B = hull print.
Pivot: base centre, +Z toward the bullet tip (like a cell in lab batteries); the round stands on its head.
Used by props/ammo/<round>/<round>.py:  rounds_lib.build("9mm", HERE)"""
import os, sys, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, ROOT)
import bpy, bmesh
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from mathutils import Matrix
from lab import scene, master, bake, export, tex, geo
from lab.labels import Label

FAST = "--fast" in sys.argv
MM = 0.001
FONT = os.path.join(ROOT, "fonts/IBMPlexMono-Bold.ttf")

# ---------------------------------------------------------------- colours (linear)
BRASS = (0.78, 0.55, 0.20)
NICKEL = (0.66, 0.66, 0.63)
LACQ_STEEL = (0.13, 0.14, 0.10)
COPPER = (0.74, 0.33, 0.15)
STEEL = (0.50, 0.51, 0.53)
LEAD = (0.22, 0.22, 0.23)
GREEN_TIP = (0.03, 0.26, 0.05)
BLACK_TIP = (0.015, 0.015, 0.015)


def case_brass(c=BRASS):
    return dict(ColorB=c, RoughB=0.26, MetalB=1.0)


def tip(c, metal=False):
    return dict(ColorA=c, RoughA=0.3 if metal else 0.5, MetalA=1.0 if metal else 0.0)


# ---------------------------------------------------------------- calibres (mm). Real references in "real".
SPECS = {
    "9mm": dict(item="Round9mm", rows=["9mmRound", "9mmRoundHP", "9mmRoundAP"], real="9x19 mm Parabellum, 124 gr FMJ (C.I.P.)",
                R_rim=4.97, rim_h=1.27, R_groove=3.95, groove_top=2.0, R_body0=4.96, z_body0=2.75, sh=None,
                R_mouth=4.83, z_mouth=19.15, R_b=4.51, oal=29.69, nose="round", z_og=21.6, meplat=1.0, primer=2.24,
                cannelure=None, paint_z=26.8, stamp=("PA", "9x19"), segs=28, segs_low=12, tex=1024, tex_low=512,
                shapes={"": dict(variants={
                    "FMJ": dict(tip(COPPER, True), **case_brass()),
                    "AP": dict(tip(STEEL, True), **case_brass(NICKEL), seal="#8a1a1a")}),
                        "_HP": dict(oal=29.2, meplat=2.7, cavity=(2.15, 3.4), z_og=21.6, variants={
                            "HP": dict(tip(COPPER, True), **case_brass())})}),
    "556": dict(item="Round556", rows=["556Round", "556RoundAP", "556RoundHP"], real="5.56x45 mm NATO, 62 gr (C.I.P.)",
                R_rim=4.80, rim_h=1.14, R_groove=3.90, groove_top=1.95, R_body0=4.79, z_body0=2.7,
                sh=(4.50, 36.5, 3.21, 39.6), R_mouth=3.21, z_mouth=44.7, R_b=2.85, oal=57.4, nose="ogive", z_og=47.4,
                meplat=0.35, primer=2.24, cannelure=45.6, paint_z=53.4, stamp=("PA", "5.56"), crimp=True,
                segs=26, segs_low=12, tex=1024, tex_low=512,
                shapes={"": dict(variants={
                    "FMJ": dict(tip(COPPER, True), **case_brass()),
                    "GreenTip": dict(tip(GREEN_TIP), **case_brass(), seal="#5a1f6e"),
                    "AP": dict(tip(BLACK_TIP), **case_brass(), seal="#1a1a1a")}),
                        "_HP": dict(oal=57.0, meplat=1.05, cavity=(0.75, 2.6), variants={
                            "HP": dict(tip(COPPER, True), **case_brass())})}),
    "762": dict(item="Round762", rows=["762Round"], real="7.62x51 mm NATO, 147 gr ball (C.I.P.)",
                R_rim=5.97, rim_h=1.37, R_groove=4.90, groove_top=2.3, R_body0=5.97, z_body0=3.2,
                sh=(5.77, 39.6, 4.36, 43.5), R_mouth=4.36, z_mouth=51.2, R_b=3.91, oal=71.1, nose="ogive", z_og=55.0,
                meplat=0.45, primer=2.67, cannelure=52.4, paint_z=65.6, stamp=("PA", "7.62"), crimp=True,
                segs=26, segs_low=12, tex=1024, tex_low=512,
                shapes={"": dict(variants={
                    "FMJ": dict(tip(COPPER, True), **case_brass()),
                    "AP": dict(tip(BLACK_TIP), **case_brass(), seal="#1a1a1a")})}),
    "792": dict(item="Round792", rows=["792Round (new)"], real="7.92x57 mm Mauser (8 mm Mauser), 198 gr s.S. spitzer",
                R_rim=5.97, rim_h=1.30, R_groove=5.0, groove_top=2.3, R_body0=5.97, z_body0=3.2,
                sh=(5.46, 45.9, 4.54, 48.6), R_mouth=4.54, z_mouth=57.0, R_b=4.11, oal=80.5, nose="ogive", z_og=61.5,
                meplat=0.5, primer=2.67, cannelure=None, paint_z=74.0, stamp=("PA", "7.92"),
                segs=26, segs_low=12, tex=1024, tex_low=512,
                shapes={"": dict(variants={
                    "FMJ": dict(tip(COPPER, True), **case_brass()),
                    "AP": dict(tip(BLACK_TIP), ColorB=LACQ_STEEL, RoughB=0.42, MetalB=0.35, seal="#7a1414")})}),
    "12ga": dict(item="Shell12ga", rows=["ShotgunShell", "ShotgunSlug"], real="12 gauge 2 3/4 in shell, star crimp (buck/bird) and roll crimp (slug)",
                 kind="shell", R_rim=11.3, rim_h=1.4, R_brass=10.45, z_brass=16.0, R_hull=10.25, top=59.0, primer=3.07,
                 stamp=("PATRIOT", "12 GA"), segs=32, segs_low=16, tex=1024, tex_low=512,
                 shapes={"": dict(crimp="star", variants={
                     "Buck": dict(ColorA=(0.42, 0.025, 0.02), RoughA=0.38, MetalA=0.0, **case_brass(), print=("00 BUCK", "9 PELLETS", "#f2efe6")),
                     "Bird": dict(ColorA=(0.04, 0.20, 0.06), RoughA=0.38, MetalA=0.0, **case_brass(NICKEL), print=("#7 1/2 SHOT", "1 1/8 OZ", "#f2efe6"))}),
                         "_Slug": dict(crimp="roll", variants={
                             "Slug": dict(ColorA=(0.02, 0.05, 0.22), RoughA=0.38, MetalA=0.0, **case_brass(), print=("RIFLED SLUG", "1 OZ", "#f2efe6"))})}),
}


# ---------------------------------------------------------------- profiles
def nose_pts(sp, n, z0, oal, meplat):
    """Bullet nose from (R_b, z0) to (meplat, oal). round = elliptical; ogive = tangent ogive, truncated at the meplat."""
    Rb, L = sp["R_b"], oal - z0
    pts = []
    if sp["nose"] == "round":
        fe = math.acos(meplat / Rb)
        for k in range(1, n + 1):
            f = fe * k / n
            pts.append((Rb * math.cos(f), z0 + L * math.sin(f) / math.sin(fe)))
    else:
        Ls = L * 1.03                                         # sharp ogive slightly longer; truncate at the meplat
        rho = (Rb ** 2 + Ls ** 2) / (2 * Rb)
        r_of = lambda zp: math.sqrt(max(rho ** 2 - zp ** 2, 0)) - (rho - Rb)
        lo, hi = 0.0, Ls                                      # z' where r = meplat
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if r_of(mid) > meplat else (lo, mid)
        ze = lo
        for k in range(1, n + 1):
            t = k / n
            zp = ze * (1 - (1 - t) ** 1.45)
            pts.append((r_of(zp) if k < n else meplat, z0 + zp * L / ze if k < n else oal))
    return pts


def cart_sections(sp, shape, low):
    """Rimless cartridge: base disc (flip), case side, bullet side. Returns [(pts_mm, mode, flip, tag, extra)]."""
    oal, mep, z_og = shape.get("oal", sp["oal"]), shape.get("meplat", sp["meplat"]), shape.get("z_og", sp["z_og"])
    cav = shape.get("cavity")
    R, rh, rp, zm, Rb = sp["R_rim"], sp["rim_h"], sp["primer"], sp["z_mouth"], sp["R_b"]
    if low:
        base = [(0, 0), (R - 0.35, 0), (R, 0.35)]
        side = [(R, 0.35)]
        if sp["sh"]:
            r1, z1, r2, z2 = sp["sh"]
            side += [(r1, z1), (r2, z2)]
        side += [(sp["R_mouth"], zm), (Rb, zm)]
        bullet = [(Rb, zm), (Rb, z_og)] + nose_pts(sp, 3, z_og, oal, mep)
        if cav:
            cr, cd = cav
            bullet += [(cr, oal - cd * 0.75), (0, oal - cd)]
        else:
            bullet += [(0, oal)]
    else:
        base = [(0, 0.10), (rp - 0.12, 0.13), (rp + 0.03, 0.42), (rp + 0.13, 0.0),
                (R - 0.36, 0.0), (R, 0.36)]
        side = [(R, 0.36), (R, rh - 0.18), (R - 0.18, rh), (sp["R_groove"], rh + 0.22),
                (sp["R_groove"], sp["groove_top"]), (sp["R_body0"] - 0.03, sp["z_body0"])]
        if sp["sh"]:
            r1, z1, r2, z2 = sp["sh"]
            side += [(r1, z1 - 1.0), (r1 - 0.06, z1), (r2, z2 + 0.2)]
        Rm = sp["R_mouth"]
        side += [(Rm, zm - 0.9), (Rm - 0.07, zm - 0.35), (Rm - 0.1, zm), (Rb + 0.01, zm)]
        bullet = [(Rb, zm)]
        if sp["cannelure"]:
            zc = sp["cannelure"]
            bullet += [(Rb, zc - 0.45), (Rb - 0.12, zc), (Rb, zc + 0.45)]
        bullet += [(Rb, z_og)] + nose_pts(sp, 6, z_og, oal, mep)
        if cav:
            cr, cd = cav
            bullet += [(mep - 0.18, oal - 0.12), (cr, oal - 0.7), (cr * 0.6, oal - cd), (0, oal - cd - 0.05)]
        else:
            bullet += [(mep * 0.5, oal + 0.02), (0, oal + 0.03)]
    return [(base, "disc", True, "base"), (side, "side", False, "case"), (bullet, "side", False, "bullet")]


def shell_sections(sp, shape, low):
    R, rh, rb, zb, Rh, top, rp = sp["R_rim"], sp["rim_h"], sp["R_brass"], sp["z_brass"], sp["R_hull"], sp["top"], sp["primer"]
    star = shape["crimp"] == "star"
    if low:
        base = [(0, 0), (R - 0.5, 0), (R, 0.4)]
        if star:
            side = [(R, 0.4), (R, rh), (rb, rh + 0.1), (rb, zb), (Rh, zb + 0.1), (Rh, top - 1.4), (Rh - 1.0, top)]
            topd = [(Rh - 1.0, top), (4.0, top - 0.6), (0, top - 1.2)]
            return [(base, "disc", True, "base"), (side, "side", False, "case"), (topd, "disc", False, "top")]
        side = [(R, 0.4), (R, rh), (rb, rh + 0.1), (rb, zb), (Rh, top - 1.4)]
        roll = [(Rh, top - 1.4), (Rh - 0.6, top), (Rh - 1.4, top - 1.2)]
        slug = [(Rh - 1.4, top - 1.2), (5.0, top + 0.8), (0, top + 1.2)]
        return [(base, "disc", True, "base"), (side, "side", False, "case"), (roll, "side", False, "roll"), (slug, "disc", False, "slug")]
    base = [(0, 0.15), (rp - 0.3, 0.15), (rp - 0.05, 0.22), (rp + 0.05, 0.5), (rp + 0.22, 0.06), (rp + 0.7, 0.04),
            (R - 0.5, 0.0), (R, 0.4)]
    side = [(R, 0.4), (R, rh - 0.25), (R - 0.25, rh), (rb + 0.1, rh + 0.06), (rb, rh + 0.3), (rb, zb - 0.35),
            (rb - 0.12, zb), (Rh + 0.02, zb + 0.05), (Rh, zb + 0.45)]
    if star:
        side += [(Rh, top - 1.5), (Rh - 0.12, top - 0.75), (Rh - 0.5, top - 0.18), (Rh - 1.2, top)]
        topd = [(Rh - 1.2, top), (Rh - 2.3, top - 0.22), (7.0, top - 0.5), (4.6, top - 0.8), (2.4, top - 1.05), (1.0, top - 1.25),
                (0.45, top - 1.55), (0, top - 1.6)]
        return [(base, "disc", True, "base"), (side, "side", False, "case"), (topd, "disc", False, "top")]
    side += [(Rh, top - 1.4)]
    roll = [(Rh, top - 1.4), (Rh - 0.12, top - 0.55), (Rh - 0.55, top), (Rh - 1.05, top - 0.2), (Rh - 1.35, top - 0.85),
            (Rh - 1.45, top - 1.35)]
    rs = Rh - 1.62
    slug = [(Rh - 1.45, top - 1.35), (rs, top - 1.45), (rs - 0.3, top - 0.7), (6.8, top + 0.25), (5.0, top + 0.8),
            (3.0, top + 1.1), (1.2, top + 1.2), (0, top + 1.22)]
    return [(base, "disc", True, "base"), (side, "side", False, "case"), (roll, "side", False, "roll"), (slug, "disc", False, "slug")]


def sections(sp, shape, low):
    return shell_sections(sp, shape, low) if sp.get("kind") == "shell" else cart_sections(sp, shape, low)


# ---------------------------------------------------------------- UV packing (shelf packer, one texel density)
def arc(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def pack(secs, gap=0.012):
    items = []
    for i, (pts, mode, flip, tag) in enumerate(secs):
        rmax = max(p[0] for p in pts)
        if mode == "side":
            items.append((i, 2 * math.pi * rmax, arc(pts)))
        else:
            items.append((i, 2 * rmax, 2 * rmax))

    def place(k):
        rects, x, y, rowh = {}, gap, gap, 0.0
        for i, w, h in sorted(items, key=lambda t: -t[2]):
            W, H = w * k, h * k
            if x + W > 1 - gap:
                x, y, rowh = gap, y + rowh + gap, 0.0
            if W > 1 - 2 * gap:
                return None
            rects[i] = (x, y, x + W, y + H)
            x += W + gap
            rowh = max(rowh, H)
        return rects if y + rowh <= 1 - gap else None
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if place(mid) else (lo, mid)
    rects = place(lo)
    return [rects[i] for i in range(len(secs))], lo


def build_mesh(name, secs, rects, segs):
    sec = []
    for (pts, mode, flip, tag), r in zip(secs, rects):
        d = dict(pts=[(a * MM, b * MM) for a, b in pts], uv=r, mode=mode)
        if flip:
            d["flip"] = True
        sec.append(d)
    o = geo.lathe(name, sec, segs=segs)
    return o


def star_folds(o, sp, secs):
    """Fold the star-crimp face: 8 V creases, deepest mid-radius, plus a slight swirl. Detailed mesh only."""
    topd = [s for s in secs if s[3] == "top"][0][0]
    r0 = topd[0][0] * MM
    bm = bmesh.new(); bm.from_mesh(o.data)
    ztop = sp["top"] * MM
    moved = set()
    for v in bm.verts:
        r = math.hypot(v.co.x, v.co.y)
        if r < r0 - 1e-6 and v.co.z > ztop - 0.002 and r > 1e-6:
            th = math.atan2(v.co.y, v.co.x)
            ph = (th / (math.pi / 4)) % 1.0
            tri = 1 - abs(ph * 2 - 1)                         # 0 at the crease, 1 between creases
            amp = 0.75 * MM * math.sin(math.pi * min(r / r0, 1.0)) ** 0.8
            v.co.z -= amp * (1 - tri) * 0.9
            moved.add(v)
    faces = [f for f in bm.faces if all(v in moved or math.hypot(v.co.x, v.co.y) < r0 + 1e-6 for v in f.verts)
             and f.calc_center_median().z > ztop - 0.002]
    bmesh.ops.triangulate(bm, faces=faces, quad_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True


# ---------------------------------------------------------------- texel maps (texel → region, r, z, θ in mm)
def texel_maps(secs, rects, S):
    u, v = tex.uv_grid(S)
    reg = np.full((S, S), -1, np.int16)
    R, Z, TH, SS, X, Y = (np.zeros((S, S), np.float32) for _ in range(6))
    for i, ((pts, mode, flip, tag), rect) in enumerate(zip(secs, rects)):
        m = tex.in_rect(u, v, rect)
        if mode == "side":
            th, s, r, z = tex.side_map(u, v, rect, pts)
            x, y = r * np.cos(th), r * np.sin(th)
        else:
            rmax = max(p[0] for p in pts)
            x, y = tex.disc_map(u, v, rect, rmax, flip)
            r = np.hypot(x, y)
            m = m & (r <= rmax * 1.02)
            srt = sorted(pts)
            z = np.interp(r, [p[0] for p in srt], [p[1] for p in srt])
            th = np.arctan2(y, x)
            acc = [0.0]
            for a, b in zip(pts, pts[1:]):
                acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            s = np.interp(r, [p[0] for p in srt], [acc[pts.index(p)] for p in srt])
        reg[m] = i
        for A, B in ((R, r), (Z, z), (TH, th % (2 * np.pi)), (SS, s), (X, x), (Y, y)):
            A[m] = B[m]
    tags = [s[3] for s in secs]
    return dict(reg=reg, r=R, z=Z, th=TH, s=SS, x=X, y=Y, tags=tags, S=S)


def isin(T, tag):
    return T["reg"] == T["tags"].index(tag) if tag in T["tags"] else np.zeros_like(T["reg"], bool)


def dilate(a, valid, n=6):
    """Bleed island values outward n px (no seams in mips)."""
    a = a.copy()
    ok = valid.copy()
    for _ in range(n):
        acc = np.zeros_like(a, np.float32)
        cnt = np.zeros(ok.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(np.roll(ok, dy, 0), dx, 1)
            sa = np.roll(np.roll(a, dy, 0), dx, 1)
            acc += np.where(sh[..., None] if a.ndim == 3 else sh, sa, 0)
            cnt += sh
        grow = (~ok) & (cnt > 0)
        a = np.where(grow[..., None] if a.ndim == 3 else grow, acc / np.maximum(cnt, 1)[..., None] if a.ndim == 3 else acc / np.maximum(cnt, 1), a)
        ok = ok | grow
    return a


# ---------------------------------------------------------------- relief (headstamp, knurl, crimp, skives)
def headstamp(T, sp, depth=0.09):
    """Stamped letters on the head between primer pocket and rim; read correctly seen from below (canvas x = -world x)."""
    R = sp["R_rim"]
    px = 60                                                    # canvas px per mm
    n = int(2 * R * px)
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    c = n / 2
    rp = sp["primer"]
    rad = (rp + 0.2 + R - 0.4) / 2                             # mid of the stamp ring
    hsz = (R - 0.45 - rp - 0.25) * 0.85
    f = ImageFont.truetype(FONT, max(8, int(hsz * px)))
    for k, (txt, ang) in enumerate(((sp["stamp"][0], 90), (sp["stamp"][1], -90))):
        g = Image.new("L", (n, n), 0)
        ImageDraw.Draw(g).text((c, c - rad * px if ang == 90 else c + rad * px), txt, font=f, fill=255, anchor="mm")
        im = Image.fromarray(np.maximum(np.asarray(im), np.asarray(g)))
    if sp.get("crimp"):                                        # military primer stab crimp: 3 small dents
        for a in (90, 210, 330):
            x, y = c + math.cos(math.radians(a)) * (rp + 0.35) * px, c - math.sin(math.radians(a)) * (rp + 0.35) * px
            ImageDraw.Draw(im).ellipse([x - 0.22 * px, y - 0.22 * px, x + 0.22 * px, y + 0.22 * px], fill=255)
    im = im.filter(ImageFilter.GaussianBlur(px * 0.04))
    a = np.asarray(im, np.float32) / 255
    cx = np.clip(((-T["x"]) * px + c).astype(int), 0, n - 1)
    cy = np.clip(((-T["y"]) * px + c).astype(int), 0, n - 1)
    return -a[cy, cx] * depth * isin(T, "base")


def relief(T, sp, shape, seed):
    S = T["S"]
    H = headstamp(T, sp) if sp.get("kind") != "shell" else headstamp(T, sp, 0.05)
    th, z, r = T["th"], T["z"], T["r"]
    H += (tex.fbm(S, S, 40, seed) - 0.5) * 0.004                       # micro dents
    if sp.get("kind") == "shell":
        hull = isin(T, "case") & (z > sp["z_brass"] + 0.5)
        H += np.where(hull, np.cos(th * 90) * 0.004, 0)                # extruded hull ribbing
        if shape["crimp"] == "roll":                                   # slug ribs show on the nose shoulder
            sl = isin(T, "slug") & (r > 5.2)
            H += np.where(sl, (np.cos(th * 12 + r * 0.9) > 0.55) * 0.08, 0)
        else:
            tp = isin(T, "top")
            ph = (th / (np.pi / 4)) % 1.0
            H += np.where(tp, -np.exp(-((ph - 0.0) ** 2) / 0.002) * 0.05 - np.exp(-((ph - 1.0) ** 2) / 0.002) * 0.05, 0)
        return H
    bl = isin(T, "bullet")
    if sp["cannelure"]:
        zc = sp["cannelure"]
        band = bl & (np.abs(z - zc) < 0.25)
        H += np.where(band, (np.cos(th * 140) > 0) * -0.03, 0)         # knurled cannelure
    cav = shape.get("cavity")
    if cav:                                                            # six skive cuts down the nose
        oal = shape.get("oal", sp["oal"])
        near = bl & (z > oal - (5.0 if sp["nose"] == "round" else 3.5))
        sk = np.minimum(np.abs(((th / (np.pi / 3)) % 1.0) - 0.5) * 2, 1)
        H += np.where(near, -np.exp(-((1 - sk) ** 2) / 0.004) * 0.06, 0)
    case = isin(T, "case")
    H += np.where(case & (np.abs(z - (sp["z_mouth"] - 0.35)) < 0.25), -0.01, 0)    # crimp ring line
    return H


# ---------------------------------------------------------------- texture sets
def textures(T, secs, sp, shape, ao, nrm_bake, item, texd, seed):
    S = T["S"]
    reg, r, z = T["reg"], T["r"], T["z"]
    valid = reg >= 0
    base, case, bl = isin(T, "base"), isin(T, "case"), isin(T, "bullet")
    noise = tex.fbm(S, S, 16, seed + 1)
    shell = sp.get("kind") == "shell"
    rp = sp["primer"]
    primer = base & (r < rp + 0.06)
    if shell:
        top_or_roll = isin(T, "top") | isin(T, "roll")
        slug = isin(T, "slug")
        brass = (base & ~primer) | (case & (z < sp["z_brass"] + 0.06))
        hull = (case & ~brass) | top_or_roll
        col = np.where(hull[..., None], np.float32(0.62), np.float32(0.0)) + 0 * np.zeros((S, S, 3), np.float32)
        col = np.where(brass[..., None], np.array(BRASS, np.float32), col)
        col = np.where(primer[..., None], np.array((0.52, 0.46, 0.34), np.float32), col)
        col = np.where(slug[..., None], np.array(LEAD, np.float32) * (0.85 + 0.3 * noise[..., None]), col)
        rough = np.where(hull, 0.42, np.where(slug, 0.55, 0.3)).astype(np.float32)
        metal = np.where(hull, 0.0, np.where(slug, 0.75, 1.0)).astype(np.float32)
        mR, mG = hull.astype(np.float32), brass.astype(np.float32)
        mB = (case & (z > sp["z_brass"] + 2.0) & (z < sp["top"] - 2.5)).astype(np.float32)
        wear = np.where(brass, np.clip((z < sp["rim_h"] + 0.3) * (noise - 0.2) * 2.0, 0, 1), 0)
        wear += np.where(hull, np.clip(tex.fbm(S, S, 30, seed + 5) - 0.62, 0, 1) * 3, 0)
    else:
        oal = shape.get("oal", sp["oal"])
        cavity = np.zeros_like(bl)
        if shape.get("cavity"):
            bp = [q for q in secs if q[3] == "bullet"][0][0]
            mep = shape.get("meplat", sp["meplat"])
            it = [i for i, q in enumerate(bp) if abs(q[0] - mep) < 1e-6 and abs(q[1] - oal) < 1e-6][0]
            cavity = bl & (T["s"] > arc(bp[:it + 1]) + 0.02)
        casem = case | (base & ~primer)
        col = np.zeros((S, S, 3), np.float32) + np.array(BRASS, np.float32)
        col = np.where(bl[..., None], np.array(COPPER, np.float32), col)
        col = np.where(cavity[..., None], np.array(LEAD, np.float32), col)
        col = np.where(primer[..., None], np.array((0.52, 0.46, 0.34), np.float32), col)
        tarn = tex.fbm(S, S, 10, seed + 2)
        col = col * (0.9 + 0.18 * tarn[..., None])
        rough = (np.where(cavity, 0.55, 0.27) + 0.12 * (tarn - 0.5)).astype(np.float32)
        metal = np.where(cavity, 0.7, 1.0).astype(np.float32)
        mR = (bl & ~cavity & (z >= sp["paint_z"])).astype(np.float32)
        mG = casem.astype(np.float32)
        mB = ((base & (np.abs(r - (rp + 0.07)) < 0.2)) | (bl & (z < sp["z_mouth"] + 0.35))).astype(np.float32)   # sealant
        wear = np.where(casem, np.clip(((z < sp["rim_h"] + 0.4) | (z > sp["z_mouth"] - 1.2)) * (noise - 0.3) * 2.5, 0, 1), 0)
        wear += np.where(bl & ~cavity, np.clip(tex.fbm(S, S, 24, seed + 6) - 0.55, 0, 1) * 2.5 * (z < oal - 0.6), 0)
    H = relief(T, sp, shape, seed)
    px_mm = S * _density
    nrm = tex.height_to_normal(tex.blur(H, 0.6), px_mm)
    if nrm_bake is not None:
        nrm = tex.blend_normals(nrm_bake, nrm)
    one = np.ones((S, S), np.float32)
    ao = np.where(valid, ao, 1.0)
    out = dict(BaseColor=tex.lin_to_srgb(dilate(col, valid)), Normal=dilate(nrm, valid),
               ORM=np.stack([dilate(ao, valid), dilate(rough, valid), dilate(metal, valid)], -1),
               Mask=np.stack([dilate(mR, valid), dilate(mG, valid), dilate(mB, valid), dilate(np.clip(wear, 0, 1), valid)], -1))
    paths = {}
    for k, a in out.items():
        paths[k] = tex.save(a, os.path.join(texd, f"T_{item}_{k}.png"), "RGBA" if k == "Mask" else None)
    return paths, dict(mB=mB, hull_print=(mB > 0))


_density = 0.0


def variant_label(T, sp, shape, prm, S, seed):
    """Per-variant print: primer/mouth sealant colour (rounds) or the hull print (shells). None = no print."""
    if sp.get("kind") == "shell":
        txt1, txt2, ink = prm["print"]
        circ, z0, z1 = 2 * math.pi * sp["R_hull"], sp["z_brass"] + 2.0, sp["top"] - 2.5
        L = Label((z1 - z0) / 10, circ / 10, 120)
        h = L.h
        for k in range(2):                                     # printed twice around the hull
            y = h * (0.25 + 0.5 * k)
            L.text(0.15, y - 0.55, "PATRIOT ARMS CO.", "sans", 0.30, ink, "lm", max_w=(z1 - z0) / 10 - 0.3)
            L.text(0.15, y - 0.12, f"12 GA  2 3/4 IN  {txt1}", "mono_b", 0.21, ink, "lm", max_w=(z1 - z0) / 10 - 0.3)
            L.text(0.15, y + 0.25, f"{txt2} · LOT 26-118", "mono", 0.16, ink, "lm", max_w=(z1 - z0) / 10 - 0.3)
        art = L.array()
        from lab.labels import age
        art = age(art, seed, fade=0.05, scuff=0.15)
        ah, aw = art.shape[:2]
        lx = np.clip(((T["z"] - z0) / (z1 - z0) * aw).astype(int), 0, aw - 1)
        ly = np.clip((T["th"] / (2 * np.pi) * ah).astype(int), 0, ah - 1)
        out = art[ly, lx]
        m = isin(T, "case") & (T["z"] > z0) & (T["z"] < z1)
        out = out * m[..., None]
        return out
    seal = prm.get("seal")
    if not seal:
        return None
    c = np.array(tex.hex2rgb(seal), np.float32)
    rp = sp["primer"]
    ring = isin(T, "base") & (np.abs(T["r"] - (rp + 0.07)) < 0.16)
    mouth = isin(T, "bullet") & (T["z"] < sp["z_mouth"] + 0.3)
    a = (ring | mouth).astype(np.float32) * 0.95
    out = np.zeros((S, S, 4), np.float32)
    out[..., :3] = c
    out[..., 3] = a
    return out


# ---------------------------------------------------------------- build
def build(cal, here):
    global _density
    sp = SPECS[cal]
    item = sp["item"]
    texd, fbxd, rend = (os.path.join(here, d) for d in ("textures", "fbx", "renders"))
    work = os.path.join(ROOT, "work", "ammo_" + cal)
    os.makedirs(work, exist_ok=True)
    scene.reset()
    col = scene.collection("Game")
    colc = scene.collection("Collision")
    meshes, mats_all, man_parts, tris, variants_all = {}, {}, {}, {}, {}
    xoff = 0.0
    for sk, shape in sp["shapes"].items():
        name = f"SM_{item}{sk}"
        sec_d, sec_l = sections(sp, shape, False), sections(sp, shape, True)
        rect_d, k_d = pack(sec_d, 0.012)
        rect_l, k_l = pack(sec_l, 0.02)
        det = scene.link(build_mesh(name, sec_d, rect_d, sp["segs"]), col)
        if sp.get("kind") == "shell" and shape["crimp"] == "star":
            star_folds(det, sp, sec_d)
        geo.weighted_normals(det)
        # bakes: detailed AO from itself (before the low exists); low normal + AO from the detailed mesh
        Sd, Sl = sp["tex"], sp["tex_low"]
        smp = 8 if FAST else 32
        ao_d = bake.bake(det, [], "AO", Sd, f"{work}/ao{sk}.png", samples=smp)
        low = scene.link(build_mesh(name + "_Low", sec_l, rect_l, sp["segs_low"]), col)
        geo.weighted_normals(low)
        ao_l = bake.bake(low, [det], "AO", Sl, f"{work}/ao{sk}_low.png", samples=smp, extrusion=0.0008, max_ray=0.0025)
        n_l = bake.bake(low, [det], "NORMAL", Sl, f"{work}/n{sk}_low.png", samples=1, extrusion=0.0008, max_ray=0.0025)
        Td, Tl = texel_maps(sec_d, rect_d, Sd), texel_maps(sec_l, rect_l, Sl)
        _density = k_d
        pd, _ = textures(Td, sec_d, sp, shape, np.clip(ao_d, 0, 1), None, f"{item}{sk}", texd, 3)
        _density = k_l
        nb = np.where((n_l.sum(-1) < 0.05)[..., None], np.array([0.5, 0.5, 1.0], np.float32), n_l)
        pl, _ = textures(Tl, sec_l, sp, shape, np.where(ao_l < 0.02, 1.0, np.clip(ao_l, 0, 1)), nb, f"{item}{sk}_Low", texd, 3)
        vmats = {}
        for vi, (vk, prm) in enumerate(shape["variants"].items()):
            params = {k: v for k, v in prm.items() if k not in ("seal", "print")}
            params.update(LabelRough=0.5, WearAmount=0.35, Dirt=0.35)
            for lvl, TT, pp, SS in (("", Td, pd, Sd), ("_Low", Tl, pl, Sl)):
                lab = variant_label(TT, sp, shape, prm, SS, vi)
                lp = tex.save(lab, os.path.join(texd, f"T_{item}{sk}{lvl}_Label_{vk}.png"), "RGBA") if lab is not None else None
                m = master.material(f"MI_{item}{sk}{lvl}_{vk}", pp, params, label=lp)
                m.use_fake_user = True
                vmats[(vk, lvl)] = (m, lp)
            variants_all[vk] = dict(mesh=name, params=params,
                                    label=os.path.basename(vmats[(vk, "")][1]) if vmats[(vk, "")][1] else None,
                                    label_low=os.path.basename(vmats[(vk, "_Low")][1]) if vmats[(vk, "_Low")][1] else None)
        first = list(shape["variants"])[0]
        for o, lvl in ((det, ""), (low, "_Low")):                  # bake left a target slot: assign into it
            if o.data.materials:
                o.data.materials[0] = vmats[(first, lvl)][0]
            else:
                o.data.materials.append(vmats[(first, lvl)][0])
        hd = scene.link(export.hull(det, f"UCX_{name}_00", 24), colc)
        hl = scene.link(export.hull(low, f"UCX_{name}_Low_00", 16), colc)
        for h in (hd, hl):
            h.hide_render = True
        export.fbx([det, hd], f"{fbxd}/{name}.fbx")
        export.fbx([low, hl], f"{fbxd}/{name}_Low.fbx")
        tris[name], tris[name + "_Low"] = scene.tris(det), scene.tris(low)
        lo, hi = scene.bounds([det])
        man_parts[name] = dict(dims_cm=[round((hi[i] - lo[i]) * 100, 2) for i in range(3)], texture=Sd, low=name + "_Low", texture_low=Sl,
                               variants=list(shape["variants"]))
        meshes[sk] = (det, low, hd, hl, vmats)
        xoff += 0.06                                         # park finished shapes away from the origin (bakes, QC)
        for o in (det, hd):
            o.location.x = xoff
        low.location.x = hl.location.x = xoff + 0.03
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(here, os.path.basename(here) + ".blend"), relative_remap=True)
    hero(sp, meshes, rend, os.path.basename(here))
    export.manifest(os.path.join(here, "manifest.json"), {
        "item": item, "replaces_rows": sp["rows"], "category": "ammo", "clip": "AS_Machete_Attack_02 (pickup/use as in DT_Ammo)",
        "real_object": sp["real"], "pivot": "base centre (head of the case on the ground), +Z toward the tip",
        "meshes": man_parts, "tris": tris,
        "textures": {"normal": "OpenGL (flip green in UE)", "sets": "one per mesh: T_<Mesh>_{BaseColor,Normal,ORM,Mask} + T_<Mesh>_Label_<Variant>",
                     "detailed": sp["tex"], "low": sp["tex_low"]},
        "variants": variants_all,
        "mask": "R tip paint / hull · G case / brass head · B sealant ring or hull print · A wear" ,
        "collision": [f"UCX_{n}_00" for n in man_parts] + [f"UCX_{n}_Low_00" for n in man_parts],
        "lods": "none (detailed < 2k tris); the _Low mesh is the world / in-magazine version with the detail baked in",
        "notes": "detailed = hero / inspect / first person; _Low = world pickup and rounds shown in magazines or boxes"})
    print("TRIS", tris)
    return tris


def hero(sp, meshes, rend, stem):
    """Detailed variants standing in a row, the low version beside each shape, one detailed round lying head-out."""
    objs = []
    x = 0.0
    big = sp.get("kind") == "shell"
    pitch = (sp["R_rim"] * 2 + (5 if big else 4)) * MM
    for sk, (det, low, hd, hl, vmats) in meshes.items():
        for (vk, lvl), (m, _) in vmats.items():
            src = det if lvl == "" else low
            d = src.copy(); d.data = src.data.copy(); d.data.materials[0] = m
            d.location = (x, 0, 0); d.rotation_euler = (0, 0, math.radians(20))
            bpy.context.scene.collection.objects.link(d)
            objs.append(d)
            x += pitch * (1.0 if lvl == "" else 1.35)
        x += pitch * 0.4
    # one detailed round lying in front, head toward the camera-left so the headstamp reads
    det0, _, _, _, vm0 = list(meshes.values())[0]
    d = det0.copy(); d.data = det0.data.copy(); d.data.materials[0] = list(vm0.values())[0][0]
    d.rotation_euler = (0, math.radians(-90), math.radians(-40))
    L = (sp["top"] if sp.get("kind") == "shell" else sp["oal"]) * MM
    d.location = (x * 0.22 + L * 0.3, -max(sp["R_rim"] * MM * 5, L * 0.75) - 0.01, sp["R_rim"] * MM)
    bpy.context.scene.collection.objects.link(d)
    objs.append(d)
    for o in list(bpy.data.collections["Game"].objects) + list(bpy.data.collections["Collision"].objects):
        o.hide_render = True
    scene.studio(objs)
    scene.frame(objs, yaw=-14, pitch=20, lens=85, fill=0.88, name="Hero", aspect=576 / 1024)
    scene.render(os.path.join(rend, f"{stem}.png"), 1024, 16 if FAST else 48, res_y=576)
    # inspect close-up of the head (rim, extractor groove, primer, headstamp) of the lying round
    lens = 100
    scene.camera(d.location, 7.5 * sp["R_rim"] * MM * lens / 36, yaw_deg=50, pitch_deg=18, lens=lens, name="Head")
    scene.render(os.path.join(rend, f"{stem}_head.png"), 640, 16 if FAST else 48)
