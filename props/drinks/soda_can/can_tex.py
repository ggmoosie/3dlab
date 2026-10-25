"""Soda can shared texture set (numpy): BaseColor, Normal, ORM, Mask at 2048. Analytic from the lathe profile:
every texel knows where it sits on the can (mm), so dents, neck rings, lid embossing, score lines and wear are
placed in real units. AO and the tab/flap edge normals come from bakes (passed in)."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import tex
import can_geo as G

S = 2048
PX_MM = S / (2 * math.pi * 33.1)                  # side island density (~9.85 px/mm)


def grids():
    j, i = np.meshgrid(np.arange(S), np.arange(S))
    u, v = (j + 0.5) / S, 1 - (i + 0.5) / S
    return u.astype(np.float32), v.astype(np.float32)


def rect_mask(u, v, r):
    return (u >= r[0]) & (u < r[2]) & (v >= r[1]) & (v < r[3])


def side_coords(u, v):
    acc = np.array(G.side_arc())
    u0, v0, u1, v1 = G.UV_SIDE
    s = (v - v0) / (v1 - v0) * acc[-1]
    rr = np.interp(s, acc, [p[0] for p in G.SIDE])
    zz = np.interp(s, acc, [p[1] for p in G.SIDE])
    x = (u - u0) / (u1 - u0) * 2 * math.pi * 33.1        # mm around the circumference
    return x, s, rr, zz, acc


def disc_coords(u, v, rect, rmax, flip=False):
    u0, v0, u1, v1 = rect
    x = ((u - u0) / (u1 - u0) - 0.5) * 2 * rmax * (-1 if flip else 1)
    y = ((v - v0) / (v1 - v0) - 0.5) * 2 * rmax
    return x, y


def profile_curv(prof, acc_like=None):
    """Signed turning angle per unit length along a profile → convexity (>0 convex) at each point."""
    p = np.array(prof, float)
    d = np.diff(p, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.diff(ang)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    k = np.zeros(len(p))
    k[1:-1] = -turn                                   # outward-convex corners positive (profile runs bottom→top)
    return k


def lid_text(draw_size, R):
    """Embossed ring text on the lid (deposit marks are generic text, no logos)."""
    im = Image.new("L", (draw_size, draw_size), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(os.path.join(tex.__file__.rsplit("/lab/", 1)[0], "fonts", "WorkSans-Bold.ttf"), int(draw_size * 0.034))
    msg = "RECYCLE ME  •  NY VT ME CT MA 5¢  •  MI 10¢  •  ALUMINUM  •  "
    c = draw_size / 2
    rad = draw_size / 2 * (19.8 / R)
    for k, ch in enumerate(msg):
        a = math.pi / 2 + 0.5 - k * (2 * math.pi * 0.70) / len(msg)
        x, y = c + rad * math.cos(a), c - rad * math.sin(a)
        g = Image.new("L", (60, 60), 0)
        ImageDraw.Draw(g).text((30, 30), ch, font=f, fill=255, anchor="mm")
        g = g.rotate(math.degrees(a - math.pi / 2), resample=Image.BICUBIC)
        im.paste(255, (int(x - 30), int(y - 30)), g)
    return np.asarray(im, np.float32) / 255


def build(out, ao=None, bake_n=None, seed=7):
    """ao: (S,S) baked AO or None. bake_n: (S,S,3) baked tab/flap normals in 0..1 or None."""
    u, v = grids()
    m_side, m_lid, m_bot = rect_mask(u, v, G.UV_SIDE), rect_mask(u, v, G.UV_LID), rect_mask(u, v, G.UV_BOTTOM)
    m_tab, m_flap, m_dark = rect_mask(u, v, G.UV_TAB), rect_mask(u, v, G.UV_FLAP), rect_mask(u, v, G.UV_DARK)
    H = np.zeros((S, S), np.float32)                  # height in mm
    rough = np.full((S, S), 0.32, np.float32)
    base = np.zeros((S, S, 3), np.float32) + 0.91
    wear = np.zeros((S, S), np.float32)
    maskR, maskG, maskB = np.zeros((S, S), np.float32), np.zeros((S, S), np.float32), np.zeros((S, S), np.float32)
    n1 = tex.fbm(S, S, 10, seed)
    n2 = tex.fbm(S, S, 40, seed + 1)

    # ---------------- side: drawn-aluminium striations, neck rings, soft dents, rub bands, scratches
    x, s, rr, zz, acc = side_coords(u, v)
    stri = tex.spectral(S, S, 2, 400, 1.2, seed + 2, ax=0.03)                         # long vertical lines
    neck = (zz > 104) & (zz < 117)
    rings = np.sin((zz - 104) / 13 * 6 * 2 * math.pi) * 0.012 * neck
    dents = np.zeros_like(H)
    rng = tex.RNG(seed)
    for _ in range(5):                                                                 # a few soft dings
        cx, cz, rad, dep = rng.uniform(0, 208), rng.uniform(25, 95), rng.uniform(4, 11), rng.uniform(0.08, 0.22)
        dx = np.minimum(np.abs(x - cx), 208 - np.abs(x - cx))
        dents -= dep * np.exp(-(dx ** 2 + (zz - cz) ** 2) / (2 * rad ** 2))
    crease = np.exp(-((x - 150) * 0.9 - (zz - 70) * 0.25) ** 2 / 0.8) * np.exp(-((zz - 70) / 9) ** 2) * 0.05   # a fine crimp
    H += np.where(m_side, (stri - 0.5) * 0.004 + rings + dents - crease, 0)
    rough = np.where(m_side, 0.24 + 0.08 * n1 + 0.05 * (stri - 0.5), rough)
    k_side = np.interp(s, acc, np.clip(profile_curv(G.SIDE), 0, None))
    band = np.exp(-((zz - 14) / 4.5) ** 2) + np.exp(-((zz - 101) / 4.5) ** 2)             # case-rub bands
    sc = tex.scratches(S, S, 70, (0.003, 0.014), 1, seed + 3, angle=0.0)
    w_side = np.clip(band * (n2 - 0.45) * 2.0, 0, 1) + sc * 0.55 * (n1 > 0.6) + np.clip(k_side * 3, 0, 1) * (0.3 + 0.7 * n2)
    wear = np.where(m_side, np.clip(w_side, 0, 1), wear)
    maskB = np.where(m_side & (zz > 4.0) & (zz < 117.0), 1.0, maskB)                    # printable area

    # ---------------- lid: rivet, emboss ring, score, deposit text, radial machining
    lx, ly = disc_coords(u, v, G.UV_LID, G.LID_RMAX)
    lr = np.hypot(lx, ly)
    rivet = np.clip(1 - (lr / 2.3) ** 2, 0, 1) ** 0.5 * 0.45
    oc = np.array(G.opening(96))
    dpoly = tex.dist_to_poly(lx, ly, oc)
    score = -0.06 * np.exp(-((dpoly - 0.55) / 0.12) ** 2)
    bead = 0.12 * np.exp(-((dpoly - 2.4) / 0.45) ** 2) * (lr < 23.5)
    txt = tex.blur(lid_text(S, G.LID_RMAX), 0.6)
    t_full = np.zeros((S, S), np.float32)
    u0, v0, u1, v1 = G.UV_LID
    j0, j1, i0, i1 = int(u0 * S), int(u1 * S), int((1 - v1) * S), int((1 - v0) * S)
    t_full[i0:i1, j0:j1] = np.asarray(Image.fromarray((txt * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    radial = (tex.spectral(S, S, 50, 600, 1.0, seed + 5) - 0.5) * 0.003
    lidH = rivet + score + bead + t_full * 0.05 + radial
    H += np.where(m_lid & (lr < 24.6), lidH, 0)
    base = np.where((m_lid & (lr < 24.6))[..., None], 0.86, base)
    rough = np.where(m_lid, 0.30 + 0.06 * n1, rough)
    k_lid = np.interp(lr, [p[0] for p in G.LID][::-1], np.clip(-profile_curv(G.LID), 0, None)[::-1])
    wear = np.where(m_lid, np.clip(k_lid * 4, 0, 1) * (0.4 + n2), wear)
    maskG = np.where(m_lid, 1.0, maskG)                                                   # lid tint region

    # ---------------- bottom: stand-ring scuffs + dot-matrix date code (ink, part of the substrate)
    bx, by = disc_coords(u, v, G.UV_BOTTOM, 28.0, flip=True)
    br = np.hypot(bx, by)
    ring_scuff = np.exp(-((br - 25.4) / 0.6) ** 2) * (0.5 + n2)
    rough = np.where(m_bot, 0.36 + 0.25 * ring_scuff, rough)
    wear = np.where(m_bot, np.clip(ring_scuff, 0, 1), wear)
    code = date_code(S, G.UV_BOTTOM, seed)
    base = np.where((m_bot & (code > 0.5))[..., None], 0.03, base)

    # ---------------- tab + flap: baked edge normals, tint region R; dark interior
    maskR = np.where(m_tab, 1.0, maskR)
    rough = np.where(m_tab | m_flap, 0.30 + 0.08 * n1, rough)
    wear = np.where(m_tab, np.clip((n2 - 0.6) * 3, 0, 1), wear)
    base = np.where(m_dark[..., None], 0.02, base)
    rough = np.where(m_dark, 0.55, rough)
    metal = np.where(m_dark, 0.0, 1.0).astype(np.float32)

    # ---------------- compose
    nrm = tex.height_to_normal(H, PX_MM)
    if bake_n is not None:
        sel = (m_tab | m_flap)[..., None]
        nrm = np.where(sel, tex.blend_normals(bake_n, tex.height_to_normal(n2 * 0.003, PX_MM)), nrm)
    if ao is None:
        ao = np.ones((S, S), np.float32)
    base = base * (0.97 + 0.03 * n1[..., None])
    os.makedirs(out, exist_ok=True)
    p = lambda n: os.path.join(out, f"T_SodaCan_{n}.png")
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(nrm, p("Normal"))
    tex.save(np.stack([ao, np.clip(rough, 0.05, 1), metal], -1), p("ORM"))
    tex.save(np.stack([maskR, maskG, maskB, np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def date_code(size, rect, seed):
    """Dot-matrix print on the dome: lot and time code, like real can bottoms."""
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    u0, v0, u1, v1 = rect
    cx, cy = (u0 + u1) / 2 * size, (1 - (v0 + v1) / 2) * size
    rng = tex.RNG(seed)
    lines = [f"L{rng.integers(1, 9)} {rng.integers(10, 23)}:{rng.integers(10, 59)}  {rng.integers(1000, 9999)}", "BEST BY 11 2026"]
    dot = max(1, int(size * 0.0011))
    for li, s in enumerate(lines):
        for ci, ch in enumerate(s):
            glyph = Image.new("L", (5, 7), 0)
            ImageDraw.Draw(glyph).text((0, -1), ch, fill=255)
            g = np.asarray(glyph) > 60
            for gy in range(7):
                for gx in range(5):
                    if g[gy, gx]:
                        x = cx + (ci - len(s) / 2) * 6.2 * dot * 1.6 + gx * dot * 1.6
                        y = cy - 30 * dot + li * 11 * dot * 1.6 + gy * dot * 1.6
                        d.ellipse([x, y, x + dot, y + dot], fill=255)
    return np.asarray(im, np.float32) / 255
