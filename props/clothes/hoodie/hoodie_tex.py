"""Texture synthesis for the hoodie (numpy, no bpy). Everything is evaluated per texel from rasterised 3D position,
fabric coordinates (fu across the grain, fv along it, in cm) and the face region, so patterns follow the garment."""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import hoodie_geo as G
from lab import tex

FABRIC = [G.R_TORSO, G.R_SLEEVE, G.R_HOOD, G.R_POCKET, G.R_CHAN]
RIB = [G.R_BAND, G.R_CUFF]
INSIDE = [G.R_INNER, G.R_POCK_IN]
CORD = [G.R_STRING, G.R_KNOT]
METAL = [G.R_GROM, G.R_AGLET]


def isin(reg, lst):
    return np.isin(reg, lst)


def seam_samples(V, edges, step=0.05):
    """Dense points along seam edges with a running arclength (restarting per edge)."""
    pts, ss = [], []
    for a, b in edges:
        p, q = V[a], V[b]
        L = np.linalg.norm(q - p)
        n = max(2, int(L / step))
        t = np.linspace(0, 1, n)
        pts.append(p + (q - p) * t[:, None]); ss.append(t * L + (hash((a, b)) % 97) * 0.13)
    return np.concatenate(pts), np.concatenate(ss)


def seam_field(pos, valid, V, typed, kinds, step=0.05):
    """Distance (cm) to the nearest seam of the given kinds and the arclength there (for stitch dashes)."""
    edges = [e for e, t in typed.items() if t in kinds]
    d = np.full(pos.shape[:2], 99.0, np.float32); s = np.zeros(pos.shape[:2], np.float32)
    if not edges:
        return d, s
    P, S = seam_samples(V, edges, step)
    tree = cKDTree(P)
    q = pos[valid]
    dist, idx = tree.query(q, distance_upper_bound=3.0)
    ok = np.isfinite(dist)
    dd = np.full(len(q), 99.0); dd[ok] = dist[ok]
    sv = np.zeros(len(q)); sv[ok] = S[idx[ok]]
    d[valid] = dd; s[valid] = sv
    return d, s


def stitch(d, s, at, length=0.30, gap=0.12, width=0.055):
    """Raised thread dashes on a line at distance `at` from the seam."""
    on = (np.mod(s, length + gap) < length).astype(np.float32)
    return on * np.exp(-((d - at) / width) ** 2)


def details(R, px_cm, seams, V, wear_pts, seed=7):
    """R: dict of rasters (reg, pos, fu, fv, valid). Returns height (cm), stitch mask, wear 0..1, fuzz 0..1."""
    reg, pos, fu, fv, valid = R["reg"], R["pos"], R["fu"], R["fv"], R["valid"]
    S = reg.shape[0]
    H = np.zeros((S, S), np.float32)
    fab, rib, ins, cord = isin(reg, FABRIC + [G.R_LINING]), isin(reg, RIB), isin(reg, INSIDE), isin(reg, CORD)
    rng = np.random.default_rng(seed)
    fuzz = tex.fbm(S, S, 48, seed) * 0.6 + tex.fbm(S, S, 220, seed + 1) * 0.4
    # jersey knit (outer fabric): wales along the grain, staggered courses -> fine V texture, plus brushed fuzz
    w_p, c_p = 0.34, 0.28
    col = np.floor(fu / w_p)
    wale = np.abs(np.sin(np.pi * fu / w_p)) ** 0.6
    course = 0.5 + 0.5 * np.cos(2 * np.pi * (fv / c_p + 0.5 * np.mod(col, 2)) + 1.2 * (np.mod(fu / w_p, 1) - 0.5))
    knit = 0.6 * wale + 0.4 * wale * course
    H += fab * (0.004 * knit + 0.004 * fuzz)          # heavy fleece: knit not visible at viewing distance
    # 2x1 ribbing on band and cuffs: raised ribs along the grain
    rp = 0.46
    ph = np.mod(fu / rp, 1.0)
    ribh = np.clip(np.cos(2 * np.pi * (ph - 0.33)) * 1.2, -1, 1) * 0.5 + 0.5
    H += rib * (0.050 * ribh ** 0.8 + 0.006 * course + 0.004 * fuzz)
    # brushed fleece inside: soft random loft
    H += ins * 0.025 * tex.fbm(S, S, 160, seed + 2)
    # cords: twisted / flat-braided
    H += cord * 0.03 * (0.5 + 0.5 * np.sin(2 * np.pi * (fv / 0.22 + fu / 0.35)))
    # seams: crease + allowance ridge; topstitches by seam kind
    stitches = np.zeros((S, S), np.float32)
    outer_like = valid & ~isin(reg, METAL + CORD)
    body_reg = outer_like & ~isin(reg, [G.R_POCKET, G.R_POCK_IN])
    d, s = seam_field(pos, body_reg, V, seams["typed"], {"side", "cover", "rib"})
    H += body_reg * (-0.075 * np.exp(-(d / 0.16) ** 2) + 0.03 * np.exp(-((d - 0.45) / 0.28) ** 2))
    dc, sc = seam_field(pos, body_reg, V, seams["typed"], {"cover"})
    st = stitch(dc, sc, 0.42) + stitch(dc, sc, 0.72)
    dr, sr = seam_field(pos, body_reg, V, seams["typed"], {"rib"})
    st += stitch(dr, sr, 0.35, 0.22, 0.10) + stitch(dr, sr, 0.62, 0.22, 0.10)
    dh, shh = seam_field(pos, body_reg, V, seams["typed"], {"chan"})
    st += stitch(dh, shh, 0.0) + stitch(dh, shh, 0.3)
    H += body_reg * (-0.03 * np.exp(-(dh / 0.12) ** 2))
    pk = valid & isin(reg, [G.R_POCKET, G.R_POCK_IN, G.R_RIM])
    dp, sp = seam_field(pos, pk, seams["pV"], seams["pk_typed"], {"pocket"})
    st += pk * (stitch(dp, sp, 0.45) + stitch(dp, sp, 0.75))
    do, so = seam_field(pos, pk, seams["pV"], seams["pk_typed"], {"opening"})
    st += pk * stitch(do, so, 1.05)
    H += pk * (0.06 * np.exp(-((do - 0.5) / 0.45) ** 2) - 0.03 * np.exp(-((do - 1.05) / 0.2) ** 2))
    st = np.clip(st, 0, 1) * outer_like
    H += 0.045 * st
    stitches += st
    # wear: openings and edges rub first, elbows and pocket edges pill, plus fuzz noise
    wear = np.zeros((S, S), np.float32)
    for P, rad, amt in wear_pts:
        tr = cKDTree(P)
        dist, _ = tr.query(pos[valid], distance_upper_bound=rad * 4)
        w = np.zeros(int(valid.sum())); ok = np.isfinite(dist); w[ok] = np.exp(-(dist[ok] / rad) ** 2) * amt
        tmp = np.zeros((S, S), np.float32); tmp[valid] = w
        wear = np.maximum(wear, tmp)
    wear = np.clip(wear * (0.5 + 0.8 * tex.fbm(S, S, 30, seed + 3)) + 0.15 * tex.remap(tex.fbm(S, S, 90, seed + 4), 0.66, 0.9), 0, 1)
    pill = tex.speckle(S, S, 0.004, seed + 5, 0.8) * wear
    H += fab * 0.02 * pill
    return H, stitches, wear, fuzz


def chest_print(pos, reg, valid, S, font_dir):
    """Generic varsity-style chest print (no brand): an arc of 'ATHLETIC DEPT.' over a big '07', cracked ink.
    Planar projection from the front onto the torso front. Returns RGBA float (S,S,4) and the print-area mask."""
    W, Hh, ppc = 26.0, 16.0, 40                    # print box 26 x 16 cm at 40 px/cm
    cz, cx = 131.0, 0.0
    im = Image.new("L", (int(W * ppc), int(Hh * ppc)), 0)
    d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(os.path.join(font_dir, "BigShoulders-Bold.ttf"), int(3.0 * ppc))
    fn = ImageFont.truetype(os.path.join(font_dir, "BigShoulders-Bold.ttf"), int(9.5 * ppc))
    txt = "ATHLETIC DEPT."
    R = 30 * ppc; cxp, cyp = W * ppc / 2, R + 1.4 * ppc
    widths = [d.textlength(ch, font=fb) for ch in txt]
    total = sum(widths) + 0.25 * ppc * (len(txt) - 1)
    ang0 = -total / 2 / R
    acc = 0
    for ch, w_ in zip(txt, widths):
        a = ang0 + (acc + w_ / 2) / R
        g = Image.new("L", (int(w_ + 20), int(4 * ppc)), 0)
        ImageDraw.Draw(g).text((10, 0), ch, font=fb, fill=255)
        g = g.rotate(-math.degrees(a), resample=Image.BICUBIC, expand=True)
        x = cxp + R * math.sin(a) - g.width / 2; y = cyp - R * math.cos(a) - g.height / 2 + 0.6 * ppc
        im.paste(255, (int(x), int(y)), g)
        acc += w_ + 0.25 * ppc
    d.text((W * ppc / 2, 9.6 * ppc), "07", font=fn, fill=255, anchor="mm")
    d.rectangle((3 * ppc, 15.2 * ppc, (W - 3) * ppc, 15.5 * ppc), fill=255)
    a = np.asarray(im, np.float32) / 255
    # crack the ink: fine network of gaps + worn patches
    h, w = a.shape
    cr = tex.fbm(h, w, 70, 11)
    cracks = np.abs(cr - 0.5) < 0.012
    worn = tex.remap(tex.fbm(h, w, 12, 12), 0.62, 0.8)
    a = a * (1 - cracks * 0.9) * (1 - 0.75 * worn)
    x, z = pos[..., 0], pos[..., 2]
    front = valid & (reg == G.R_TORSO) & (pos[..., 1] < -2.0)
    px = ((x - cx) + W / 2) * ppc; py = (cz + Hh / 2 - z) * ppc
    inside = front & (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
    alpha = np.zeros((S, S), np.float32)
    alpha[inside] = a[py[inside].astype(int), px[inside].astype(int)]
    rgba = np.zeros((S, S, 4), np.float32)
    rgba[..., :3] = np.array(tex.hex2rgb("#EDE8DC"))
    rgba[..., 3] = alpha
    area = front & (np.abs(x - cx) < W / 2 + 1) & (np.abs(z - cz) < Hh / 2 + 1)
    return rgba, area.astype(np.float32)
