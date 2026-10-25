"""Water bottle shared textures (numpy, 2048): mould seams, waist flutes, base ribs + resin code, neck thread,
cap knurl, scuffs. Bottle + cap + liquid share one UV space (one texture set)."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import tex
import bottle_geo as G

S = 2048
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../fonts/WorkSans-Bold.ttf")


def resin_code(rmax):
    """'1 PET' recycling triangle embossed on the base (a standard symbol, not a brand)."""
    n = 512
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    c, R = n / 2, n * 0.5 * (9.0 / rmax)
    pts = [(c + R * math.cos(a), c - R * math.sin(a)) for a in (math.pi / 2, math.pi / 2 + 2.094, math.pi / 2 + 4.189)]
    d.line(pts + [pts[0]], fill=255, width=int(n * 0.012))
    f = ImageFont.truetype(FONT, int(n * 0.05))
    d.text((c, c + R * 0.12), "1", font=f, fill=255, anchor="mm")
    d.text((c, c + R * 0.95), "PET", font=ImageFont.truetype(FONT, int(n * 0.04)), fill=255, anchor="mm")
    return im


def paste_disc(dst, img, rect):
    u0, v0, u1, v1 = rect
    j0, j1, i0, i1 = int(u0 * S), int(u1 * S), int((1 - v1) * S), int((1 - v0) * S)
    dst[i0:i1, j0:j1] = np.maximum(dst[i0:i1, j0:j1], np.asarray(img.resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255)


def build(out, ao=None, seed=11):
    u, v = tex.uv_grid(S)
    H = np.zeros((S, S), np.float32)
    rough = np.full((S, S), 0.05, np.float32)
    base = np.zeros((S, S, 3), np.float32) + 0.92
    mR, mG, mB, wear = (np.zeros((S, S), np.float32) for _ in range(4))
    n1, n2 = tex.fbm(S, S, 10, seed), tex.fbm(S, S, 48, seed + 1)
    px = S / 204.0                                                     # ~10 px/mm (label/cap); body islands ~5

    # lower body: mould seams at 0/180 deg, waist flutes, scuffs at the widest points
    m = tex.in_rect(u, v, G.UV_LOWER)
    th, s, r, z = tex.side_map(u, v, G.UV_LOWER, G.LOWER)
    around = th * r
    seam = sum(np.exp(-((r * np.angle(np.exp(1j * (th - a)))) / 0.25) ** 2) for a in (0, math.pi)) * 0.05
    flutes = (np.cos(th * 12) * 0.5 + 0.5) ** 3 * 0.35 * np.clip(1 - ((z - 95) / 28) ** 2, 0, 1)
    H += np.where(m, seam - flutes, 0)
    scuff = np.clip((n2 - 0.55) * 3, 0, 1) * (np.exp(-((z - 30) / 25) ** 2) + np.exp(-((z - 120) / 8) ** 2))
    rough = np.where(m, 0.05 + 0.25 * scuff + 0.02 * n1, rough)
    wear = np.where(m, scuff, wear)

    # upper body: seams continue, neck thread (2.7 mm pitch, double start), tamper-ring bridges
    m = tex.in_rect(u, v, G.UV_UPPER)
    th, s, r, z = tex.side_map(u, v, G.UV_UPPER, G.UPPER)
    seam = sum(np.exp(-((r * np.angle(np.exp(1j * (th - a)))) / 0.25) ** 2) for a in (0, math.pi)) * 0.05
    neck = (z > 206.5) & (z < 217.5)
    thread = np.clip(np.sin(2 * math.pi * (z / 2.7 + th / math.pi)) * 1.6, -1, 1) * 0.45 * neck
    bridges = (np.cos(th * 16) > 0.85) * ((z > 201.2) & (z < 205.2)) * -0.25
    H += np.where(m, seam + thread + bridges, 0)
    rough = np.where(m, 0.05 + 0.02 * n1, rough)

    # label band: film, printable
    m = tex.in_rect(u, v, G.UV_LABEL)
    base = np.where(m[..., None], 0.88, base)
    mB = np.where(m, 1.0, mB)
    edge = np.clip(1 - np.minimum(v - G.UV_LABEL[1], G.UV_LABEL[3] - v) / 0.01, 0, 1)
    wear = np.where(m, np.clip(edge * (0.4 + n2) + (n2 > 0.8) * 0.5, 0, 1), wear)
    rough = np.where(m, 0.4, rough)

    # base: 6 radial ribs, gate dimple, resin code
    m = tex.in_rect(u, v, G.UV_BASE)
    bx, by = tex.disc_map(u, v, G.UV_BASE, 32.4, flip=True)
    br, ba = np.hypot(bx, by), np.arctan2(by, bx)
    ribs = (np.cos(ba * 6) > 0.92) * ((br > 12) & (br < 23)) * 0.4
    gate = -0.5 * np.exp(-(br / 1.4) ** 2)
    code = np.zeros((S, S), np.float32)
    paste_disc(code, resin_code(32.4).transpose(Image.FLIP_LEFT_RIGHT), G.UV_BASE)
    H += np.where(m, tex.blur(ribs, 1.5) + gate + tex.blur(code, 1.0) * 0.12, 0)
    wear = np.where(m, np.clip(np.exp(-((br - 25) / 2.5) ** 2) * (0.3 + n2), 0, 1), wear)

    # cap: 120 knurl ribs on the skirt, concentric rings on top, plastic (layer A colours it)
    m = tex.in_rect(u, v, G.UV_CAP_SIDE)
    cu = (u - G.UV_CAP_SIDE[0]) / (G.UV_CAP_SIDE[2] - G.UV_CAP_SIDE[0])
    cz = (v - G.UV_CAP_SIDE[1]) / (G.UV_CAP_SIDE[3] - G.UV_CAP_SIDE[1])
    knurl = (np.abs(np.sin(cu * 120 * math.pi)) ** 0.6) * 0.25 * ((cz > 0.1) & (cz < 0.85))
    H += np.where(m, knurl, 0)
    mc = m | tex.in_rect(u, v, G.UV_CAP_TOP) | tex.in_rect(u, v, (0.47, 0.20, 0.49, 0.36))
    mR = np.where(mc, 1.0, mR)
    cx, cy = tex.disc_map(u, v, G.UV_CAP_TOP, G.CAP_R - 1.4 + 1.4)
    crr = np.hypot(cx, cy)
    H += np.where(tex.in_rect(u, v, G.UV_CAP_TOP), 0.1 * (np.abs(crr - 10.5) < 0.35) + 0.08 * (np.abs(crr - 6) < 0.3), 0)
    wear = np.where(mc, np.clip(knurl * 2.5 * (n2 > 0.5), 0, 1) * 0.6, wear)

    nrm = tex.height_to_normal(tex.blur(H, 0.6), px)
    if ao is None:
        ao = np.ones((S, S), np.float32)
    os.makedirs(out, exist_ok=True)
    p = lambda k: os.path.join(out, f"T_WaterBottle_{k}.png")
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(nrm, p("Normal"))
    tex.save(np.stack([ao, np.clip(rough, 0.03, 1), np.zeros_like(ao)], -1), p("ORM"))
    tex.save(np.stack([mR, mG, mB, np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
