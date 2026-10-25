"""Food can shared textures (2048): tinplate, beads (normal only), welded side seam, label paper area, gold lacquer
inside, chunky food relief, lid score + PULL emboss, bottom expansion rings + date code, light rust at the bottom seam."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import tex
import food_can_geo as G

S = 2048
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../fonts/WorkSans-Bold.ttf")
LABEL_Z = (8.0, 103.0)
_r = np.random.default_rng(31)
CHUNKS = [(*_r.uniform(-36, 36, 2), _r.uniform(0, np.pi), _r.uniform(3, 7), _r.uniform(2, 4)) for _ in range(140)]


def emboss_text(rect, rmax, text, y_mm, size_mm):
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    u0, v0, u1, v1 = rect
    ppm = (u1 - u0) * S / (2 * rmax)
    cx, cy = (u0 + u1) / 2 * S, (1 - (v0 + v1) / 2) * S - y_mm * ppm
    d.text((cx, cy), text, font=ImageFont.truetype(FONT, int(size_mm * ppm)), fill=255, anchor="mm")
    return np.asarray(im, np.float32) / 255


def build(out, ao=None, seed=31):
    u, v = tex.uv_grid(S)
    H = np.zeros((S, S), np.float32)
    rough = np.full((S, S), 0.3, np.float32)
    base = np.zeros((S, S, 3), np.float32) + np.array([0.70, 0.70, 0.72])
    metal = np.ones((S, S), np.float32)
    mR, mG, mB, wear = (np.zeros((S, S), np.float32) for _ in range(4))
    n1, n2 = tex.fbm(S, S, 10, seed), tex.fbm(S, S, 50, seed + 1)
    rust_col = np.array([0.30, 0.13, 0.05])

    m = tex.in_rect(u, v, G.UV_SIDE)
    th, s, r, z = tex.side_map(u, v, G.UV_SIDE, G.SIDE)
    beads = sum(-0.9 * np.exp(-((z - b) / 0.9) ** 2) for b in G.BEADS)
    seam = 0.25 * np.exp(-((r * np.angle(np.exp(1j * (th - 0.0)))) / 0.6) ** 2) * (z > 6) * (z < 105)
    lab_z = (z > LABEL_Z[0]) & (z < LABEL_Z[1])
    H += np.where(m, beads * np.where(lab_z, 0.35, 1.0) + seam, 0)
    lab = (z > LABEL_Z[0]) & (z < LABEL_Z[1])
    mB = np.where(m & lab, 1.0, mB)
    bead_rub = sum(np.exp(-((z - b) / 1.4) ** 2) for b in G.BEADS) * 0.0
    rim = np.clip(np.exp(-((z - 2.5) / 2.0) ** 2) + np.exp(-((z - 109) / 2.5) ** 2), 0, 1)
    wear = np.where(m, np.clip(rim * (0.4 + n2) + (lab & ((z < 12) | (z > 99))) * (n2 > 0.6) * 0.7 + bead_rub, 0, 1), wear)
    rust = np.clip((np.exp(-((z - 2) / 3) ** 2) * (n1 * 1.6 - 0.55)), 0, 1)
    base = np.where(m[..., None], base * (1 - rust[..., None]) + rust_col * rust[..., None], base)
    rough = np.where(m, 0.28 + 0.5 * rust + 0.05 * n1, rough)
    metal = np.where(m, 1 - rust, metal)

    m = tex.in_rect(u, v, G.UV_INNER)                       # lacquered rim + inner wall (layer B)
    mG = np.where(m, 1.0, mG)
    th, s, r, z = tex.side_map(u, v, G.UV_INNER, G.INNER)
    wear = np.where(m, np.clip((z > 110) * (0.5 + n2), 0, 1), wear)

    m = tex.in_rect(u, v, G.UV_FOOD)                        # contents: lumps + sauce, colour from the variant print
    blobs = np.zeros((S, S), np.float32)
    fx, fy = tex.disc_map(u, v, G.UV_FOOD, 37.4)
    for cx, cy, a, rx, ry in CHUNKS:
        dx, dy = (fx - cx) * np.cos(a) + (fy - cy) * np.sin(a), -(fx - cx) * np.sin(a) + (fy - cy) * np.cos(a)
        blobs = np.maximum(blobs, np.clip(1 - (dx / rx) ** 2 - (dy / ry) ** 2, 0, 1) ** 0.5 * 1.4)
    H += np.where(m, blobs + (tex.fbm(S, S, 30, seed + 4) - 0.5) * 0.6, 0)
    mB = np.where(m, 1.0, mB)
    base = np.where(m[..., None], 0.3, base); metal = np.where(m, 0.0, metal)
    rough = np.where(m, 0.18 + 0.25 * (blobs > 0.3), rough)               # glossy sauce, duller chunks

    m = tex.in_rect(u, v, G.UV_BOTTOM)
    bx, by = tex.disc_map(u, v, G.UV_BOTTOM, 37.5, flip=True)
    br = np.hypot(bx, by)
    H += np.where(m, 0.25 * np.exp(-((br - 17) / 1.2) ** 2) - 0.2 * np.exp(-((br - 27) / 1.0) ** 2), 0)
    code = emboss_text(G.UV_BOTTOM, 37.5, "B2 11 2026 P3", 4.0, 2.2)
    base = np.where((m & (code > 0.5))[..., None], 0.04, base)
    wear = np.where(m, np.clip(np.exp(-((br - 36.5) / 1.2) ** 2) * (0.5 + n2), 0, 1), wear)

    m = tex.in_rect(u, v, G.UV_LID)                          # lid: score groove near the edge, PULL emboss
    lx, ly = tex.disc_map(u, v, G.UV_LID, 36.4)
    lr = np.hypot(lx, ly)
    pull = tex.blur(emboss_text(G.UV_LID, 36.4, "PULL ▲", -14.0, 4.5), 1.0)
    H += np.where(m, -0.12 * np.exp(-((lr - 35.0) / 0.25) ** 2) + 0.15 * pull, 0)
    rough = np.where(m, 0.25 + 0.06 * n1, rough)
    wear = np.where(m, np.clip(np.exp(-((lr - 36) / 0.8) ** 2) * (0.4 + n2), 0, 1), wear)
    mG = np.where(tex.in_rect(u, v, (0.57, 0.42, 0.79, 0.50)), 1.0, mG)       # lid underside lacquered
    rough = np.where(tex.in_rect(u, v, G.UV_RING), 0.25, rough)

    nrm = tex.height_to_normal(tex.blur(H, 0.6), S / 239.0)
    if ao is None:
        ao = np.ones((S, S), np.float32)
    os.makedirs(out, exist_ok=True)
    p = lambda k: os.path.join(out, f"T_FoodCan_{k}.png")
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(nrm, p("Normal"))
    tex.save(np.stack([ao, np.clip(rough, 0.05, 1), metal], -1), p("ORM"))
    tex.save(np.stack([mR, mG, mB, np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
