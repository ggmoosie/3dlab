"""Glass bottle shared textures (2048): glass seams, crate scuff band, stippled heel, waviness; label paper; neck foil;
crown cap paint + steel wear. One set shared by glass, decals, liquid and cap."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import tex
import glass_geo as G

S = 2048


def build(out, ao=None, seed=21):
    u, v = tex.uv_grid(S)
    H = np.zeros((S, S), np.float32)
    rough = np.full((S, S), 0.03, np.float32)
    base = np.zeros((S, S, 3), np.float32) + 0.9
    metal = np.zeros((S, S), np.float32)
    mR, mG, mB, wear = (np.zeros((S, S), np.float32) for _ in range(4))
    n1, n2 = tex.fbm(S, S, 10, seed), tex.fbm(S, S, 50, seed + 1)

    m = tex.in_rect(u, v, G.UV_SIDE)                          # glass
    th, s, r, z = tex.side_map(u, v, G.UV_SIDE, G.SIDE)
    seam = sum(np.exp(-((r * np.angle(np.exp(1j * (th - a)))) / 0.3) ** 2) for a in (0, math.pi)) * 0.06
    wavy = (tex.fbm(S, S, 6, seed + 2) - 0.5) * 0.08
    band = np.exp(-((z - 14) / 4.0) ** 2) + np.exp(-((z - 146) / 4.0) ** 2)      # crate scuff rings
    scuff = np.clip(band * (0.3 + 1.2 * n2) * (n1 > 0.35), 0, 1)
    stip = (np.cos(th * 60) * np.cos(z * 2.5) > 0.6) * (z < 8) * 0.12
    H += np.where(m, seam + wavy + stip - scuff * 0.02, 0)
    rough = np.where(m, 0.03 + 0.55 * scuff + 0.02 * n1, rough)

    m = tex.in_rect(u, v, G.UV_BASE)                          # base push-up: stipple ring + gate
    bx, by = tex.disc_map(u, v, G.UV_BASE, 31.5, flip=True)
    br = np.hypot(bx, by)
    H += np.where(m, (np.cos(bx * 3) * np.cos(by * 3) > 0.5) * ((br > 25) & (br < 30)) * 0.15, 0)
    rough = np.where(m, 0.03 + 0.5 * np.exp(-((br - 27) / 2) ** 2) * n2, rough)

    m = tex.in_rect(u, v, G.UV_LABEL)                         # paper: grain, wrinkle, edge wear
    lu = (u - G.UV_LABEL[0]) / (G.UV_LABEL[2] - G.UV_LABEL[0]); lv = (v - G.UV_LABEL[1]) / (G.UV_LABEL[3] - G.UV_LABEL[1])
    grain = tex.spectral(S, S, 200, 900, 1.0, seed + 3)
    wrinkle = tex.spectral(S, S, 3, 30, 2.4, seed + 4, ax=0.4)
    H += np.where(m, (grain - 0.5) * 0.01 + (wrinkle - 0.5) * 0.05, 0)
    edge = np.clip(1 - np.minimum.reduce([lu, 1 - lu, lv, 1 - lv]) / 0.025, 0, 1)
    base = np.where(m[..., None], 0.86, base)
    rough = np.where(m, 0.62, rough)
    mB = np.where(m, 1.0, mB)
    wear = np.where(m, np.clip(edge * (0.5 + n2) + (n2 > 0.82) * 0.6, 0, 1), wear)

    m = tex.in_rect(u, v, G.UV_NECK)                          # foil: crinkled metal, tinted by layer B
    crinkle = tex.spectral(S, S, 20, 160, 1.6, seed + 5)
    H += np.where(m, (crinkle - 0.5) * 0.05, 0)
    mG = np.where(m, 1.0, mG); mB = np.where(m, 1.0, mB)
    base = np.where(m[..., None], 0.8, base); metal = np.where(m, 1.0, metal); rough = np.where(m, 0.3, rough)
    nv = (v - G.UV_NECK[1]) / (G.UV_NECK[3] - G.UV_NECK[1])
    wear = np.where(m, np.clip((np.minimum(nv, 1 - nv) < 0.04) * (0.6 + n2) + (crinkle > 0.7) * 0.4, 0, 1), wear)

    mc = tex.in_rect(u, v, G.UV_CAP_SIDE) | tex.in_rect(u, v, G.UV_CAP_TOP) | tex.in_rect(u, v, G.UV_CAP_IN)
    base = np.where(mc[..., None], 0.62, base); metal = np.where(mc, 1.0, metal); rough = np.where(mc, 0.35, rough)
    mR = np.where(tex.in_rect(u, v, G.UV_CAP_SIDE) | tex.in_rect(u, v, G.UV_CAP_TOP), 1.0, mR)
    mB = np.where(tex.in_rect(u, v, G.UV_CAP_TOP), 1.0, mB)
    cu = (u - G.UV_CAP_SIDE[0]) / (G.UV_CAP_SIDE[2] - G.UV_CAP_SIDE[0])
    flute_edge = (np.abs(np.cos(cu * 21 * 2 * math.pi)) > 0.985) * tex.in_rect(u, v, G.UV_CAP_SIDE) * (n2 > 0.55)
    cx, cy = tex.disc_map(u, v, G.UV_CAP_TOP, 17.0)
    rim = np.clip((np.hypot(cx, cy) - 13.5) / 1.5, 0, 1) * tex.in_rect(u, v, G.UV_CAP_TOP)
    wear = np.where(mc, np.clip(flute_edge * (0.3 + n2) + rim * np.clip((n2 - 0.35) * 2, 0, 1) * 0.9, 0, 1), wear)
    H += np.where(tex.in_rect(u, v, G.UV_CAP_TOP), -0.08 * (np.abs(np.hypot(cx, cy) - 13.0) < 0.4), 0)

    mi = tex.in_rect(u, v, G.UV_MOUTH)
    base = np.where(mi[..., None], 0.02, base)

    nrm = tex.height_to_normal(tex.blur(H, 0.5), S / 200.0)
    if ao is None:
        ao = np.ones((S, S), np.float32)
    os.makedirs(out, exist_ok=True)
    p = lambda k: os.path.join(out, f"T_GlassBottle_{k}.png")
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(nrm, p("Normal"))
    tex.save(np.stack([ao, np.clip(rough, 0.02, 1), metal], -1), p("ORM"))
    tex.save(np.stack([mR, mG, mB, np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
